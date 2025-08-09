#!/usr/bin/env python3
"""
개선된 생성 모델 - 모든 최적화 통합
- 반복 제거 필터
- 최적화된 생성 파라미터
- 개선된 프롬프트 템플릿
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append("..")
from utils.inference_filter import GenerationFilter
import warnings
warnings.filterwarnings("ignore")

class ImprovedMinwonGenerator:
    """개선된 민원 생성 모델"""
    
    def __init__(self):
        """초기화"""
        self.model_path = "../ai-service/models/lora_gen_4bit"
        self.model_id = "openchat/openchat-3.5-0106"
        self.filter = GenerationFilter()
        self.model = None
        self.tokenizer = None
        
        # 최적 생성 설정 (테스트 결과 기반)
        self.generation_config = {
            "max_new_tokens": 100,
            "temperature": 0.4,
            "do_sample": True,
            "top_p": 0.75,
            "top_k": 30,
            "repetition_penalty": 2.5,
            "no_repeat_ngram_size": 4,
            "encoder_repetition_penalty": 1.0,
        }
        
    def load_model(self):
        """모델 로드"""
        print("개선된 생성 모델 로딩 중...")
        
        # 4-bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "6GB", "cpu": "10GB"},
            offload_folder="offload",
        )
        
        # LoRA 어댑터 로드
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        print("모델 로드 완료\n")
    
    def create_prompt(self, instruction: str, input_text: str, use_few_shot: bool = False) -> str:
        """
        프롬프트 생성
        
        Args:
            instruction: 지시사항
            input_text: 입력 텍스트
            use_few_shot: Few-shot 예시 사용 여부
            
        Returns:
            생성된 프롬프트
        """
        if use_few_shot:
            # Few-shot 프롬프트
            prompt = f"""GPT4 Correct User: 다음 민원을 요약해주세요.

입력: 도로에 포트홀이 생겨서 위험합니다. 빠른 조치 바랍니다.

간단하고 명확하게 한국어로 답변해주세요.<|end_of_turn|>GPT4 Correct Assistant: 도로 포트홀로 인한 안전 문제 민원입니다. 신속한 보수 작업이 필요합니다.<|end_of_turn|>GPT4 Correct User: {instruction}

입력: {input_text}

간단하고 명확하게 한국어로 답변해주세요. 100자 이내로 작성하세요.<|end_of_turn|>GPT4 Correct Assistant:"""
        else:
            # 기본 프롬프트
            prompt = f"""GPT4 Correct User: {instruction}

입력: {input_text}

간단하고 명확하게 한국어로 답변해주세요. 100자 이내로 작성하세요.<|end_of_turn|>GPT4 Correct Assistant:"""
        
        return prompt
    
    def generate(self, instruction: str, input_text: str, use_filter: bool = True) -> dict:
        """
        텍스트 생성
        
        Args:
            instruction: 지시사항
            input_text: 입력 텍스트
            use_filter: 필터 사용 여부
            
        Returns:
            생성 결과 딕셔너리
        """
        if not self.model:
            self.load_model()
        
        # 프롬프트 생성
        prompt = self.create_prompt(instruction, input_text, use_few_shot=True)
        
        # 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.generation_config
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_output = response.split("GPT4 Correct Assistant:")[-1].strip()
        
        # 필터 적용
        if use_filter:
            filtered_output = self.filter.filter(raw_output, max_length=150)
        else:
            filtered_output = raw_output
        
        # 품질 검증
        is_valid = self.filter.validate_korean(filtered_output)
        
        return {
            "instruction": instruction,
            "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
            "filtered_output": filtered_output,
            "is_valid": is_valid,
            "token_count": len(self.tokenizer.encode(filtered_output))
        }
    
    def batch_generate(self, test_cases: list) -> list:
        """
        배치 생성
        
        Args:
            test_cases: 테스트 케이스 리스트
            
        Returns:
            생성 결과 리스트
        """
        results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n테스트 {i}/{len(test_cases)}")
            result = self.generate(case["instruction"], case["input"])
            results.append(result)
            
            print(f"📝 지시: {result['instruction']}")
            print(f"📥 입력: {result['input']}")
            print(f"📤 생성: {result['filtered_output']}")
            print(f"✅ 유효: {result['is_valid']}")
            print(f"📊 토큰: {result['token_count']}")
        
        return results


def main():
    """메인 함수"""
    print("=== 개선된 생성 모델 테스트 ===\n")
    
    # 생성기 초기화
    generator = ImprovedMinwonGenerator()
    generator.load_model()
    
    # 테스트 케이스
    test_cases = [
        {
            "instruction": "다음 민원 내용을 간단히 요약해주세요.",
            "input": "안녕하세요. 저는 oo동에 거주하는 주민입니다. 최근 우리 동네 공원에 쓰레기가 많이 쌓여 있어 불편을 겪고 있습니다. 특히 주말이 지나면 음식물 쓰레기와 일회용품이 곳곳에 버려져 있어 악취가 나고 미관상 좋지 않습니다. 구청에서 정기적인 청소와 쓰레기통 추가 설치를 검토해 주시기 바랍니다."
        },
        {
            "instruction": "다음 민원에 대한 답변을 작성해주세요.",
            "input": "횡단보도 신호등 시간이 너무 짧아서 노인분들이 건너기 어렵습니다. 신호 시간을 늘려주세요."
        },
        {
            "instruction": "다음 민원의 핵심 요구사항을 정리해주세요.",
            "input": "우리 아파트 앞 도로에서 밤마다 오토바이 소음이 심합니다. 특히 새벽 시간대에 굉음을 내며 지나가는 오토바이 때문에 잠을 설치는 날이 많습니다. 단속을 강화해 주시고, 방지턱이나 CCTV 설치도 검토해 주세요."
        },
        {
            "instruction": "다음 민원을 분석해주세요.",
            "input": "주차장이 부족해서 불법주차가 많습니다. 주차공간을 늘려주세요."
        },
        {
            "instruction": "다음 민원에 대한 처리 방안을 제시해주세요.",
            "input": "공원에 가로등이 없어서 밤에 위험합니다."
        }
    ]
    
    # 배치 생성
    print("\n📋 배치 생성 시작")
    print("=" * 60)
    results = generator.batch_generate(test_cases)
    
    # 결과 요약
    print("\n\n" + "=" * 60)
    print("📊 결과 요약")
    print("=" * 60)
    
    valid_count = sum(1 for r in results if r['is_valid'])
    avg_tokens = sum(r['token_count'] for r in results) / len(results)
    
    print(f"\n총 테스트: {len(results)}개")
    print(f"유효 생성: {valid_count}개 ({valid_count/len(results)*100:.1f}%)")
    print(f"평균 토큰: {avg_tokens:.1f}개")
    
    # 문제 케이스 분석
    print("\n⚠️ 문제 케이스:")
    for i, result in enumerate(results, 1):
        if not result['is_valid']:
            print(f"  - 테스트 {i}: 한국어 비율 부족")
            print(f"    원본: {result['raw_output'][:100]}...")
    
    # 성공 예시
    print("\n✅ 성공 예시:")
    for i, result in enumerate(results[:3], 1):
        if result['is_valid']:
            print(f"  {i}. {result['filtered_output']}")
    
    print("\n✅ 테스트 완료!")
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        print(f"\n💾 최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    main()