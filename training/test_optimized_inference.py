#!/usr/bin/env python3
"""
최적화된 추론 파라미터 테스트
- 다양한 생성 전략 비교
- 최적 파라미터 도출
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from inference_filter import GenerationFilter
import warnings
warnings.filterwarnings("ignore")

class OptimizedGenerator:
    """최적화된 생성기"""
    
    def __init__(self, model_path: str, model_id: str):
        """
        초기화
        
        Args:
            model_path: LoRA 모델 경로
            model_id: 베이스 모델 ID
        """
        self.model_path = model_path
        self.model_id = model_id
        self.filter = GenerationFilter()
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """모델 로드"""
        print("모델 로딩 중...")
        
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
    
    def generate(self, prompt: str, strategy: dict) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            strategy: 생성 전략 설정
            
        Returns:
            생성된 텍스트
        """
        # 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **strategy
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split("GPT4 Correct Assistant:")[-1].strip()
        
        # 필터 적용
        filtered = self.filter.filter(generated, max_length=200)
        
        return filtered
    
    def test_strategies(self, test_case: dict):
        """
        다양한 전략 테스트
        
        Args:
            test_case: 테스트 케이스 (instruction, input)
        """
        # 다양한 생성 전략
        strategies = [
            {
                "name": "전략 1: 강한 반복 패널티",
                "config": {
                    "max_new_tokens": 150,
                    "temperature": 0.5,
                    "do_sample": True,
                    "top_p": 0.8,
                    "repetition_penalty": 2.0,
                    "no_repeat_ngram_size": 3,
                }
            },
            {
                "name": "전략 2: Greedy 디코딩",
                "config": {
                    "max_new_tokens": 150,
                    "do_sample": False,
                    "repetition_penalty": 1.5,
                    "no_repeat_ngram_size": 3,
                }
            },
            {
                "name": "전략 3: Contrastive Search",
                "config": {
                    "max_new_tokens": 150,
                    "penalty_alpha": 0.6,
                    "top_k": 4,
                }
            },
            {
                "name": "전략 4: 매우 낮은 temperature",
                "config": {
                    "max_new_tokens": 150,
                    "temperature": 0.3,
                    "do_sample": True,
                    "top_p": 0.7,
                    "top_k": 40,
                    "repetition_penalty": 1.8,
                    "no_repeat_ngram_size": 3,
                }
            },
            {
                "name": "전략 5: Beam Search",
                "config": {
                    "max_new_tokens": 150,
                    "num_beams": 3,
                    "no_repeat_ngram_size": 3,
                    "early_stopping": True,
                    "repetition_penalty": 1.3,
                }
            },
            {
                "name": "전략 6: 최적 조합",
                "config": {
                    "max_new_tokens": 100,
                    "temperature": 0.4,
                    "do_sample": True,
                    "top_p": 0.75,
                    "top_k": 30,
                    "repetition_penalty": 2.5,
                    "no_repeat_ngram_size": 4,
                    "encoder_repetition_penalty": 1.0,
                }
            }
        ]
        
        # 프롬프트 템플릿들
        prompt_templates = [
            # 기본 템플릿
            """GPT4 Correct User: {instruction}

{input}<|end_of_turn|>GPT4 Correct Assistant:""",
            
            # 개선된 템플릿
            """GPT4 Correct User: {instruction}

입력: {input}

간단하고 명확하게 한국어로 답변해주세요. 답변은 100자 이내로 작성하세요.<|end_of_turn|>GPT4 Correct Assistant:""",
            
            # Few-shot 템플릿
            """GPT4 Correct User: 다음 민원을 요약해주세요.

입력: 도로에 포트홀이 생겨서 위험합니다.

간단하고 명확하게 한국어로 답변해주세요.<|end_of_turn|>GPT4 Correct Assistant: 도로 포트홀 발생으로 인한 안전 문제 민원입니다. 신속한 보수가 필요합니다.<|end_of_turn|>GPT4 Correct User: {instruction}

입력: {input}

간단하고 명확하게 한국어로 답변해주세요.<|end_of_turn|>GPT4 Correct Assistant:"""
        ]
        
        print("\n=== 전략별 테스트 결과 ===\n")
        print(f"📝 지시: {test_case['instruction']}")
        print(f"📥 입력: {test_case['input'][:100]}...")
        print("=" * 80)
        
        best_result = None
        best_score = -1
        
        for strategy in strategies:
            print(f"\n🔧 {strategy['name']}")
            print(f"   설정: {strategy['config']}")
            print("-" * 60)
            
            results = []
            for i, template in enumerate(prompt_templates, 1):
                prompt = template.format(
                    instruction=test_case['instruction'],
                    input=test_case['input']
                )
                
                try:
                    generated = self.generate(prompt, strategy['config'])
                    
                    # 품질 점수 계산
                    score = self.calculate_quality_score(generated)
                    results.append((generated, score))
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'strategy': strategy['name'],
                            'template': f'템플릿 {i}',
                            'text': generated,
                            'score': score
                        }
                    
                    print(f"   템플릿 {i}: {generated[:150]}...")
                    print(f"   점수: {score:.2f}")
                    
                except Exception as e:
                    print(f"   템플릿 {i}: 오류 - {str(e)[:50]}")
                    results.append(("오류", 0))
            
            # 평균 점수
            avg_score = sum(r[1] for r in results) / len(results)
            print(f"   평균 점수: {avg_score:.2f}")
        
        print("\n" + "=" * 80)
        print("\n🏆 최적 결과:")
        if best_result:
            print(f"   전략: {best_result['strategy']}")
            print(f"   템플릿: {best_result['template']}")
            print(f"   점수: {best_result['score']:.2f}")
            print(f"   생성: {best_result['text']}")
        
        return best_result
    
    def calculate_quality_score(self, text: str) -> float:
        """
        생성 텍스트 품질 점수 계산
        
        Args:
            text: 평가할 텍스트
            
        Returns:
            품질 점수 (0-100)
        """
        score = 100.0
        
        # 길이 체크
        words = text.split()
        if len(words) < 5:
            score -= 30
        elif len(words) > 150:
            score -= 20
        
        # 한국어 비율
        if not self.filter.validate_korean(text):
            score -= 50
        
        # 반복 체크
        has_repetition, _ = self.filter.detect_word_repetition(text)
        if has_repetition:
            score -= 40
        
        # 특수문자 과다
        special_ratio = len([c for c in text if not c.isalnum() and not c.isspace()]) / max(len(text), 1)
        if special_ratio > 0.3:
            score -= 20
        
        # 의미없는 텍스트 체크
        if "죄송합니다. 적절한 답변을 생성하지 못했습니다." in text:
            score = 10
        
        return max(0, score)


def main():
    """메인 함수"""
    print("=== 최적화된 추론 테스트 ===\n")
    
    # 설정
    model_path = "../ai-service/models/lora_gen_4bit"
    model_id = "openchat/openchat-3.5-0106"
    
    # 생성기 초기화
    generator = OptimizedGenerator(model_path, model_id)
    generator.load_model()
    
    # 테스트 케이스들
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
        }
    ]
    
    # 각 테스트 케이스 실행
    best_configs = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#' * 80}")
        print(f"테스트 케이스 {i}")
        print('#' * 80)
        
        best = generator.test_strategies(test_case)
        if best:
            best_configs.append(best)
    
    # 최종 결과 요약
    print("\n\n" + "=" * 80)
    print("📊 최종 요약")
    print("=" * 80)
    
    if best_configs:
        avg_score = sum(c['score'] for c in best_configs) / len(best_configs)
        print(f"\n평균 점수: {avg_score:.2f}")
        
        # 가장 많이 선택된 전략
        strategies = [c['strategy'] for c in best_configs]
        most_common = max(set(strategies), key=strategies.count)
        print(f"최적 전략: {most_common}")
        
        # 가장 많이 선택된 템플릿
        templates = [c['template'] for c in best_configs]
        most_common_template = max(set(templates), key=templates.count)
        print(f"최적 템플릿: {most_common_template}")
    
    print("\n✅ 테스트 완료")
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        print(f"\n💾 최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    main()