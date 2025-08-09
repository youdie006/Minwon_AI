#!/usr/bin/env python3
"""
파인튜닝 모델 최적화 - 파인튜닝 토크나이저 유지
- apply_chat_template 활용
- 극단적 디코딩 파라미터
- 멀티 패스 생성
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append('..')
from utils.inference_filter import GenerationFilter
import re
import random
import warnings
warnings.filterwarnings("ignore")

class OptimizedFinetunedGenerator:
    """최적화된 파인튜닝 생성 모델"""
    
    def __init__(self):
        """초기화"""
        self.model_path = "../ai-service/models/lora_gen_4bit"
        self.model_id = "openchat/openchat-3.5-0106"
        self.filter = GenerationFilter()
        self.model = None
        self.tokenizer = None
        
        # 극단적 생성 설정
        self.extreme_config = {
            "max_new_tokens": 50,           # 매우 짧게
            "min_new_tokens": 10,           # 최소 길이 보장
            "temperature": 0.2,             # 매우 낮은 온도
            "do_sample": True,
            "top_p": 0.5,                   # 좁은 확률 분포
            "top_k": 10,                    # 상위 10개만
            "repetition_penalty": 5.0,      # 매우 강한 반복 방지
            "no_repeat_ngram_size": 6,      # 6-gram 반복 차단
            "eos_token_id": 32000,          # <|end_of_turn|>
            "pad_token_id": 32000,
            "bad_words_ids": [[32000, 32000]],  # EOS 연속 방지
            "exponential_decay_length_penalty": (30, 1.5),  # 길이 패널티
        }
        
        # 다양한 설정 옵션들
        self.config_variants = [
            {
                "name": "극단적 반복 방지",
                "config": self.extreme_config
            },
            {
                "name": "Greedy + 강한 패널티",
                "config": {
                    "max_new_tokens": 50,
                    "do_sample": False,
                    "repetition_penalty": 8.0,
                    "no_repeat_ngram_size": 4,
                    "eos_token_id": 32000,
                    "pad_token_id": 32000,
                }
            },
            {
                "name": "매우 낮은 온도",
                "config": {
                    "max_new_tokens": 40,
                    "temperature": 0.1,
                    "do_sample": True,
                    "top_p": 0.3,
                    "repetition_penalty": 6.0,
                    "no_repeat_ngram_size": 5,
                    "eos_token_id": 32000,
                    "pad_token_id": 32000,
                }
            }
        ]
        
    def load_model(self):
        """모델 로드"""
        print("파인튜닝 모델 로딩 중...")
        
        # 4-bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # 파인튜닝 토크나이저 사용
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
    
    def create_prompt_with_template(self, instruction: str, input_text: str) -> str:
        """apply_chat_template 활용한 프롬프트 생성"""
        # 메시지 형식
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{input_text}\n\n간단하고 명확하게 한국어로 답변하세요. 50자 이내로 작성하세요."}
        ]
        
        # apply_chat_template 사용
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 폴백: 수동 템플릿
            prompt = f"GPT4 Correct User: {messages[0]['content']}<|end_of_turn|>GPT4 Correct Assistant:"
        
        return prompt
    
    def evaluate_quality(self, text: str) -> float:
        """생성 텍스트 품질 점수"""
        score = 100.0
        
        # 빈 텍스트
        if not text or len(text.strip()) < 5:
            return 0
        
        # 한국어 비율
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣A-Za-z]', text))
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            if korean_ratio < 0.7:
                score -= 50
        else:
            score -= 80
        
        # 반복 체크
        words = text.split()
        for i in range(len(words) - 2):
            if i + 6 <= len(words) and words[i:i+3] == words[i+3:i+6]:
                score -= 30
                break
        
        # 특수문자 과다
        special_ratio = len(re.findall(r'[^가-힣\s.,!?]', text)) / max(len(text), 1)
        if special_ratio > 0.3:
            score -= 20
        
        # 길이 체크
        if len(words) < 5:
            score -= 20
        elif len(words) > 50:
            score -= 10
        
        # 의미없는 패턴
        if re.search(r'[A-Z]{5,}', text):  # 긴 대문자
            score -= 15
        if re.search(r'[0-9]{10,}', text):  # 긴 숫자
            score -= 15
        
        return max(0, score)
    
    def generate_single(self, prompt: str, config: dict) -> tuple:
        """단일 생성"""
        # 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **config
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Assistant 응답 추출
        if "GPT4 Correct Assistant:" in response:
            generated = response.split("GPT4 Correct Assistant:")[-1].strip()
        else:
            generated = response[len(prompt):].strip()
        
        # EOS 토큰 이후 제거
        if "<|end_of_turn|>" in generated:
            generated = generated.split("<|end_of_turn|>")[0].strip()
        
        # 필터 적용
        filtered = self.filter.filter(generated, max_length=50)
        
        # 품질 점수
        score = self.evaluate_quality(filtered)
        
        return filtered, score
    
    def generate_best_of_n(self, instruction: str, input_text: str, n: int = 5) -> dict:
        """여러 번 생성 후 최적 선택"""
        if not self.model:
            self.load_model()
        
        # 프롬프트 생성
        prompt = self.create_prompt_with_template(instruction, input_text)
        
        candidates = []
        
        # 다양한 설정으로 생성
        for i in range(n):
            # 설정 선택
            config_variant = self.config_variants[i % len(self.config_variants)]
            config = config_variant["config"].copy()
            
            # 랜덤 시드
            torch.manual_seed(i * 42 + random.randint(0, 1000))
            
            try:
                text, score = self.generate_single(prompt, config)
                candidates.append({
                    "text": text,
                    "score": score,
                    "config": config_variant["name"]
                })
            except Exception as e:
                print(f"생성 실패 ({i+1}/{n}): {str(e)[:50]}")
                continue
        
        # 최적 선택
        if candidates:
            best = max(candidates, key=lambda x: x["score"])
        else:
            best = {
                "text": "죄송합니다. 적절한 답변을 생성하지 못했습니다.",
                "score": 0,
                "config": "실패"
            }
        
        return {
            "instruction": instruction,
            "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "best_output": best["text"],
            "best_score": best["score"],
            "best_config": best["config"],
            "all_candidates": candidates
        }
    
    def test_generation(self):
        """테스트 실행"""
        test_cases = [
            {
                "instruction": "다음 민원을 요약하세요.",
                "input": "공원에 쓰레기가 많습니다. 청소를 요청합니다."
            },
            {
                "instruction": "다음 민원에 답변하세요.",
                "input": "횡단보도 신호 시간이 짧습니다."
            },
            {
                "instruction": "핵심 요구사항을 정리하세요.",
                "input": "밤에 오토바이 소음이 심합니다. 단속해주세요."
            }
        ]
        
        print("=== 멀티 패스 생성 테스트 ===\n")
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n테스트 {i}:")
            print(f"지시: {test['instruction']}")
            print(f"입력: {test['input']}")
            print("-" * 60)
            
            result = self.generate_best_of_n(
                test["instruction"],
                test["input"],
                n=5
            )
            
            print(f"최적 답변: {result['best_output']}")
            print(f"점수: {result['best_score']:.1f}")
            print(f"사용 설정: {result['best_config']}")
            
            # 모든 후보 출력
            print("\n후보들:")
            for j, cand in enumerate(result['all_candidates'], 1):
                print(f"  {j}. [{cand['score']:.0f}점] {cand['text'][:50]}...")
        
        print("\n완료!")


def main():
    """메인 함수"""
    print("=== 파인튜닝 모델 최적화 테스트 ===\n")
    
    generator = OptimizedFinetunedGenerator()
    generator.load_model()
    generator.test_generation()
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        print(f"\n최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    main()