#!/usr/bin/env python3
"""
생성 모델 반복 문제 해결 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import warnings
import re
warnings.filterwarnings("ignore")

def detect_repetition(text, min_length=10, max_repeats=3):
    """텍스트에서 반복 패턴 감지"""
    words = text.split()
    for length in range(min_length, len(words) // max_repeats + 1):
        for i in range(len(words) - length * max_repeats + 1):
            pattern = words[i:i+length]
            is_repeated = True
            for j in range(1, max_repeats):
                if words[i+j*length:i+(j+1)*length] != pattern:
                    is_repeated = False
                    break
            if is_repeated:
                return True, i + length
    return False, len(words)

def test_generator_with_fixes():
    """생성 모델 테스트 - 반복 문제 해결 포함"""
    print("=== 생성 모델 테스트 (반복 문제 해결) ===\n")
    
    model_path = "../ai-service/models/lora_gen_4bit"
    model_id = "openchat/openchat-3.5-0106"
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("모델 로딩 중...")
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "10GB"},
        offload_folder="offload",
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 테스트 케이스
    test_cases = [
        {
            "instruction": "다음 민원 내용을 요약해주세요.",
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
    
    # 다양한 생성 설정 테스트
    generation_configs = [
        {
            "name": "기본 설정 (repetition_penalty 1.5)",
            "config": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.5,  # 증가
                "no_repeat_ngram_size": 3,  # 3-gram 반복 방지
            }
        },
        {
            "name": "강화된 설정 (repetition_penalty 2.0)",
            "config": {
                "max_new_tokens": 300,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 50,  # top-k 샘플링 추가
                "repetition_penalty": 2.0,  # 더 증가
                "no_repeat_ngram_size": 4,  # 4-gram 반복 방지
            }
        },
        {
            "name": "Beam Search (deterministic)",
            "config": {
                "max_new_tokens": 300,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "repetition_penalty": 1.2,
            }
        },
        {
            "name": "낮은 temperature + 높은 penalty",
            "config": {
                "max_new_tokens": 300,
                "temperature": 0.5,  # 낮은 temperature
                "do_sample": True,
                "top_p": 0.8,
                "repetition_penalty": 1.8,
                "no_repeat_ngram_size": 3,
                "encoder_repetition_penalty": 1.0,  # 입력 텍스트 반복 방지
            }
        }
    ]
    
    print("\n=== 생성 결과 비교 ===\n")
    
    # 첫 번째 테스트 케이스로 다양한 설정 테스트
    test = test_cases[0]
    print(f"테스트 케이스:")
    print(f"📝 지시: {test['instruction']}")
    print(f"📥 입력: {test['input'][:100]}...")
    print("\n" + "="*80 + "\n")
    
    for config_info in generation_configs:
        print(f"\n🔧 {config_info['name']}")
        print(f"   설정: {config_info['config']}")
        print("-" * 60)
        
        # 프롬프트 생성
        prompt = f"""GPT4 Correct User: {test['instruction']}

{test['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **config_info['config']
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split("GPT4 Correct Assistant:")[-1].strip()
        
        # 반복 감지
        has_repetition, stop_idx = detect_repetition(generated)
        
        if has_repetition:
            print(f"⚠️  반복 감지됨! (위치: {stop_idx} 단어)")
            # 반복 시작 전까지만 출력
            generated_words = generated.split()[:stop_idx]
            generated_clean = " ".join(generated_words)
            print(f"📤 생성 답변 (반복 제거): {generated_clean}")
        else:
            print(f"✅ 반복 없음")
            print(f"📤 생성 답변: {generated}")
        
        print(f"   토큰 수: {len(tokenizer.encode(generated))}")
        print("-" * 60)
    
    print("\n" + "="*80 + "\n")
    print("\n=== 최적 설정으로 전체 테스트 ===\n")
    
    # 가장 좋은 설정으로 모든 테스트 케이스 실행
    best_config = {
        "max_new_tokens": 300,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.8,
        "no_repeat_ngram_size": 3,
        "encoder_repetition_penalty": 1.0,
    }
    
    print(f"선택된 최적 설정: {best_config}\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n테스트 {i}:")
        print(f"📝 지시: {test['instruction']}")
        print(f"📥 입력: {test['input'][:100]}...")
        
        # 프롬프트 생성
        prompt = f"""GPT4 Correct User: {test['instruction']}

{test['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **best_config
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split("GPT4 Correct Assistant:")[-1].strip()
        
        # 반복 감지 및 처리
        has_repetition, stop_idx = detect_repetition(generated)
        
        if has_repetition:
            generated_words = generated.split()[:stop_idx]
            generated = " ".join(generated_words)
            print(f"⚠️  반복 제거 적용")
        
        print(f"📤 생성 답변: {generated}")
        print("-" * 60)
    
    print("\n✅ 테스트 완료")
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        print(f"\n💾 최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    test_generator_with_fixes()