#!/usr/bin/env python3
"""
간단한 베이스 모델 테스트
파인튜닝 모델이 문제가 있으므로 베이스 모델만 사용
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
sys.path.append("..")
from utils.inference_filter import GenerationFilter
import warnings
warnings.filterwarnings("ignore")

def test_base_model_only():
    """베이스 모델만 테스트"""
    print("=== 베이스 모델 직접 테스트 ===\n")
    
    model_id = "openchat/openchat-3.5-0106"
    filter_obj = GenerationFilter()
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("베이스 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "10GB"},
        offload_folder="offload",
    )
    model.eval()
    print("모델 로드 완료\n")
    
    # 테스트 케이스
    test_cases = [
        {
            "instruction": "다음 텍스트를 한국어로 요약하세요.",
            "input": "The park is dirty with trash everywhere. Need regular cleaning and more trash cans."
        },
        {
            "instruction": "Summarize in Korean:",
            "input": "Citizens complain about noise from motorcycles at night. They request stronger enforcement."
        },
        {
            "instruction": "한국어로 답변:",
            "input": "공원에 쓰레기가 많습니다. 청소를 요청합니다."
        }
    ]
    
    # 다양한 프롬프트 형식 테스트
    prompt_formats = [
        # OpenChat 기본 형식
        lambda inst, inp: f"GPT4 Correct User: {inst}\n\n{inp}<|end_of_turn|>GPT4 Correct Assistant:",
        
        # 단순 형식
        lambda inst, inp: f"Instruction: {inst}\nInput: {inp}\nResponse:",
        
        # 한국어 강조
        lambda inst, inp: f"User: {inst}\n{inp}\nAssistant (Korean):",
    ]
    
    print("테스트 시작\n")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n테스트 {i}:")
        print(f"지시: {test['instruction']}")
        print(f"입력: {test['input']}")
        print("-" * 60)
        
        for j, format_fn in enumerate(prompt_formats, 1):
            prompt = format_fn(test['instruction'], test['input'])
            
            # 토크나이징
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 디코딩
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거
            if "Assistant:" in response:
                generated = response.split("Assistant:")[-1].strip()
            elif "Response:" in response:
                generated = response.split("Response:")[-1].strip()
            else:
                generated = response[len(prompt):].strip()
            
            # 필터 적용
            filtered = filter_obj.filter(generated, max_length=100)
            
            print(f"\n형식 {j} 결과:")
            print(f"원본: {generated[:150]}...")
            print(f"필터: {filtered}")
            is_valid = filter_obj.validate_korean(filtered)
            print(f"한국어 유효: {is_valid}")
    
    print("\n" + "=" * 80)
    print("\n✅ 테스트 완료")
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        print(f"\n💾 최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    test_base_model_only()