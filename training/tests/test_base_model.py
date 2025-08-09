#!/usr/bin/env python3
"""
베이스 모델과 파인튜닝 모델 비교 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def test_models():
    """베이스 모델과 파인튜닝 모델 비교"""
    print("=== 모델 비교 테스트 ===\n")
    
    model_id = "openchat/openchat-3.5-0106"
    lora_path = "../ai-service/models/lora_gen_4bit"
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 테스트 케이스
    test_case = {
        "instruction": "다음 민원 내용을 요약해주세요.",
        "input": "안녕하세요. 저는 oo동에 거주하는 주민입니다. 최근 우리 동네 공원에 쓰레기가 많이 쌓여 있어 불편을 겪고 있습니다. 특히 주말이 지나면 음식물 쓰레기와 일회용품이 곳곳에 버려져 있어 악취가 나고 미관상 좋지 않습니다. 구청에서 정기적인 청소와 쓰레기통 추가 설치를 검토해 주시기 바랍니다."
    }
    
    # 생성 설정
    gen_config = {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }
    
    print("1. 베이스 모델 테스트")
    print("-" * 60)
    
    # 베이스 모델 로드
    print("베이스 모델 로딩 중...")
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "10GB"},
        offload_folder="offload",
    )
    base_model.eval()
    
    # 베이스 모델로 생성
    prompt = f"""GPT4 Correct User: {test_case['instruction']}

{test_case['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
    
    inputs = base_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            pad_token_id=base_tokenizer.eos_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
            **gen_config
        )
    
    response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response.split("GPT4 Correct Assistant:")[-1].strip()
    
    print(f"📝 지시: {test_case['instruction']}")
    print(f"📥 입력: {test_case['input'][:100]}...")
    print(f"📤 베이스 모델 답변:\n{generated}\n")
    
    # 메모리 정리
    del base_model
    torch.cuda.empty_cache()
    
    print("\n2. 파인튜닝 모델 테스트")
    print("-" * 60)
    
    # 파인튜닝 모델 로드
    print("파인튜닝 모델 로딩 중...")
    ft_tokenizer = AutoTokenizer.from_pretrained(lora_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "10GB"},
        offload_folder="offload",
    )
    
    ft_model = PeftModel.from_pretrained(base_model, lora_path)
    ft_model.eval()
    
    # 파인튜닝 모델로 생성
    inputs = ft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = ft_model.generate(
            **inputs,
            pad_token_id=ft_tokenizer.eos_token_id,
            eos_token_id=ft_tokenizer.eos_token_id,
            **gen_config
        )
    
    response = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response.split("GPT4 Correct Assistant:")[-1].strip()
    
    print(f"📝 지시: {test_case['instruction']}")
    print(f"📥 입력: {test_case['input'][:100]}...")
    print(f"📤 파인튜닝 모델 답변:\n{generated}\n")
    
    print("\n3. 추가 테스트 - 간단한 질문")
    print("-" * 60)
    
    simple_tests = [
        "안녕하세요?",
        "오늘 날씨가 어떤가요?",
        "민원 처리 절차를 알려주세요."
    ]
    
    for question in simple_tests:
        prompt = f"GPT4 Correct User: {question}<|end_of_turn|>GPT4 Correct Assistant:"
        
        inputs = ft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ft_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=ft_tokenizer.eos_token_id,
                eos_token_id=ft_tokenizer.eos_token_id,
            )
        
        response = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split("GPT4 Correct Assistant:")[-1].strip()
        
        print(f"Q: {question}")
        print(f"A: {generated}\n")
    
    print("✅ 테스트 완료")
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        print(f"\n💾 최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    test_models()