#!/usr/bin/env python3
"""
생성 모델 간단 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import warnings
warnings.filterwarnings("ignore")

def test_generator():
    """생성 모델 테스트"""
    print("=== 생성 모델 테스트 ===\n")
    
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
    
    print("\n=== 생성 결과 ===\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"테스트 {i}:")
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
                max_new_tokens=1000,  # 충분히 긴 답변을 위해 1000토큰으로 설정
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 반복 방지
                min_length=100  # 최소 길이 보장
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split("GPT4 Correct Assistant:")[-1].strip()
        
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
    
    test_generator()