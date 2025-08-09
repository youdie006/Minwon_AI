#!/usr/bin/env python3
"""
파인튜닝된 모델 성능 평가 스크립트
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
import os
from dotenv import load_dotenv
load_dotenv()

def load_test_data(file_path: str) -> List[Dict]:
    """테스트 데이터 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_classifier(model_path: str, test_data_path: str):
    """분류 모델 평가"""
    print("\n=== 분류 모델 평가 ===")
    
    # 모델 로드
    print("모델 로딩 중...")
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right"
    )
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 테스트 데이터 로드
    test_data = load_test_data(test_data_path)[:20]  # 20개만 테스트 (속도를 위해)
    
    # 카테고리 매핑
    categories = [
        "교통", "환경", "안전", "복지",
        "문화", "경제", "주택/건설", "기타"
    ]
    
    predictions = []
    true_labels = []
    
    print(f"테스트 샘플 수: {len(test_data)}")
    print("추론 시작...")
    
    for item in tqdm(test_data):
        # 프롬프트 생성
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 민원 분류 전문가입니다. 주어진 민원을 8개 카테고리 중 하나 이상으로 분류하세요.
카테고리: 교통, 환경, 안전, 복지, 문화, 경제, 주택/건설, 기타<|eot_id|>

<|start_header_id|>user<|end_header_id|>
민원 내용: {item['text'][:500]}

이 민원이 속하는 카테고리를 모두 선택하세요.<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
이 민원의 카테고리는"""

        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response.split("이 민원의 카테고리는")[-1].strip()
        
        # 예측 레이블 추출
        pred_labels = []
        for i, cat in enumerate(categories):
            if cat in response_text:
                pred_labels.append(i)
        
        if not pred_labels:  # 예측이 없으면 기타로 분류
            pred_labels = [7]
        
        # 실제 레이블
        true_label_indices = [i for i, val in enumerate(item['labels']) if val == 1]
        
        predictions.append(pred_labels)
        true_labels.append(true_label_indices)
    
    # 성능 계산
    print("\n=== 분류 성능 ===")
    
    # 정확도 계산 (완전 일치)
    exact_matches = sum(1 for p, t in zip(predictions, true_labels) if set(p) == set(t))
    exact_accuracy = exact_matches / len(predictions)
    print(f"완전 일치 정확도: {exact_accuracy:.4f}")
    
    # 각 카테고리별 성능
    print("\n카테고리별 성능:")
    for i, cat in enumerate(categories):
        cat_true = [1 if i in t else 0 for t in true_labels]
        cat_pred = [1 if i in p else 0 for p in predictions]
        
        if sum(cat_true) > 0:  # 해당 카테고리가 테스트 데이터에 있는 경우만
            acc = accuracy_score(cat_true, cat_pred)
            f1 = f1_score(cat_true, cat_pred)
            print(f"  {cat}: Accuracy={acc:.4f}, F1={f1:.4f}")

def evaluate_generator(model_path: str, test_data_path: str):
    """생성 모델 평가"""
    print("\n=== 생성 모델 평가 ===")
    
    # 모델 로드
    print("모델 로딩 중...")
    model_id = "openchat/openchat-3.5-0106"
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 베이스 모델 로드 (메모리 최적화)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "7GB", "cpu": "10GB"},  # GPU 메모리 제한
        offload_folder="offload",
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 테스트 데이터 로드
    test_data = load_test_data(test_data_path)[:3]  # 3개만 테스트 (생성은 느림)
    
    print(f"테스트 샘플 수: {len(test_data)}")
    print("\n=== 생성 예시 ===")
    
    for i, item in enumerate(test_data, 1):
        print(f"\n예시 {i}:")
        print(f"지시: {item['instruction']}")
        print(f"입력: {item['input'][:200]}...")
        print(f"정답: {item['output'][:200]}...")
        
        # 프롬프트 생성
        prompt = f"""GPT4 Correct User: {item['instruction']}

{item['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response.split("GPT4 Correct Assistant:")[-1].strip()
        
        print(f"생성: {generated_text[:200]}...")
        print("-" * 50)

def main():
    """메인 실행 함수"""
    # CUDA 확인
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    else:
        print("GPU를 사용할 수 없습니다.")
        return
    
    # 경로 설정
    cls_model_path = "../ai-service/models/lora_cls_4bit"
    gen_model_path = "../ai-service/models/lora_gen_4bit"
    cls_test_path = "../data/processed/cls_test.jsonl"
    gen_test_path = "../data/processed/gen_test.jsonl"
    
    # 분류 모델 평가
    if Path(cls_model_path).exists() and Path(cls_test_path).exists():
        evaluate_classifier(cls_model_path, cls_test_path)
    else:
        print(f"분류 모델 또는 테스트 데이터를 찾을 수 없습니다.")
    
    print("\n" + "="*50 + "\n")
    
    # 생성 모델 평가
    if Path(gen_model_path).exists() and Path(gen_test_path).exists():
        evaluate_generator(gen_model_path, gen_test_path)
    else:
        print(f"생성 모델 또는 테스트 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()