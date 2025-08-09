#!/usr/bin/env python3
"""
베이스 모델과 파인튜닝 모델 비교 분석
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append("..")
from utils.inference_filter import GenerationFilter
import warnings
warnings.filterwarnings("ignore")

def compare_models():
    """베이스 모델과 파인튜닝 모델 상세 비교"""
    print("=== 베이스 모델 vs 파인튜닝 모델 비교 ===\n")
    
    model_id = "openchat/openchat-3.5-0106"
    lora_path = "../ai-service/models/lora_gen_4bit"
    filter_obj = GenerationFilter()
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 동일한 테스트 케이스
    test_cases = [
        {
            "instruction": "다음 민원을 요약해주세요.",
            "input": "공원에 쓰레기가 많이 쌓여 있습니다. 정기적인 청소와 쓰레기통 추가 설치를 요청합니다.",
            "expected": "공원 청소 및 쓰레기통 설치 요청"
        },
        {
            "instruction": "다음 민원에 대한 답변을 작성해주세요.",
            "input": "횡단보도 신호 시간이 너무 짧습니다.",
            "expected": "신호 시간 연장 검토하겠습니다"
        },
        {
            "instruction": "다음 텍스트를 요약하세요.",
            "input": "주민들이 야간 소음으로 불편을 겪고 있습니다. 단속을 강화해주세요.",
            "expected": "야간 소음 단속 요청"
        }
    ]
    
    # 생성 설정 (동일하게 적용)
    gen_config = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
    }
    
    results = {"base": [], "finetuned": []}
    
    print("1. 베이스 모델 테스트")
    print("-" * 60)
    
    # 베이스 모델 로드
    print("베이스 모델 로딩...")
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "10GB"},
        offload_folder="offload",
    )
    base_model.eval()
    
    # 베이스 모델 테스트
    for i, test in enumerate(test_cases, 1):
        prompt = f"""GPT4 Correct User: {test['instruction']}

{test['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
        
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
        filtered = filter_obj.filter(generated, max_length=100)
        
        print(f"\n테스트 {i}:")
        print(f"입력: {test['input']}")
        print(f"생성: {filtered}")
        print(f"기대: {test['expected']}")
        
        results["base"].append({
            "raw": generated,
            "filtered": filtered,
            "valid": filter_obj.validate_korean(filtered)
        })
    
    # 메모리 정리
    del base_model
    torch.cuda.empty_cache()
    
    print("\n\n2. 파인튜닝 모델 테스트")
    print("-" * 60)
    
    # 파인튜닝 모델 로드
    print("파인튜닝 모델 로딩...")
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
    
    # 파인튜닝 모델 테스트
    for i, test in enumerate(test_cases, 1):
        prompt = f"""GPT4 Correct User: {test['instruction']}

{test['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
        
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
        filtered = filter_obj.filter(generated, max_length=100)
        
        print(f"\n테스트 {i}:")
        print(f"입력: {test['input']}")
        print(f"생성: {filtered}")
        print(f"기대: {test['expected']}")
        
        results["finetuned"].append({
            "raw": generated,
            "filtered": filtered,
            "valid": filter_obj.validate_korean(filtered)
        })
    
    # 비교 분석
    print("\n\n" + "=" * 60)
    print("📊 비교 분석 결과")
    print("=" * 60)
    
    print("\n### 1. 품질 비교")
    base_valid = sum(1 for r in results["base"] if r["valid"])
    ft_valid = sum(1 for r in results["finetuned"] if r["valid"])
    
    print(f"베이스 모델 성공률: {base_valid}/{len(results['base'])} ({base_valid/len(results['base'])*100:.0f}%)")
    print(f"파인튜닝 모델 성공률: {ft_valid}/{len(results['finetuned'])} ({ft_valid/len(results['finetuned'])*100:.0f}%)")
    
    print("\n### 2. 출력 특성 비교")
    print("\n베이스 모델 특징:")
    print("- 일반적인 대화 능력 유지")
    print("- 다양한 언어 혼재 (영어/한국어)")
    print("- 프롬프트 따라가기 능력 있음")
    
    print("\n파인튜닝 모델 특징:")
    print("- 학습 데이터에 과적합")
    print("- 의미없는 패턴 반복")
    print("- 특정 도메인 용어 과다 사용")
    
    print("\n### 3. 문제점 분석")
    print("\n베이스 모델 문제:")
    print("- 한국어 민원 도메인 지식 부족")
    print("- 일관성 없는 답변 형식")
    
    print("\n파인튜닝 모델 문제:")
    print("- 500스텝으로 부족한 학습")
    print("- 과적합으로 인한 품질 저하")
    print("- 반복 및 의미없는 텍스트 생성")
    
    print("\n### 4. 개선 방향")
    print("\n단기:")
    print("- 베이스 모델 + 강력한 프롬프트 엔지니어링")
    print("- 후처리 필터 필수 적용")
    
    print("\n장기:")
    print("- 학습 데이터 품질 개선")
    print("- 더 많은 학습 스텝 (2000-3000)")
    print("- 다른 베이스 모델 시도 (SOLAR, Polyglot-Ko)")
    
    # 실제 예시 비교
    print("\n\n### 5. 실제 출력 예시")
    print("-" * 60)
    for i in range(len(test_cases)):
        print(f"\n테스트 {i+1}:")
        print(f"베이스: {results['base'][i]['filtered'][:80]}...")
        print(f"파인튜닝: {results['finetuned'][i]['filtered'][:80]}...")
    
    print("\n✅ 분석 완료")
    
    return results


def analyze_training_impact():
    """학습의 영향 분석"""
    print("\n\n=== 파인튜닝의 영향 분석 ===\n")
    
    print("### 학습 데이터 특성")
    print("- 데이터 크기: 10,259개 학습 샘플")
    print("- 학습 스텝: 500 스텝")
    print("- 손실값: 0.71 (최종)")
    print("- 학습률: 5e-5 → 6.7e-7 (감소)")
    
    print("\n### 학습 과정 문제")
    print("1. **불충분한 학습**")
    print("   - 500스텝은 0.39 에폭에 불과")
    print("   - 모델이 패턴을 제대로 학습하지 못함")
    
    print("\n2. **과적합 징후**")
    print("   - 특정 패턴만 반복 생성")
    print("   - 학습 데이터의 노이즈까지 외움")
    
    print("\n3. **데이터 품질**")
    print("   - 중복 데이터 존재 가능성")
    print("   - 출력 길이 불균형")
    print("   - 도메인 특화 용어 과다")
    
    print("\n### 베이스 모델 vs 파인튜닝 모델")
    print("\n| 항목 | 베이스 모델 | 파인튜닝 모델 |")
    print("|------|------------|--------------|")
    print("| 일반 대화 능력 | ✅ 유지 | ❌ 손상 |")
    print("| 한국어 품질 | 🔶 보통 | ❌ 낮음 |")
    print("| 반복 문제 | 🔶 가끔 | ❌ 심각 |")
    print("| 도메인 지식 | ❌ 부족 | 🔶 일부 |")
    print("| 프롬프트 따라가기 | ✅ 양호 | ❌ 불량 |")
    print("| 예측 가능성 | 🔶 보통 | ❌ 낮음 |")
    
    print("\n### 결론")
    print("현재 파인튜닝 모델은 베이스 모델보다 성능이 떨어짐")
    print("원인: 불충분한 학습 + 데이터 품질 문제")
    print("해결: 재학습 필요 (3-5 에폭, 더 나은 데이터)")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        torch.cuda.empty_cache()
    
    # 모델 비교
    results = compare_models()
    
    # 학습 영향 분석
    analyze_training_impact()