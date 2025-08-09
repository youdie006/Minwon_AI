#!/usr/bin/env python3
"""
빠른 성능 테스트 스크립트
"""

import json
import numpy as np
from pathlib import Path

def analyze_training_logs():
    """학습 로그 분석"""
    print("\n=== 학습 성능 요약 ===\n")
    
    # 분류 모델 로그 분석
    print("1. 분류 모델 (Llama-3.1-8B)")
    print("-" * 40)
    
    cls_log_path = Path("train_classifier/classifier_training.log")
    if cls_log_path.exists():
        with open(cls_log_path, 'r') as f:
            lines = f.readlines()
            
        # 최종 평가 메트릭 찾기
        for line in reversed(lines):
            if "eval_accuracy" in line and "eval_f1" in line:
                try:
                    metrics = eval(line.strip())
                    print(f"  • 정확도: {metrics['eval_accuracy']:.4f}")
                    print(f"  • F1 Score (Macro): {metrics['eval_f1_macro']:.4f}")
                    print(f"  • 검증 Loss: {metrics['eval_loss']:.4f}")
                    print(f"  • 각 카테고리 정확도:")
                    categories = ["교통", "환경", "안전", "복지", "문화", "경제", "주택/건설", "기타"]
                    for cat, acc in zip(categories, metrics['eval_label_accuracies']):
                        print(f"    - {cat}: {acc:.4f}")
                    break
                except:
                    pass
    
    print("\n2. 생성 모델 (OpenChat-3.5)")
    print("-" * 40)
    
    # 생성 모델 로그 분석
    gen_logs = [
        ("train_generator/step_0to200.log", "Step 0-200"),
        ("train_generator/step_200to277.log", "Step 200-277"),
        ("train_generator/step_400to500_final.log", "Step 400-500")
    ]
    
    for log_path, phase in gen_logs:
        if Path(log_path).exists():
            with open(log_path, 'r') as f:
                content = f.read()
                
            # Loss 값 찾기
            if "'loss':" in content:
                losses = []
                for line in content.split('\n'):
                    if "'loss':" in line and "grad_norm" in line:
                        try:
                            metrics = eval("{" + line.split("{")[1].split("}")[0] + "}")
                            losses.append(metrics['loss'])
                        except:
                            pass
                
                if losses:
                    print(f"  • {phase}:")
                    print(f"    - 최종 Loss: {losses[-1]:.4f}")
                    print(f"    - 평균 Loss: {np.mean(losses):.4f}")

def test_samples():
    """몇 가지 샘플로 간단 테스트"""
    print("\n\n=== 샘플 테스트 ===\n")
    
    # 테스트 데이터 로드
    cls_test = []
    gen_test = []
    
    if Path("../data/processed/cls_test.jsonl").exists():
        with open("../data/processed/cls_test.jsonl", 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                cls_test.append(json.loads(line))
    
    if Path("../data/processed/gen_test.jsonl").exists():
        with open("../data/processed/gen_test.jsonl", 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                gen_test.append(json.loads(line))
    
    # 분류 테스트 샘플
    print("1. 분류 모델 테스트 샘플")
    print("-" * 40)
    categories = ["교통", "환경", "안전", "복지", "문화", "경제", "주택/건설", "기타"]
    
    for i, item in enumerate(cls_test, 1):
        print(f"\n샘플 {i}:")
        print(f"  민원: {item['text'][:150]}...")
        true_cats = [categories[j] for j, label in enumerate(item['labels']) if label == 1]
        print(f"  실제 카테고리: {', '.join(true_cats)}")
        print(f"  출처: {item.get('source', 'N/A')}")
    
    # 생성 테스트 샘플
    print("\n\n2. 생성 모델 테스트 샘플")
    print("-" * 40)
    
    for i, item in enumerate(gen_test, 1):
        print(f"\n샘플 {i}:")
        print(f"  지시: {item['instruction']}")
        print(f"  입력: {item['input'][:150]}...")
        print(f"  정답: {item['output'][:150]}...")

def analyze_data_distribution():
    """데이터 분포 분석"""
    print("\n\n=== 데이터 분포 ===\n")
    
    stats_path = Path("../data/processed/data_stats.json")
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print("1. 데이터셋 크기")
        print("-" * 40)
        print(f"  • 전체 샘플: {stats['total_samples']:,}")
        print(f"  • 학습 데이터: {stats['train_samples']:,}")
        print(f"  • 검증 데이터: {stats['val_samples']:,}")
        print(f"  • 테스트 데이터: {stats['test_samples']:,}")
        
        print("\n2. 카테고리 분포")
        print("-" * 40)
        for cat, count in stats['category_distribution'].items():
            percentage = (count / stats['total_samples']) * 100
            print(f"  • {cat}: {count:,} ({percentage:.1f}%)")
        
        print("\n3. 데이터 특성")
        print("-" * 40)
        print(f"  • 평균 텍스트 길이: {stats['avg_text_length']:.0f} 문자")
        print(f"  • 최대 텍스트 길이: {stats['max_text_length']:,} 문자")
        print(f"  • 최소 텍스트 길이: {stats['min_text_length']} 문자")
        
        if 'multi_label_ratio' in stats:
            print(f"  • 다중 레이블 비율: {stats['multi_label_ratio']:.1%}")

def main():
    """메인 실행"""
    print("="*60)
    print("MinwonAI 모델 성능 분석 리포트")
    print("="*60)
    
    # 학습 로그 분석
    analyze_training_logs()
    
    # 데이터 분포 분석
    analyze_data_distribution()
    
    # 샘플 테스트
    test_samples()
    
    print("\n" + "="*60)
    print("분석 완료")
    print("="*60)

if __name__ == "__main__":
    main()