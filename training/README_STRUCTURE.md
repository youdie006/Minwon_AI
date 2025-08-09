# Training 폴더 구조

## 디렉토리 구조
```
training/
├── train_classifier/        # 분류 모델 학습 스크립트
│   ├── train_classifier.py
│   └── *.log
├── train_generator/         # 생성 모델 학습 스크립트  
│   ├── train_generator.py
│   ├── resume_*.sh
│   └── *.log
├── scripts/                 # 데이터 처리 스크립트
│   ├── download_data_local.sh
│   └── prepare_data_v2.py
├── tests/                   # 테스트 및 평가 스크립트
│   ├── test_generator.py
│   ├── test_base_model.py
│   ├── simple_test.py
│   ├── compare_models.py
│   ├── improved_generator.py
│   └── optimized_finetuned_generator.py
├── utils/                   # 유틸리티 모듈
│   └── inference_filter.py  # 반복 제거 필터
├── evaluate_models.py       # 모델 평가 메인 스크립트
├── GENERATION_MODEL_IMPROVEMENT.md  # 개선 결과 문서
└── requirements.txt

```

## 주요 파일 설명

### 학습 스크립트
- `train_classifier/`: LLaMA 분류 모델 학습
- `train_generator/`: OpenChat 생성 모델 학습

### 평가 스크립트
- `evaluate_models.py`: 전체 모델 평가
- `tests/compare_models.py`: 베이스 vs 파인튜닝 비교
- `tests/optimized_finetuned_generator.py`: 최적화된 생성

### 유틸리티
- `utils/inference_filter.py`: 텍스트 후처리 필터

## 삭제된 파일
- 불필요한 로그 파일들 (generator_test_*.log)
- 중복/임시 테스트 파일들