# MinwonAI 모델 학습

한국어 민원 분류 및 생성 모델 학습을 위한 스크립트입니다.

## 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 학습 과정

### 1. 데이터 다운로드

```bash
cd training
./download_data.sh
```

### 2. 데이터 전처리

```bash
python prepare_data.py
```

전처리 결과:
- `data/processed/cls_train.jsonl`: 분류 학습 데이터
- `data/processed/cls_val.jsonl`: 분류 검증 데이터
- `data/processed/gen_train.jsonl`: 생성 학습 데이터
- `data/processed/gen_val.jsonl`: 생성 검증 데이터

### 3. 분류 모델 학습 (QLoRA 4-bit)

```bash
python train_classifier.py
```

- 모델: Meta-Llama-3-7B-Instruct
- 양자화: 4-bit (NF4)
- LoRA: r=8, alpha=16
- VRAM 사용량: ~5GB
- 출력: `ai-service/models/lora_cls_4bit/`

### 4. 생성 모델 학습 (LoRA 8-bit)

```bash
python train_generator.py
```

- 모델: Meta-Llama-3-7B-Instruct
- 양자화: 8-bit
- LoRA: r=16, alpha=32
- VRAM 사용량: ~7GB
- 출력: `ai-service/models/lora_gen_8bit/`

## 학습 모니터링

W&B (Weights & Biases)를 통해 학습 과정을 모니터링할 수 있습니다:

```bash
wandb login
```

## 주의사항

1. RTX 3060 Ti (8GB VRAM) 기준으로 최적화되었습니다.
2. 학습 중 OOM 에러가 발생하면 batch_size를 줄이세요.
3. 데이터 구조가 다른 경우 `prepare_data.py`를 수정해야 합니다.