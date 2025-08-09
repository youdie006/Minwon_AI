# MinwonAI - 공공 민원 상담 AI 시스템

AI-Hub 공공 민원 데이터를 활용한 지능형 민원 처리 시스템

## 프로젝트 개요

공공 민원 상담 LLM 데이터를 활용하여 민원을 자동으로 분류하고 적절한 답변을 생성하는 AI 시스템입니다.

### 주요 기능
- 8개 카테고리 민원 자동 분류 (교통, 환경, 안전, 복지, 문화, 경제, 주택/건설, 기타)
- 민원 내용 자동 요약
- 맞춤형 답변 생성

## 모델 아키텍처

### 1. 분류 모델
- **Base Model**: Meta Llama-3.1-8B-Instruct
- **Fine-tuning**: QLoRA 4-bit 양자화
- **Performance**: 96% 정확도 (검증 데이터셋)

### 2. 생성 모델  
- **Base Model**: OpenChat-3.5-0106 (Mistral-7B 기반)
- **Fine-tuning**: QLoRA 4-bit 양자화
- **Training**: 500 steps, Loss: 0.69

## 프로젝트 구조

```
Minwon_AI/
├── data/
│   ├── raw/                 # AI-Hub 원본 데이터
│   └── processed/           # 전처리된 데이터
├── training/
│   ├── train_classifier/    # 분류 모델 학습
│   └── train_generator/     # 생성 모델 학습
├── ai-service/
│   └── models/             # 학습된 모델
└── bin/                    # 유틸리티 스크립트
```

## 설치 및 실행

### 요구사항
- Python 3.10+
- CUDA 11.8+
- GPU: 8GB+ VRAM (RTX 3060 Ti 이상 권장)

### 환경 설정

```bash
# Conda 환경 생성
conda create -n Minwon_AI python=3.10
conda activate Minwon_AI

# 패키지 설치
pip install -r requirements.txt
```

### 데이터 준비

```bash
# AI-Hub 데이터 다운로드 (API 키 필요)
cd training
./download_data_local.sh

# 데이터 전처리
python prepare_data_v2.py
```

### 모델 학습

```bash
# 분류 모델
cd training/train_classifier
python train_classifier.py

# 생성 모델
cd training/train_generator
python train_generator.py
```

## 성능 지표

### 분류 모델
- Accuracy: 96%
- F1-Score: 0.94
- Training Loss: 0.0196

### 생성 모델
- Final Loss: 0.69
- Perplexity: ~2.0
- Training Steps: 500

## 기술 스택

- **ML Framework**: PyTorch, Transformers, PEFT
- **Quantization**: BitsAndBytes 4-bit
- **Models**: Llama-3.1-8B, OpenChat-3.5
- **Data**: AI-Hub 공공 민원 상담 데이터셋

## 라이선스

이 프로젝트는 AI-Hub 데이터 사용 약관을 따릅니다.

## 참고 자료

- [AI-Hub 공공 민원 상담 데이터](https://aihub.or.kr/)
- [Meta Llama 3.1](https://github.com/meta-llama/llama3)
- [OpenChat](https://github.com/imoneoi/openchat)