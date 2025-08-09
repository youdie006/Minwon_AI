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

### 분류 모델 (LLaMA-3.1-8B with QLoRA)
- **검증 정확도**: 96.0%
- **F1-Score (Macro)**: 0.756
- **Validation Loss**: 0.355
- **카테고리별 정확도**: 교통 98.47%, 환경 99.72%, 안전 90.85%, 복지 96.81%, 문화 96.81%, 경제 92.65%, 주택/건설 95.84%, 기타 96.81%

### 생성 모델 (OpenChat-3.5 with QLoRA)
- **Final Loss**: 0.71
- **학습 시간**: 8시간 (500 steps)
- **VRAM 사용**: 5.1GB (추론 시)

## 실제 테스트 결과

### 분류 결과
| 카테고리 | 정확도 | F1 Score |
|----------|--------|----------|
| 교통 | 90.0% | 0.00 |
| 환경 | 90.0% | 0.00 |
| 문화 | 100.0% | 1.00 |
| 경제 | 65.0% | 0.36 |
| 주택/건설 | 75.0% | 0.29 |

### 생성 결과

**테스트 1: 민원 요약**
- 📝 지시: 다음 민원 내용을 요약해주세요.
- 📥 입력: 안녕하세요. 저는 oo동에 거주하는 주민입니다. 최근 우리 동네 공원에 쓰레기가 많이 쌓여 있어 불편을 겪고 있습니다...
- 📤 생성: 민원인은 주민들이 쓰레기를 많이 버리고 있어 주민들이 불편하다고 하며, 최근 주말에 음식물 쓰레기와 일회용품이 곳곳에 버려져 있어 악취가 나고 미관상 좋지 않다고 하는데, 구청에서 정기적인 청소와 쓰레기통 추가 설치를 검토해 주시기 바란다. 따라서,

**테스트 2: 답변 생성**
- 📝 지시: 다음 민원에 대한 답변을 작성해주세요.
- 📥 입력: 횡단보도 신호등 시간이 너무 짧아서 노인분들이 건너기 어렵습니다. 신호 시간을 늘려주세요.
- 📤 생성: 1. 안녕하십니까? 국토교통행정에 관심과 애정을 가져주셔서 감사드리며, 귀하께서 신청하신 민원에 대하여 아래와 같이 답변드립니다. 가. 민원 내용 ㅇ 횡단보도 신호등 시간 연장 요청 나. 회신 내용 ㅇ 우리 부에서는 보도 건설 기준 등

**테스트 3: 요구사항 정리**
- 📝 지시: 다음 민원의 핵심 요구사항을 정리해주세요.
- 📥 입력: 우리 아파트 앞 도로에서 밤마다 오토바이 소음이 심합니다...
- 📤 생성: 민원인은 아파트 앞 도로에서 밤마다 오토바이 소음이 심하다고 하며, 특히 새벽 시간대에 굉음을 내며 지나가는 오토바이 때문에 잠을 설치는 날이 많다고 하였다. 민원인은 단속을 강화해 주시고, 방지턱이나 CCTV 설치도 검토해 주기를 요구하고 있다

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