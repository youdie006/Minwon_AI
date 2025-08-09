#!/bin/bash

# 타임스탬프 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="generator_final_${TIMESTAMP}.log"

echo "생성 모델 최종 학습: $(date)"
echo "체크포인트: checkpoint-400 (마지막 저장 지점)"
echo "중단 지점: 462/500 스텝"
echo "남은 스텝: 약 38스텝"
echo "로그 파일: $LOG_FILE"
echo ""

# GPU 상태 확인
nvidia-smi
echo ""

# conda 환경 활성화 및 학습 시작
tmux new-session -d -s train_final "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Minwon_AI && python train_generator.py 2>&1 | tee -a $LOG_FILE"

echo "최종 학습이 시작되었습니다."
echo "진행상황 확인: tmux attach -t train_final"
echo "로그 확인: tail -f $LOG_FILE"
echo "세션 종료: tmux kill-session -t train_final"
echo ""
echo "예상 완료 시간: 약 2-3시간 (38스텝 × 4분)"