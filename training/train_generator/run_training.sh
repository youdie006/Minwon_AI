#!/bin/bash

# 타임스탬프 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="generator_training_${TIMESTAMP}.log"

echo "생성 모델 학습 시작: $(date)"
echo "로그 파일: $LOG_FILE"
echo "체크포인트는 50스텝마다 저장됩니다."

# tmux 세션 생성 및 학습 시작
tmux new-session -d -s train_gen "python train_generator.py 2>&1 | tee -a $LOG_FILE"

echo ""
echo "학습이 백그라운드에서 시작되었습니다."
echo "진행상황 확인: tmux attach -t train_gen"
echo "로그 확인: tail -f $LOG_FILE"
echo "세션 종료: tmux kill-session -t train_gen"