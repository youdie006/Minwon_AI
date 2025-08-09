#!/bin/bash

# 타임스탬프 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="generator_resume_${TIMESTAMP}.log"

echo "생성 모델 학습 재개: $(date)"
echo "체크포인트: checkpoint-200 (200/500 스텝 완료)"
echo "로그 파일: $LOG_FILE"
echo ""

# GPU 메모리 정리
nvidia-smi
echo "GPU 메모리 정리 중..."
sleep 2

# tmux 세션 생성 및 학습 재개
tmux new-session -d -s train_gen_resume "python train_generator.py 2>&1 | tee -a $LOG_FILE"

echo "학습이 재개되었습니다."
echo "진행상황 확인: tmux attach -t train_gen_resume"
echo "로그 확인: tail -f $LOG_FILE"
echo "세션 종료: tmux kill-session -t train_gen_resume"
echo ""
echo "남은 스텝: 300/500"
echo "예상 완료 시간: 약 5-6시간"