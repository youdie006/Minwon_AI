#!/usr/bin/env python3
"""
QLoRA 4-bit 분류 모델 학습 스크립트
- Meta-Llama-3-7B-Instruct 기반
- 5개 레이블 다중 분류
- RTX 3060 Ti 8GB 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

class MinwonClassificationDataset(Dataset):
    """민원 분류 데이터셋"""
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 프롬프트 형식으로 텍스트 구성
        text = f"다음 민원을 분류하세요:\n\n{item['text']}\n\n분류:"
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['labels'], dtype=torch.float)
        }

class WeightedBCELoss(nn.Module):
    """클래스 불균형을 위한 가중치 BCE Loss"""
    def __init__(self, pos_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weights = pos_weights
    
    def forward(self, logits, labels):
        if self.pos_weights is not None:
            pos_weight = self.pos_weights.to(logits.device)
        else:
            pos_weight = None
        
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )
        return loss

def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """평가 메트릭 계산"""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Sigmoid 적용 후 임계값 0.5로 이진화
    predictions = torch.sigmoid(torch.tensor(predictions))
    predictions = (predictions > 0.5).float().numpy()
    
    # 메트릭 계산
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    
    # 각 레이블별 F1 스코어
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # 레이블별 정확도
    label_accuracies = []
    for i in range(labels.shape[1]):
        acc = accuracy_score(labels[:, i], predictions[:, i])
        label_accuracies.append(acc)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'label_accuracies': label_accuracies
    }

class CustomTrainer(Trainer):
    """커스텀 Loss를 사용하는 Trainer"""
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def train_classifier():
    """분류 모델 학습 메인 함수"""
    # 설정
    # Llama-3.1-8B-Instruct 사용
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "../ai-service/models/lora_cls_4bit"
    
    # HuggingFace 토큰 설정
    import os
    from dotenv import load_dotenv
    
    # .env 파일에서 환경 변수 로드
    load_dotenv()
    
    # HF_TOKEN이 환경 변수에 없으면 에러 발생
    if "HF_TOKEN" not in os.environ:
        raise ValueError("HF_TOKEN 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    # 토크나이저 로드
    print("토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"]
    )
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 모델 로드
    print("모델 로딩 (4-bit 양자화)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=8,  # 8개 카테고리로 수정
        problem_type="multi_label_classification",
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True
    )
    
    # 4-bit 학습을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    # 패딩 토큰 ID 설정
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    
    # PEFT 모델 생성
    print("LoRA 어댑터 추가...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 데이터셋 로드
    print("데이터셋 로딩...")
    train_dataset = MinwonClassificationDataset(
        "../data/processed/cls_train.jsonl", tokenizer
    )
    val_dataset = MinwonClassificationDataset(
        "../data/processed/cls_val.jsonl", tokenizer
    )
    
    # 클래스 가중치 계산 (불균형 처리)
    all_labels = []
    for item in train_dataset:
        all_labels.append(item['labels'].numpy())
    all_labels = np.array(all_labels)
    
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts
    pos_weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float)
    
    # Loss 함수
    loss_fn = WeightedBCELoss(pos_weights)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 3 epochs로 줄임
        per_device_train_batch_size=2,  # RTX 3060 Ti에 맞게 줄임
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # 총 배치 크기 16
        gradient_checkpointing=True,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=200,  # eval_steps와 동일하게 설정
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",  # wandb 비활성화
        optim="paged_adamw_8bit",  # 8-bit optimizer로 메모리 절약
    )
    
    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer 초기화
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_fn=loss_fn,
    )
    
    # 학습 시작
    print("학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"모델 저장: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 어댑터만 별도 저장
    model.save_pretrained(output_dir)
    
    print("학습 완료!")
    
    # VRAM 사용량 출력
    if torch.cuda.is_available():
        print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    # CUDA 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"사용 가능한 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 학습 실행
    train_classifier()