#!/usr/bin/env python3
"""
LoRA 4-bit 생성 모델 학습 스크립트
- OpenChat-3.5-0106 기반 (Mistral 7B)
- 요약 및 질의응답 생성
- RTX 3060 Ti 8GB 최적화
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import warnings
warnings.filterwarnings("ignore")

class MinwonGenerationDataset(Dataset):
    """민원 생성 데이터셋 (요약/QA)"""
    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
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
        
        # OpenChat 3.5 프롬프트 템플릿
        prompt = f"""GPT4 Correct User: {item['instruction']}

{item['input']}<|end_of_turn|>GPT4 Correct Assistant: {item['output']}<|end_of_turn|>"""
        
        # 토크나이징
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 레이블 설정 (입력 부분은 -100으로 마스킹)
        labels = encoding['input_ids'].clone()
        
        # assistant 응답 부분만 학습하도록 설정
        response_start = prompt.find('GPT4 Correct Assistant:')
        response_tokens_start = len(self.tokenizer.encode(prompt[:response_start]))
        labels[:, :response_tokens_start] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def compute_perplexity(eval_preds) -> Dict:
    """Perplexity 계산"""
    loss = eval_preds.predictions.mean()
    perplexity = np.exp(loss)
    return {"perplexity": perplexity}

def train_generator():
    """생성 모델 학습 메인 함수"""
    # 설정
    model_id = "openchat/openchat-3.5-0106"
    output_dir = "../../ai-service/models/lora_gen_4bit"
    
    # 토크나이저 로드
    print("토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 4-bit 학습을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정 (Mistral 아키텍처에 맞춤)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # PEFT 모델 생성
    print("LoRA 어댑터 추가...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 데이터셋 로드
    print("데이터셋 로딩...")
    train_dataset = MinwonGenerationDataset(
        "../../data/processed/gen_train.jsonl", tokenizer
    )
    val_dataset = MinwonGenerationDataset(
        "../../data/processed/gen_val.jsonl", tokenizer
    )
    # 평가 데이터셋 축소 (1099개 → 100개) for faster evaluation
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices[:min(len(indices), len(dataset))]
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    val_dataset = SubsetDataset(val_dataset, list(range(100)))
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=50,
        max_steps=500,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=25,
        save_steps=200,  # 체크포인트 저장 (eval_steps와 동일)
        eval_strategy="steps",
        eval_steps=200,  # 평가 주기를 늘려서 학습 속도 개선
        save_strategy="steps",
        save_total_limit=3,  # 체크포인트 3개 유지
        load_best_model_at_end=True,
        report_to="none",
        run_name="minwon_gen_lora_4bit",
        optim="paged_adamw_32bit",
        group_by_length=True,
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_perplexity,
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

def test_generation(model_path: str):
    """학습된 모델 테스트"""
    print("\n모델 테스트...")
    
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        "openchat/openchat-3.5-0106",
        load_in_4bit=True,
        device_map="auto",
    )
    
    # LoRA 어댑터 로드
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    # 테스트 프롬프트
    test_prompts = [
        {
            "instruction": "다음 민원 내용을 요약해주세요.",
            "input": "안녕하세요. 저는 oo동에 거주하는 주민입니다. 최근 우리 동네 공원에 쓰레기가 많이 쌓여 있어 불편을 겪고 있습니다. 특히 주말이 지나면 음식물 쓰레기와 일회용품이 곳곳에 버려져 있어 악취가 나고 미관상 좋지 않습니다. 구청에서 정기적인 청소와 쓰레기통 추가 설치를 검토해 주시기 바랍니다."
        },
        {
            "instruction": "다음 민원 내용을 읽고 질문에 답해주세요.\n\n질문: 민원인이 요청하는 사항은 무엇인가요?",
            "input": "횡단보도 신호등 시간이 너무 짧아서 노인분들이 건너기 어렵습니다. 신호 시간을 늘려주세요."
        }
    ]
    
    for prompt_data in test_prompts:
        prompt = f"""GPT4 Correct User: {prompt_data['instruction']}

{prompt_data['input']}<|end_of_turn|>GPT4 Correct Assistant:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n질문: {prompt_data['instruction']}")
        print(f"응답: {response.split('assistant')[-1].strip()}")

if __name__ == "__main__":
    # CUDA 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"사용 가능한 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # wandb 초기화 (비활성화)
    # wandb.init(project="minwon-ai", name="generator-lora-4bit")
    
    # 학습 실행
    train_generator()
    
    # 테스트 실행 (옵션)
    # test_generation("../../ai-service/models/lora_gen_4bit")