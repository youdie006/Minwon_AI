#!/usr/bin/env python3
"""
AI-Hub 민원 데이터 전처리 스크립트 v2
- 분류용 데이터셋: 8개 카테고리 분류
- 생성용 데이터셋: 민원-요약-질의응답 쌍
"""

import json
import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path(processed_data_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # 카테고리 매핑 (데이터에서 확인된 카테고리)
        self.label_mapping = {
            "교통": 0,
            "환경": 1,
            "안전": 2,
            "복지": 3,
            "문화": 4,
            "경제": 5,
            "주택/건설": 6,
            "기타(국방/세무/방송통신/경찰 등)": 7
        }
        
    def extract_zip_files(self):
        """모든 zip 파일 압축 해제"""
        print("ZIP 파일 압축 해제 중...")
        base_dir = self.raw_path / "24.공공_민원_상담_LLM_사전학습_및_Instruction_Tuning_데이터/3.개방데이터/1.데이터"
        
        # Training과 Validation 데이터 모두 처리
        for split in ["Training", "Validation"]:
            for data_type in ["01.원천데이터", "02.라벨링데이터"]:
                zip_dir = base_dir / split / data_type
                if zip_dir.exists():
                    zip_files = list(zip_dir.glob("*.zip"))
                    for zip_file in tqdm(zip_files, desc=f"{split}/{data_type} 압축 해제"):
                        extract_dir = zip_dir / zip_file.stem
                        if not extract_dir.exists():
                            try:
                                with zipfile.ZipFile(zip_file, 'r') as zf:
                                    zf.extractall(extract_dir)
                            except Exception as e:
                                print(f"압축 해제 실패: {zip_file} - {e}")
                        else:
                            print(f"이미 압축 해제됨: {extract_dir}")
    
    def load_classification_data(self) -> List[Dict]:
        """분류 데이터 로드"""
        all_data = []
        base_dir = self.raw_path / "24.공공_민원_상담_LLM_사전학습_및_Instruction_Tuning_데이터/3.개방데이터/1.데이터"
        
        # Training과 Validation 모두 로드
        for split in ["Training", "Validation"]:
            label_dir = base_dir / split / "02.라벨링데이터"
            
            # 각 기관별 분류 데이터 로드
            for org in ["중앙행정기관", "지방행정기관", "국립아시아문화전당"]:
                cls_dir = label_dir / f"T{'L' if split == 'Training' else 'VL'}_{org}_분류"
                if cls_dir.exists():
                    json_files = list(cls_dir.glob("*.json"))
                    for json_file in tqdm(json_files, desc=f"{split}/{org} 분류 데이터 로딩"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for item in data:
                                    item['split'] = split.lower()
                                    item['organization'] = org
                                    all_data.append(item)
                        except Exception as e:
                            print(f"에러 발생: {json_file} - {e}")
        
        return all_data
    
    def load_generation_data(self) -> List[Dict]:
        """생성 데이터 로드 (요약, 질의응답)"""
        all_data = []
        base_dir = self.raw_path / "24.공공_민원_상담_LLM_사전학습_및_Instruction_Tuning_데이터/3.개방데이터/1.데이터"
        
        for split in ["Training", "Validation"]:
            label_dir = base_dir / split / "02.라벨링데이터"
            
            for org in ["중앙행정기관", "지방행정기관", "국립아시아문화전당"]:
                # 요약 데이터
                summary_dir = label_dir / f"T{'L' if split == 'Training' else 'VL'}_{org}_요약"
                if summary_dir.exists():
                    json_files = list(summary_dir.glob("*.json"))
                    for json_file in tqdm(json_files, desc=f"{split}/{org} 요약 데이터 로딩"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for item in data:
                                    item['split'] = split.lower()
                                    item['organization'] = org
                                    item['task_type'] = 'summary'
                                    all_data.append(item)
                        except Exception as e:
                            print(f"에러 발생: {json_file} - {e}")
                
                # 질의응답 데이터
                qa_dir = label_dir / f"T{'L' if split == 'Training' else 'VL'}_{org}_질의응답"
                if qa_dir.exists():
                    json_files = list(qa_dir.glob("*.json"))
                    for json_file in tqdm(json_files, desc=f"{split}/{org} QA 데이터 로딩"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for item in data:
                                    item['split'] = split.lower()
                                    item['organization'] = org
                                    item['task_type'] = 'qa'
                                    all_data.append(item)
                        except Exception as e:
                            print(f"에러 발생: {json_file} - {e}")
        
        return all_data
    
    def prepare_classification_dataset(self, data: List[Dict]):
        """분류 데이터셋 준비"""
        print("\n분류 데이터셋 준비 중...")
        
        processed_data = []
        label_counts = defaultdict(int)
        
        for item in tqdm(data, desc="분류 데이터 처리"):
            # instructions에서 분류 태스크만 추출
            if 'instructions' in item:
                for inst in item['instructions']:
                    if inst['tuning_type'] == '분류' and inst['data']:
                        for task_data in inst['data']:
                            if task_data['task'] == '분류' and task_data['task_category'] == '상담 주제':
                                # 레이블 원-핫 인코딩
                                label = task_data['output']
                                if label in self.label_mapping:
                                    labels = [0] * len(self.label_mapping)
                                    labels[self.label_mapping[label]] = 1
                                    
                                    processed_data.append({
                                        'text': task_data['input'],
                                        'labels': labels,
                                        'category': label,
                                        'source': item.get('source', ''),
                                        'split': item.get('split', 'training')
                                    })
                                    label_counts[label] += 1
        
        print(f"\n처리된 분류 데이터: {len(processed_data)}개")
        print("카테고리별 분포:")
        for label, count in sorted(label_counts.items()):
            print(f"  - {label}: {count}개")
        
        # Train/Val/Test 분할
        train_data = [d for d in processed_data if d['split'] == 'training']
        val_data = [d for d in processed_data if d['split'] == 'validation']
        
        # Validation 데이터를 Val/Test로 분할
        if val_data:
            val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
        else:
            # Validation 데이터가 없으면 Train에서 분할
            train_data, temp_data = train_test_split(train_data, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # JSON Lines 형식으로 저장
        def save_jsonl(data_list, filename):
            with open(self.processed_path / filename, 'w', encoding='utf-8') as f:
                for item in data_list:
                    # split 필드 제거 (저장 시 불필요)
                    save_item = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(save_item, ensure_ascii=False) + '\n')
        
        save_jsonl(train_data, 'cls_train.jsonl')
        save_jsonl(val_data, 'cls_val.jsonl')
        save_jsonl(test_data, 'cls_test.jsonl')
        
        print(f"\n분류 데이터셋 저장 완료:")
        print(f"  - Train: {len(train_data)}")
        print(f"  - Val: {len(val_data)}")
        print(f"  - Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def prepare_generation_dataset(self, data: List[Dict]):
        """생성 데이터셋 준비"""
        print("\n생성 데이터셋 준비 중...")
        
        processed_data = []
        task_counts = defaultdict(int)
        
        for item in tqdm(data, desc="생성 데이터 처리"):
            if 'instructions' in item:
                for inst in item['instructions']:
                    if inst['data']:
                        for task_data in inst['data']:
                            # 요약과 QA 데이터 처리
                            if inst['tuning_type'] in ['요약', '질의응답']:
                                processed_data.append({
                                    'instruction': task_data['instruction'],
                                    'input': task_data['input'],
                                    'output': task_data['output'],
                                    'task_type': inst['tuning_type'],
                                    'source': item.get('source', ''),
                                    'split': item.get('split', 'training')
                                })
                                task_counts[inst['tuning_type']] += 1
        
        print(f"\n처리된 생성 데이터: {len(processed_data)}개")
        print("태스크별 분포:")
        for task, count in sorted(task_counts.items()):
            print(f"  - {task}: {count}개")
        
        # Train/Val/Test 분할
        train_data = [d for d in processed_data if d['split'] == 'training']
        val_data = [d for d in processed_data if d['split'] == 'validation']
        
        if val_data:
            val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
        else:
            train_data, temp_data = train_test_split(train_data, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # JSON Lines 형식으로 저장
        def save_jsonl(data_list, filename):
            with open(self.processed_path / filename, 'w', encoding='utf-8') as f:
                for item in data_list:
                    save_item = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(save_item, ensure_ascii=False) + '\n')
        
        save_jsonl(train_data, 'gen_train.jsonl')
        save_jsonl(val_data, 'gen_val.jsonl')
        save_jsonl(test_data, 'gen_test.jsonl')
        
        print(f"\n생성 데이터셋 저장 완료:")
        print(f"  - Train: {len(train_data)}")
        print(f"  - Val: {len(val_data)}")
        print(f"  - Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def perform_eda(self, cls_data: List[Dict], gen_data: List[Dict]):
        """탐색적 데이터 분석"""
        print("\n=== EDA 수행 중 ===")
        
        # 분류 데이터 EDA
        cls_df = pd.DataFrame(cls_data)
        cls_df['text_length'] = cls_df['text'].str.len()
        
        # 생성 데이터 EDA
        gen_df = pd.DataFrame(gen_data)
        gen_df['input_length'] = gen_df['input'].str.len()
        gen_df['output_length'] = gen_df['output'].str.len()
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        # 1. 텍스트 길이 분포
        plt.subplot(2, 3, 1)
        plt.hist(cls_df['text_length'], bins=50, edgecolor='black', alpha=0.7)
        plt.title('Classification Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        
        # 2. 카테고리 분포
        plt.subplot(2, 3, 2)
        category_counts = cls_df['category'].value_counts()
        plt.bar(range(len(category_counts)), category_counts.values)
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
        
        # 3. 출처별 분포
        plt.subplot(2, 3, 3)
        source_counts = cls_df['source'].value_counts()
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Data Source Distribution')
        
        # 4. 생성 태스크 분포
        plt.subplot(2, 3, 4)
        task_counts = gen_df['task_type'].value_counts()
        plt.bar(task_counts.index, task_counts.values)
        plt.title('Generation Task Distribution')
        plt.xlabel('Task Type')
        plt.ylabel('Count')
        
        # 5. 입력 길이 분포 (생성)
        plt.subplot(2, 3, 5)
        plt.hist(gen_df['input_length'], bins=50, edgecolor='black', alpha=0.7)
        plt.title('Generation Input Length Distribution')
        plt.xlabel('Input Length')
        plt.ylabel('Frequency')
        
        # 6. 출력 길이 분포 (생성)
        plt.subplot(2, 3, 6)
        plt.hist(gen_df['output_length'], bins=50, edgecolor='black', alpha=0.7)
        plt.title('Generation Output Length Distribution')
        plt.xlabel('Output Length')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.processed_path / 'eda_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 통계 정보 저장
        stats = {
            'classification': {
                'total_samples': len(cls_df),
                'avg_text_length': float(cls_df['text_length'].mean()),
                'max_text_length': int(cls_df['text_length'].max()),
                'min_text_length': int(cls_df['text_length'].min()),
                'category_distribution': category_counts.to_dict(),
                'source_distribution': source_counts.to_dict()
            },
            'generation': {
                'total_samples': len(gen_df),
                'avg_input_length': float(gen_df['input_length'].mean()),
                'avg_output_length': float(gen_df['output_length'].mean()),
                'task_distribution': task_counts.to_dict()
            }
        }
        
        with open(self.processed_path / 'data_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print("\nEDA 완료! 결과가 저장되었습니다.")
    
    def process(self):
        """전체 전처리 프로세스 실행"""
        print("데이터 전처리를 시작합니다...")
        
        # 1. ZIP 파일 압축 해제
        self.extract_zip_files()
        
        # 2. 데이터 로드
        print("\n데이터 로딩 중...")
        cls_data = self.load_classification_data()
        gen_data = self.load_generation_data()
        
        if not cls_data and not gen_data:
            print("데이터를 찾을 수 없습니다.")
            return
        
        # 3. 데이터셋 준비
        if cls_data:
            cls_train, cls_val, cls_test = self.prepare_classification_dataset(cls_data)
        
        if gen_data:
            gen_train, gen_val, gen_test = self.prepare_generation_dataset(gen_data)
        
        # 4. EDA 수행
        if cls_data and gen_data:
            all_cls = cls_train + cls_val + cls_test if cls_data else []
            all_gen = gen_train + gen_val + gen_test if gen_data else []
            self.perform_eda(all_cls, all_gen)
        
        print("\n전처리 완료!")


if __name__ == "__main__":
    processor = DataProcessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed"
    )
    processor.process()