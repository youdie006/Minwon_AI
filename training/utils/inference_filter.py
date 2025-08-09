#!/usr/bin/env python3
"""
생성 텍스트 후처리 필터
- 반복 패턴 감지 및 제거
- 의미없는 문자 패턴 정리
"""

import re
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

class GenerationFilter:
    """생성 텍스트 후처리 필터"""
    
    def __init__(self):
        """초기화"""
        self.min_pattern_length = 3  # 최소 반복 패턴 길이 (단어 수)
        self.max_repeats = 2  # 최대 허용 반복 횟수
        
    def detect_word_repetition(self, text: str, min_length: int = 3) -> Tuple[bool, int]:
        """
        단어 수준 반복 감지
        
        Args:
            text: 검사할 텍스트
            min_length: 최소 패턴 길이 (단어 수)
            
        Returns:
            (반복 여부, 반복 시작 위치)
        """
        words = text.split()
        
        # 다양한 패턴 길이로 검사
        for pattern_len in range(min_length, min(30, len(words) // 2 + 1)):
            for i in range(len(words) - pattern_len * 2 + 1):
                pattern = words[i:i+pattern_len]
                
                # 연속 반복 검사
                repeat_count = 1
                j = i + pattern_len
                while j + pattern_len <= len(words):
                    if words[j:j+pattern_len] == pattern:
                        repeat_count += 1
                        j += pattern_len
                    else:
                        break
                
                # 3회 이상 반복되면 문제로 판단
                if repeat_count >= 3:
                    return True, i + pattern_len
                    
        return False, len(words)
    
    def detect_char_repetition(self, text: str) -> str:
        """
        문자 수준 반복 제거
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            반복 제거된 텍스트
        """
        # 동일 문자 3개 이상 반복 제거
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 특수문자 반복 제거
        text = re.sub(r'([^\w\s])\1{1,}', r'\1', text)
        
        # 의미없는 패턴 제거 (예: ABCDEF... 같은 연속)
        text = re.sub(r'[A-Z]{10,}', '', text)
        
        # 랜덤 문자열 패턴 제거
        text = re.sub(r'[A-Za-z0-9]{20,}', '', text)
        
        return text
    
    def detect_sentence_repetition(self, text: str) -> str:
        """
        문장 수준 반복 제거
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            반복 제거된 텍스트
        """
        # 문장 분리 (간단한 규칙)
        sentences = re.split(r'[.!?]\s+', text)
        
        # 중복 문장 제거
        seen = set()
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent not in seen:
                seen.add(sent)
                result.append(sent)
            elif sent in seen:
                # 중복 발견 시 중단
                break
                
        return '. '.join(result) + ('.' if result and not result[-1].endswith(('.', '!', '?')) else '')
    
    def clean_special_patterns(self, text: str) -> str:
        """
        특수 패턴 정리
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            정리된 텍스트
        """
        # 이모지 및 특수 유니코드 제거
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # 이모티콘
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # 심볼
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # 교통/지도
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # 국기
        
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # 과도한 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 의미없는 특수문자 조합 제거
        text = re.sub(r'[^\w\s가-힣.,!?;:\-\'"()]+', ' ', text)
        
        return text.strip()
    
    def truncate_at_repetition(self, text: str) -> str:
        """
        반복 시작 지점에서 절단
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            절단된 텍스트
        """
        # 단어 반복 검사
        has_repetition, stop_idx = self.detect_word_repetition(text)
        
        if has_repetition:
            words = text.split()
            text = ' '.join(words[:stop_idx])
        
        return text
    
    def filter(self, text: str, max_length: int = 300) -> str:
        """
        종합 필터링
        
        Args:
            text: 처리할 텍스트
            max_length: 최대 길이 (단어 수)
            
        Returns:
            필터링된 텍스트
        """
        if not text:
            return ""
        
        # 1. 특수 패턴 정리
        text = self.clean_special_patterns(text)
        
        # 2. 문자 수준 반복 제거
        text = self.detect_char_repetition(text)
        
        # 3. 반복 지점 절단
        text = self.truncate_at_repetition(text)
        
        # 4. 문장 수준 반복 제거
        text = self.detect_sentence_repetition(text)
        
        # 5. 길이 제한
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
            # 마지막 문장 완성
            last_period = text.rfind('.')
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
        
        # 6. 최종 정리
        text = text.strip()
        
        # 빈 결과 방지
        if len(text) < 10:
            return "죄송합니다. 적절한 답변을 생성하지 못했습니다."
        
        return text
    
    def validate_korean(self, text: str) -> bool:
        """
        한국어 텍스트 유효성 검사
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            한국어 텍스트 여부
        """
        # 한글 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣A-Za-z]', text))
        
        if total_chars == 0:
            return False
            
        korean_ratio = korean_chars / total_chars
        
        # 70% 이상 한글이면 유효
        return korean_ratio > 0.7


def test_filter():
    """필터 테스트"""
    filter_obj = GenerationFilter()
    
    test_cases = [
        # 단어 반복
        "공원 청소가 필요합니다. 공원 청소가 필요합니다. 공원 청소가 필요합니다.",
        
        # 문자 반복
        "안녕하세요!!!!!!! 도와주세요.......",
        
        # 의미없는 패턴
        "답변: ABCDEFGHIJKLMNOP 123456789 xyz xyz xyz",
        
        # 혼합 문제
        "민원 답변입니다. 민원 답변입니다. 민원 답변입니다. AAAAAA !!!",
        
        # 정상 텍스트
        "안녕하세요. 문의하신 공원 청소 건에 대해 답변드립니다. 조속한 시일 내에 처리하겠습니다.",
    ]
    
    print("=== 필터 테스트 ===\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"테스트 {i}:")
        print(f"원본: {test}")
        filtered = filter_obj.filter(test)
        print(f"필터: {filtered}")
        print(f"한국어 유효: {filter_obj.validate_korean(filtered)}")
        print("-" * 60)


if __name__ == "__main__":
    test_filter()