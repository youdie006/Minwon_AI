# Contributing Guide

## 브랜치 전략

### 브랜치 구조
- `main`: 프로덕션 준비 완료 코드
- `develop`: 개발 통합 브랜치
- `feature/*`: 새 기능 개발
- `fix/*`: 버그 수정
- `hotfix/*`: 긴급 수정

### 브랜치 네이밍 규칙
```
feature/[기능명]  예: feature/api-inference
fix/[이슈번호]   예: fix/issue-23
hotfix/[설명]     예: hotfix/memory-leak
```

## 개발 워크플로우

### 1. 새 기능 개발
```bash
# develop 브랜치에서 시작
git checkout develop
git pull origin develop

# 기능 브랜치 생성
git checkout -b feature/새기능명

# 개발 및 커밋
git add .
git commit -m "feat: 새 기능 설명"

# 푸시 및 PR 생성
git push origin feature/새기능명
```

### 2. PR 규칙
- PR 제목: `[Type] 간단한 설명`
- Type: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- 본문에 변경사항 상세 설명
- 관련 이슈 번호 링크

### 3. 자동 코드 리뷰 (Gemini Code Assist)
- PR 생성 시 Gemini Code Assist가 자동으로 코드 리뷰 수행
- 보안 취약점, 성능 이슈, 코드 품질 자동 검토
- 리뷰 코멘트 확인 후 수정사항 반영

### 4. 커밋 메시지 규칙
```
type: 제목 (50자 이내)

본문 (선택사항, 72자 줄바꿈)

Closes #이슈번호
```

## 계획된 기능 브랜치

### Phase 1: API 개발
- `feature/api-inference`: FastAPI 추론 서버
- `feature/model-optimization`: 모델 최적화

### Phase 2: 배포
- `feature/modal-deploy`: Modal 서버리스 배포
- `feature/docker-setup`: Docker 컨테이너화

### Phase 3: 프론트엔드
- `feature/frontend-ui`: Vite React SPA
- `feature/netlify-functions`: 서버리스 API

### Phase 4: 데이터베이스
- `feature/supabase-integration`: Supabase 연동
- `feature/redis-cache`: Upstash Redis 캐싱

## 코드 리뷰 체크리스트

### 자동 리뷰 (Gemini Code Assist)
- 코드 품질 및 가독성
- 보안 취약점 검사
- 성능 최적화 제안
- 베스트 프랙티스 준수

### 수동 리뷰
- [ ] 비즈니스 로직이 요구사항과 일치하는가?
- [ ] 테스트 커버리지가 충분한가?
- [ ] 문서가 업데이트되었는가?
- [ ] 에러 처리가 적절한가?
- [ ] 코드 중복이 없는가?

## 테스트
```bash
# 단위 테스트
pytest tests/

# 모델 추론 테스트
python -m pytest tests/test_inference.py

# 통합 테스트
python -m pytest tests/integration/
```

## PR 머지 기준
1. Gemini Code Assist 리뷰 통과
2. 모든 테스트 통과
3. 최소 1명의 리뷰어 승인
4. 충돌 해결 완료

## 문의
Issues 탭을 통해 문의해주세요.