#!/bin/bash

# AI-Hub 민원 데이터셋 다운로드 스크립트 (로컬 설치 버전)
# Dataset ID: 71852

echo "AI-Hub 데이터셋 다운로드를 시작합니다..."

# 로컬 bin 디렉토리 생성
mkdir -p ../bin

# aihubshell 다운로드 및 로컬 설치
if [ ! -f ../bin/aihubshell ]; then
    echo "aihubshell을 다운로드합니다..."
    curl -o ../bin/aihubshell https://api.aihub.or.kr/api/aihubshell.do
    chmod +x ../bin/aihubshell
fi

# 데이터 디렉토리 생성
mkdir -p ../data/raw

# API 키 설정
API_KEY='1af93171-c405-4d12-bae5-9c4667bd88a0'

# 데이터 다운로드
echo "데이터셋 71852를 다운로드합니다..."
../bin/aihubshell -mode d -datasetkey 71852 -aihubapikey "$API_KEY" -dir ../data/raw -thread 8

# 다운로드 결과 확인
if [ -d "../data/raw/71852" ]; then
    cd ../data/raw/71852
    
    # 분할된 압축 파일 확인
    if ls *.zip.part* 1> /dev/null 2>&1; then
        echo "분할된 압축 파일을 합칩니다..."
        for b in $(find . -name '*.zip.part*' | sed -E 's/\.zip\.part.*$//' | sort -u); do
            find $(dirname $b) -name "$(basename $b).zip.part*" -print0 | \
            sort -zt'.' -k2V | xargs -0 cat > "${b}.zip"
        done
    fi
    
    # 압축 해제 (unzip이 없으면 python 사용)
    if command -v unzip &> /dev/null; then
        echo "압축을 해제합니다..."
        for z in *.zip; do
            unzip -q "$z" -d "${z%.zip}"
        done
    else
        echo "unzip이 없어 Python으로 압축을 해제합니다..."
        python3 -c "
import zipfile
import os
import glob

for zip_file in glob.glob('*.zip'):
    extract_dir = os.path.splitext(zip_file)[0]
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(extract_dir)
    print(f'압축 해제: {zip_file} -> {extract_dir}')
"
    fi
    
    echo "다운로드 완료!"
else
    echo "데이터 다운로드에 실패했습니다. AI-Hub 로그인이 필요할 수 있습니다."
    echo "브라우저에서 AI-Hub에 로그인 후 데이터셋을 직접 다운로드해주세요."
    echo "데이터셋 ID: 71852"
fi