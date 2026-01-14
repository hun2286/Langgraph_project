# 1. 파이썬 3.10 슬림 버전 사용 (가볍고 빠름)
FROM python:3.10-slim

# 2. 필수 시스템 도구 설치 (HuggingFace 모델 빌드용)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 4. 라이브러리 설치 (캐시를 활용해 속도 향상)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 DB 폴더 전체 복사
COPY . .

# 6. FastAPI/Uvicorn 기본 포트 개방
EXPOSE 8000

# 7. 서버 실행 명령어
# 실행 파일명이 agent.py이면 agent:app으로 수정!
CMD ["uvicorn", "main:api_app", "--host", "0.0.0.0", "--port", "8000"]