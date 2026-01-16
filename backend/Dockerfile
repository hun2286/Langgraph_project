# 1. 파이썬 리눅스 베이스 이미지 사용
FROM python:3.10-slim

# 2. 필수 패키지 설치 (리눅스 환경)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 의존성 설치 (requirements.txt에 gunicorn, uvicorn이 포함되어야 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn uvicorn

# 4. 소스 코드 복사
COPY . .

# 5. 실행 (윈도우에선 안 되었던 gunicorn이 여기선 잘 돌아갑니다)
CMD ["gunicorn", "main:api_app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]