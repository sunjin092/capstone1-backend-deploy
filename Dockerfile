FROM python:3.10-slim

# ✅ 시스템 패키지 설치
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# ✅ 작업 디렉토리 설정
WORKDIR /app

# ✅ requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 프로젝트 코드 복사
COPY . .

# ✅ FastAPI 실행 (PORT는 Railway가 환경변수로 넘김)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
