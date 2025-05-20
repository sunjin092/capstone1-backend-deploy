FROM python:3.10

# 👇 OpenCV 라이브러리 설치
RUN apt-get update && apt-get install -y libgl1

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
