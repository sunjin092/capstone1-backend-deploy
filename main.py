from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import json
from analyze_image_skin_type import run_analysis
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import os
import zipfile
import requests
import math

def download_checkpoints():
    url = "https://drive.google.com/uc?id=1uR2MqrKcm9K4PxAEVD-giXIEQX82H5ZV"
    zip_path = "checkpoint.zip"

    if not os.path.exists("checkpoint"):
        print("📦 체크포인트 다운로드 중...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)
        print("✅ 체크포인트 다운로드 완료!")

download_checkpoints()

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://jiwow-wow.github.io",
    "https://front-seven-chi.vercel.app"  # ✅ 네 Vercel 프론트 주소 추가됨
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

products = pd.read_csv("Total_DB.csv", encoding='cp949')

def safe_float(val, default=0.0):
    try:
        if val is None or math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    except:
        return default

def recommend_products(result):
    regions = result.get("regions", {})
    skin_type = result.get("skin_type", "중성")

    if not regions or not skin_type:
        return []

    try:
        moisture_avg = safe_float(np.mean([
            regions['이마']['수분'],
            regions['왼쪽 볼']['수분'],
            regions['오른쪽 볼']['수분'],
            regions['턱']['수분']
        ]))
        elasticity_avg = safe_float(np.mean([
            regions['이마']['탄력'],
            regions['왼쪽 볼']['탄력'],
            regions['오른쪽 볼']['탄력'],
            regions['턱']['탄력']
        ]))
        pore_avg = safe_float(np.mean([
            regions['왼쪽 볼']['모공 개수'],
            regions['오른쪽 볼']['모공 개수']
        ]))
        pigment_avg = safe_float(regions['전체']['색소침착 개수'])
    except:
        return []

    concern_scores = {
        '모공': (pore_avg - 500) / 500 if pore_avg >= 500 else 0,
        '주름': (50 - elasticity_avg) / 50 if elasticity_avg <= 50 else 0,
        '수분': (55 - moisture_avg) / 55 if moisture_avg <= 55 else 0,
        '색소침착': (pigment_avg - 130) / 130 if pigment_avg >= 130 else 0,
    }
    user_concerns = [k for k, v in sorted(concern_scores.items(), key=lambda x: x[1], reverse=True) if v > 0]

    concern_keywords = {
        '모공': ['모공', '피지','노폐물','피부결','각질'],
        '주름': ['주름', '탄력','영양공급','피부활력','피부재생','나이트','아이'],
        '수분': ['수분', '보습'],
        '색소침착': ['미백', '브라이트닝', '비타민', '피부톤', '투명','트러블케어','피부재생','피부보호','스팟','저자극','진정']
    }
    exclude_keywords = {
        '지성': ['페이스오일', '멀티밤', '보습크림', '나이트크림'],
        '건성': ['워터토너', '브라이트닝'],
        '복합건성': [],
        '복합지성': [],
        '중성': []
    }

    def score_product(row):
        tags = str(row['태그'])
        detail = str(row.get('세부', ''))
        score = 0

        for block_word in exclude_keywords.get(skin_type, []):
            if block_word in detail or block_word in tags:
                return -1

        weights = [3, 2, 1]
        for idx, concern in enumerate(user_concerns):
            weight = weights[idx] if idx < len(weights) else 1
            for keyword in concern_keywords.get(concern, []):
                if keyword in tags:
                    score += weight
        return score

    products['score'] = products.apply(score_product, axis=1)
    top5 = products[products['score'] >= 0].sort_values(by='score', ascending=False).head(5)

    def safe_row(row):
        return {
            "브랜드": str(row.get("브랜드", "")),
            "제품명": str(row.get("제품명", "")),
            "용량/가격": str(row.get("용량/가격", "")),
            "별점": safe_float(row.get("별점", 0.0)),
            "이미지": str(row.get("이미지", ""))
        }

    return [safe_row(row) for _, row in top5.iterrows()]

@app.post("/analyze-recommend")
async def analyze_and_recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = run_analysis(image_bytes)
        recommended = recommend_products(result)

        response_data = {"analysis": result, "recommend": recommended}
        safe_json = json.dumps(response_data, ensure_ascii=False, allow_nan=False)
        return JSONResponse(content=json.loads(safe_json))

    except Exception as e:
        print("🚨 직렬화 또는 처리 에러:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
