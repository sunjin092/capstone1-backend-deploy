from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import json
from analyze_image_skin_type import run_analysis
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ✅ CORS 허용 설정
origins = [
    "http://localhost:3000",
    "https://jiwow-wow.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 화장품 CSV 로드
products = pd.read_csv("Total_DB.csv", encoding='cp949')

# ✅ 추천 함수
def recommend_products(result):
    regions = result.get("regions", {})
    if not regions:
        return []

    try:
        moisture_avg = np.mean([
            regions['이마']['수분'],
            regions['왼쪽 볼']['수분'],
            regions['오른쪽 볼']['수분'],
            regions['턱']['수분']
        ])
        elasticity_avg = np.mean([
            regions['이마']['탄력'],
            regions['왼쪽 볼']['탄력'],
            regions['오른쪽 볼']['탄력'],
            regions['턱']['탄력']
        ])
        pore_avg = np.mean([
            regions['왼쪽 볼']['모공 개수'],
            regions['오른쪽 볼']['모공 개수']
        ])
        pigment_avg = regions['전체']['색소침착 개수']
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

    def score_product(row):
        tags = str(row['태그'])
        detail = str(row.get('세부', ''))
        score = 0
        weights = [3, 2, 1]
        for idx, concern in enumerate(user_concerns):
            weight = weights[idx] if idx < len(weights) else 1
            for keyword in concern_keywords.get(concern, []):
                if keyword in tags:
                    score += weight
        return score

    products['score'] = products.apply(score_product, axis=1)
    top5 = products[products['score'] >= 0].sort_values(by='score', ascending=False).head(5)

    return top5[['브랜드', '제품명', '용량/가격', '별점', '이미지']].to_dict(orient='records')

# ✅ FastAPI 엔드포인트
@app.post("/analyze-recommend")
async def analyze_and_recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = run_analysis(image_bytes)
        recommended = recommend_products(result)
        return JSONResponse(content={"analysis": result, "recommend": recommended})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)