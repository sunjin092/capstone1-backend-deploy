# ✅ main.py (추천 시스템 완전 교체 + concerns 입력 연동)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pandas as pd
import json
import io
import numpy as np
import math
from typing import List, Optional
from analyze_image_skin_type import model_image

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000",
    "https://jiwow-wow.github.io",
    "https://front-seven-chi.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 화장품 데이터 로드
products = pd.read_csv("Total_DB.csv", encoding='cp949')

# 안전한 float 변환 함수
def safe_float(val, default=0.0):
    try:
        if val is None or math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    except:
        return default

# 새로운 추천 시스템
from sklearn.preprocessing import StandardScaler

def recommend_products(skin_type: str, regions: dict, priority_concern: Optional[tuple], user_selected_concerns: Optional[List[str]] = None):
    moisture_values = {
        '이마': regions.get('이마', {}).get('수분', 0),
        '왼쪽 볼': regions.get('왼쪽 볼', {}).get('수분', 0),
        '오른쪽 볼': regions.get('오른쪽 볼', {}).get('수분', 0),
        '턱': regions.get('턱', {}).get('수분', 0)
    }
    elasticity_avg = np.mean([
        regions.get('이마', {}).get('탄력', 0),
        regions.get('왼쪽 볼', {}).get('탄력', 0),
        regions.get('오른쪽 볼', {}).get('탄력', 0),
        regions.get('턱', {}).get('탄력', 0)
    ])
    pore_avg = np.mean([
        regions.get('왼쪽 볼', {}).get('모공 개수', 0),
        regions.get('오른쪽 볼', {}).get('모공 개수', 0)
    ])
    pigment_avg = regions.get('전체', {}).get('색소침착 개수', 0)

    low_moisture_vals = [v for v in moisture_values.values() if v < 62]
    if len(low_moisture_vals) > 0:
        low_moisture_avg = np.mean(low_moisture_vals)
        moisture_score = (62 - low_moisture_avg) / 62
    else:
        moisture_score = 0

    raw_scores = [
        (pore_avg - 400) / 400 if pore_avg >= 400 else 0,
        (50 - elasticity_avg) / 50 if elasticity_avg <= 50 else 0,
        moisture_score,
        (pigment_avg - 130) / 130 if pigment_avg >= 130 else 0
    ]
    concern_keys = ['모공', '탄력', '수분', '색소침착']

    scaler = StandardScaler()
    scaled_scores = scaler.fit_transform(np.array(raw_scores).reshape(-1, 1)).flatten()
    concern_scores = dict(zip(concern_keys, scaled_scores))

    if priority_concern:
        priority_label = priority_concern[0]  # label
        user_concerns = [priority_label]
    else:
        user_concerns = []

    if user_selected_concerns is None:
        user_selected_concerns = ['트러블']

    concern_keywords = {
        '모공': ['모공관리', '모공케어', '피지조절', '노폐물제거', '안티폴루션','BHA', 'LHA'],
        '탄력': ['피부탄력', '주름개선', '피부장벽강화', '피부재생', '영양공급', '앰플', '피부활력', '생기부여'],
        '수분': ['수분공급', '보습', '고보습', '피부유연', '피부결정돈', '피부장벽강화', '멀티크림', '밤타입', '피부보호', '피부활력', '보습패드','AHA', 'PHA','유수분조절','유수분밸런스'],
        '색소침착': ['비타민함유','AHA','스팟케어']
    }
    user_concern_keywords = {
        '트러블': ['트러블케어', '약산성', '저자극', '민감성', '피지조절', '노폐물제거', '피부진정', '스팟케어', '피부재생', '오일프리', '안티폴루션','BHA', 'LHA'],
        '피부톤': ['미백', '브라이트닝', '톤업', '피부톤보정', '투명피부', '광채', '생기부여', '피부활력', '비타민함유','다크서클완화','안티다크닝'],
        '각질/피부결': ['각질관리', '각질케어', '피부결정돈', '피부유연', 'AHA', 'BHA', 'PHA', 'LHA', '피지조절', '보습', '고보습','노폐물제거', '피부장벽강화'],
        '민감성': ['민감성', '저자극', '약산성', '피부진정', '피부보호', '클린뷰티', '피부장벽강화', '비건뷰티', '크루얼티프리','PHA', 'LHA','안티폴루션'],
        '자외선 차단': ['자외선차단'],
        '유기농': ['유기농화장품', '클린뷰티', '제로웨이스트', '친환경', '비건뷰티', '크루얼티프리', '한방화장품']
    }

    def score_product(row):
        tags = str(row['태그'])
        tag_set = set([tag.strip() for tag in tags.split(',')])
        score = 0
        for concern in concern_keywords:
            for keyword in concern_keywords[concern]:
                for tag in tag_set:
                    if keyword == tag:
                        if concern in user_concerns and any(keyword in user_concern_keywords.get(u, []) for u in user_selected_concerns):
                            score += 5
                        elif concern in user_concerns:
                            score += 3
                        elif any(keyword in user_concern_keywords.get(u, []) for u in user_selected_concerns):
                            score += 2
                        break
        return score

    products['score'] = products.apply(score_product, axis=1)
    recommended = products[products['score'] > 0].sort_values(by='score', ascending=False).head(5)

    def safe_row(row):
        return {
            "브랜드": str(row.get("브랜드", "")),
            "제품명": str(row.get("제품명", "")),
            "용량/가격": str(row.get("용량/가격", "")),
            "별점": safe_float(row.get("별점", 0.0)),
            "이미지": str(row.get("이미지", ""))
        }

    return [safe_row(row) for _, row in recommended.iterrows()]

# ✅ 최종 API 엔드포인트
@app.post("/analyze-recommend")
async def analyze_and_recommend(
    file: UploadFile = File(...),
    gender: str = Form(...),
    age: int = Form(...),
    concerns: Optional[str] = Form(None)  # JSON 문자열 형태로 받음
):
    image_bytes = await file.read()
    try:
        gender_age = f"{gender}_{(age // 10) * 10}대"
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model_image(image, gender_age)
        user_selected_concerns = json.loads(concerns) if concerns else None
        recommended = recommend_products(
            skin_type=result.get("skin_type"),
            regions=result.get("regions"),
            priority_concern=result.get("priority_concern"),
            user_selected_concerns=user_selected_concerns
        )
        response_data = {"analysis": result, "recommend": recommended}
        safe_json = json.dumps(response_data, ensure_ascii=False, allow_nan=False)
        return JSONResponse(content=json.loads(safe_json))

    except Exception as e:
        print("🚨 처리 에러:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
