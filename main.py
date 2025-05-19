# ✅ main.py (CV 분석 + 추천 시스템 완전 통합)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import pandas as pd
import json
import io
import numpy as np
import math
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import mediapipe as mp
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

# FastAPI 앱 생성
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

# 공통 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

regression_ckpt = os.path.join("checkpoint", "regression")
regression_num_output = [1, 2, 0, 0, 0, 3, 3, 0, 2]
area_label = {0: "전체", 1: "이마", 2: "미간", 3: "왼쪽 눈가", 4: "오른쪽 눈가", 5: "왼쪽 볼", 6: "오른쪽 볼", 7: "입술", 8: "턱"}
reg_desc = {0: ["색소침착 개수"], 1: ["수분", "탄력"], 5: ["수분", "탄력", "모공 개수"], 6: ["수분", "탄력", "모공 개수"], 8: ["수분", "탄력"]}

REGION_LANDMARKS = {
    0: list(range(468)),
    1: [10, 67, 69, 71, 109, 151, 337, 338, 297],
    2: [168, 6, 197, 195, 5, 4],
    3: [130, 133, 160, 159, 158],
    4: [359, 362, 386, 385, 384],
    5: [205, 50, 187, 201, 213],
    6: [425, 280, 411, 427, 434],
    7: [13, 14, 17, 84, 181],
    8: [152, 377, 400, 378, 379]
}

def crop_regions_by_ratio(pil_img):
    img = np.array(pil_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    regions = [None] * 9
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("❗ 얼굴을 찾을 수 없습니다.")
        landmarks = results.multi_face_landmarks[0].landmark
        points = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        face_x1, face_y1 = np.min(points, axis=0)
        face_x2, face_y2 = np.max(points, axis=0)
        face_w, face_h = face_x2 - face_x1, face_y2 - face_y1
        for idx, lm_indices in REGION_LANDMARKS.items():
            pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in lm_indices])
            cx, cy = np.mean(pts, axis=0)
            if idx == 8: cx -= face_w * 0.15
            if idx == 1:
                box_w, box_h = int(face_w * 0.70), int(face_h * 0.3)
                cy -= box_h * 0.2
            elif idx == 2:
                box_w, box_h = int(face_w * 0.35), int(face_h * 0.15)
                cy -= box_h * 2.5
            else:
                box_w, box_h = int(face_w * 0.28), int(face_h * 0.25)
            x1, y1 = max(int(cx - box_w / 2), 0), max(int(cy - box_h / 2), 0)
            x2, y2 = min(int(cx + box_w / 2), w), min(int(cy + box_h / 2), h)
            crop = img[y1:y2, x1:x2]
            regions[idx] = Image.fromarray(crop)
    return regions

def load_average_data(path):
    df = pd.read_csv(path, encoding='cp949')
    avg_dict = {}
    for _, row in df.iterrows():
        key = row["성별"]
        avg_dict[key] = {
            "mean": {
                "수분_이마": row["수분_이마"],
                "수분_왼쪽 볼": row["수분_왼쪽 볼"],
                "수분_오른쪽 볼": row["수분_오른쪽 볼"],
                "수분_턱": row["수분_턱"],
                "탄력_이마": row["탄력_이마"],
                "탄력_왼쪽 볼": row["탄력_왼쪽 볼"],
                "탄력_오른쪽 볼": row["탄력_오른쪽 볼"],
                "탄력_턱": row["탄력_턱"],
                "모공 개수_왼쪽 볼": row["모공 개수_왼쪽 볼"],
                "모공 개수_오른쪽 볼": row["모공 개수_오른쪽 볼"],
                "색소침착 개수_전체": row["색소침착 개수_전체"]
            },
            "std": {
                "수분_이마": row["수분_이마_표준편차"],
                "수분_왼쪽 볼": row["수분_왼쪽 볼_표준편차"],
                "수분_오른쪽 볼": row["수분_오른쪽 볼_표준편차"],
                "수분_턱": row["수분_턱_표준편차"],
                "탄력_이마": row["탄력_이마_표준편차"],
                "탄력_왼쪽 볼": row["탄력_왼쪽 볼_표준편차"],
                "탄력_오른쪽 볼": row["탄력_오른쪽 볼_표준편차"],
                "탄력_턱": row["탄력_턱_표준편차"],
                "모공 개수_왼쪽 볼": row["모공 개수_왼쪽 볼_표준편차"],
                "모공 개수_오른쪽 볼": row["모공 개수_오른쪽 볼_표준편차"],
                "색소침착 개수_전체": row["색소침착 개수_전체_표준편차"]
            }
        }
    return avg_dict

def compute_z_score(value, mean, std):
    if std == 0:
        return 0
    return round((value - mean) / std, 2)

def safe_float(val, default=0.0):
    try:
        if val is None or math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    except:
        return default

def get_restore_stats():
    return {
        1: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9)},
        5: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9), "모공 개수": "log"},
        6: {"수분": (60.2, 9.6),  "탄력": (49.3, 12.1), "모공 개수": "log"},
        8: {"수분": (61.3, 10.0), "탄력": (47.5, 12.0)},
        0: {"색소침착 개수": 300}
    }

# 회귀 모델 로딩
restore_stats = get_restore_stats()
reg_models = []
for idx, out_dim in enumerate(regression_num_output):
    if out_dim == 0:
        reg_models.append(None)
        continue
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    ckpt_path = os.path.join(regression_ckpt, str(idx), "state_dict.bin")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        model.eval()
        reg_models.append(model.to(device))
    else:
        reg_models.append(None)

def model_image(image: Image.Image, gender_age: str, average_data_path="average_data.csv") -> dict:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    regions = crop_regions_by_ratio(image)
    result = {"regions": {}, "z_scores": {}, "priority_concern": None}

    avg_data = load_average_data(average_data_path)
    if gender_age not in avg_data:
        raise ValueError(f"❗ 평균 데이터에 '{gender_age}' 항목이 없습니다.")
    mean_dict = avg_data[gender_age]["mean"]
    std_dict = avg_data[gender_age]["std"]

    z_score_list = []

    for idx in range(9):
        if reg_models[idx] is None or regions[idx] is None or idx in [3, 4]:
            continue
        crop_tensor = transform(regions[idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = reg_models[idx](crop_tensor).squeeze().cpu().numpy()
        if output.ndim == 0:
            output = [output]
        sub_result, sub_zscore = {}, {}
        for i, val in enumerate(output):
            label = reg_desc[idx][i]
            if label == "모공 개수" and restore_stats[idx].get(label) == "log":
                val = np.clip(np.exp(val) - 1, 0, 2500)
            elif isinstance(restore_stats[idx].get(label), tuple):
                mean, std = restore_stats[idx][label]
                val = val * std + mean
            elif label == "색소침착 개수":
                val *= 300
            val = float(val)
            sub_result[label] = round(val, 2)
            z_key = f"{label}_{area_label[idx]}" if label != "색소침착 개수" else "색소침착 개수_전체"
            if z_key in mean_dict:
                z = compute_z_score(val, mean_dict[z_key], std_dict[z_key])
                sub_zscore[label] = z
                z_score_list.append((label, area_label[idx], z))
        result["regions"][area_label[idx]] = sub_result
        if sub_zscore:
            result["z_scores"][area_label[idx]] = sub_zscore

    concerns = []
    for label, area, z in z_score_list:
        if label in ["수분", "탄력"] and z < -0.2:
            concerns.append((label, area, z))
        elif label in ["모공 개수", "색소침착 개수"] and z > 0.2:
            concerns.append((label, area, z))
    if concerns:
        result["priority_concern"] = sorted(concerns, key=lambda x: abs(x[2]), reverse=True)[0]

    return result

# 화장품 CSV 로드
products = pd.read_csv("Total_DB.csv", encoding='cp949')

# 추천 함수 정의
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
        priority_label = priority_concern[0]
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
    concerns: Optional[str] = Form(None)  # JSON 문자열
):
    image_bytes = await file.read()
    try:
        gender_age = f"{gender}_{(age // 10) * 10}대"
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model_image(image, gender_age)

        user_selected_concerns = json.loads(concerns) if concerns else None
        recommended = recommend_products(
            skin_type=None,
            regions=result.get("regions"),
            priority_concern=result.get("priority_concern"),
            user_selected_concerns=user_selected_concerns
        )

        response_data = {
            "analysis": result,
            "recommend": recommended
        }

        safe_json = json.dumps(response_data, ensure_ascii=False, allow_nan=False)
        return JSONResponse(content=json.loads(safe_json))

    except Exception as e:
        print("🚨 처리 에러:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)