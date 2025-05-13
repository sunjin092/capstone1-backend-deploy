import json
import pandas as pd
import numpy as np

# 안전한 float 변환 함수
import math
def safe_float(val, default=0.0):
    try:
        if val is None or math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    except:
        return default

# CSV 파일 경로
csv_path = "Total_DB.csv"
products = pd.read_csv(csv_path, encoding='cp949')

# 예시 JSON 입력 (실제 서비스에서는 이 부분을 API에서 받음)
raw_json = '{"regions":{"오른쪽 볼":{"모공 개수":750.84,"수분":64.9,"탄력":47.93},"왼쪽 볼":{"모공 개수":778.89,"수분":63.88,"탄력":65.1},"이마":{"수분":67.01,"탄력":50.42},"전체":{"색소침착 개수":155.41},"턱":{"수분":62.94,"탄력":43.34}}}'
decoded = json.loads(raw_json)
regions = decoded["regions"]

# 평균 계산 (안전하게)
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

# 고민 점수 계산 (음수 or NaN 방지)
concern_scores = {
    '모공': (pore_avg - 500) / 500 if pore_avg >= 500 else 0,
    '주름': (50 - elasticity_avg) / 50 if elasticity_avg <= 50 else 0,
    '수분': (55 - moisture_avg) / 55 if moisture_avg <= 55 else 0,
    '색소침착': (pigment_avg - 130) / 130 if pigment_avg >= 130 else 0,
}
user_concerns = [k for k, v in sorted(concern_scores.items(), key=lambda x: x[1], reverse=True) if v > 0]

print(f"✔️ 사용자 고민 (우선순위): {user_concerns}")

# 고민 키워드 사전
concern_keywords = {
    '모공': ['모공', '피지','노폐물','피부결','각질'],
    '주름': ['주름', '탄력','영양공급','피부활력','피부재생','나이트','아이'],
    '수분': ['수분', '보습'],
    '색소침착': ['미백', '브라이트닝', '비타민', '피부톤', '투명','트러블케어','피부재생','피부보호','스팟','저자극','진정']
}

# 제품 스코어링 함수
def score_product(row):
    tags = str(row['태그'])
    score = 0
    weights = [3, 2, 1]
    for idx, concern in enumerate(user_concerns):
        weight = weights[idx] if idx < len(weights) else 1
        for keyword in concern_keywords.get(concern, []):
            if keyword in tags:
                score += weight
    return score

# 추천 제품 추출
products['score'] = products.apply(score_product, axis=1)
recommended = products[products['score'] > 0].sort_values(by='score', ascending=False).head(5)

# 출력 및 JSON 반환 예시
result = {
    "평균값": {
        "수분 평균": round(moisture_avg, 2),
        "탄력 평균": round(elasticity_avg, 2),
        "모공 평균": round(pore_avg, 2),
        "색소침착 평균": round(pigment_avg, 2)
    },
    "고민 우선순위": user_concerns,
    "추천 제품": recommended[['브랜드', '제품명', '용량/가격', '별점']].to_dict(orient='records')
}

# JSON 직렬화 (API 응답 예시로 활용 가능)
response_json = json.dumps(result, ensure_ascii=False)
print("\n📄 추천 결과 JSON:")
print(response_json)
