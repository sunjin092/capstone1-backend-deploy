import json
import sys
print("현재 파이썬 경로:", sys.executable)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from io import BytesIO
import numpy as np

# 1. CSV 경로 & 제품 불러오기
csv_path = "C:\capstone1\Total_DB.csv"
products = pd.read_csv(csv_path, encoding='cp949')

# 2. CV api 출력 결과 JSON 문자열 예시 (일단은 예시로 돌아가게 함)
raw_json = '{"regions":{"\\uc624\\ub978\\ucabd \\ubcfc":{"\\ubaa8\\uacf5 \\uac1c\\uc218":750.84,"\\uc218\\ubd84":64.9,"\\ud0c4\\ub825":47.93},"\\uc67c\\ucabd \\ubcfc":{"\\ubaa8\\uacf5 \\uac1c\\uc218":778.89,"\\uc218\\ubd84":63.88,"\\ud0c4\\ub825":65.1},"\\uc774\\ub9c8":{"\\uc218\\ubd84":67.01,"\\ud0c4\\ub825":50.42},"\\uc804\\uccb4":{"\\uc0c9\\uc18c\\uce68\\ucc29 \\uac1c\\uc218":155.41},"\\ud131":{"\\uc218\\ubd84":62.94,"\\ud0c4\\ub825":43.34}},"skin_type":"\\uac74\\uc131"}'

# 3. JSON 디코딩
decoded = json.loads(raw_json)
regions = decoded["regions"]
skin_type = decoded["skin_type"]

# 4. 부위별 평균 계산
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

# 5. 사용자 고민 도출 + 정렬 (가장 심각한 고민을 먼저)
concern_scores = {
    '모공': (pore_avg - 500) / 500 if pore_avg >= 500 else 0,
    '주름': (50 - elasticity_avg) / 50 if elasticity_avg <= 50 else 0,
    '수분': (55 - moisture_avg) / 55 if moisture_avg <= 55 else 0,
    '색소침착': (pigment_avg - 130) / 130 if pigment_avg >= 130 else 0,
}
user_concerns = [k for k, v in sorted(concern_scores.items(), key=lambda x: x[1], reverse=True) if v > 0]

print(f"✔️ 사용자 고민 (우선순위): {user_concerns}")
print(f"✔️ 피부 타입: {skin_type}")

# 6. 고민 키워드 사전 정의
concern_keywords = {
    '모공': ['모공', '피지','노폐물','피부결','각질'],
    '주름': ['주름', '탄력','영양공급','피부활력','피부재생','나이트','아이'],
    '수분': ['수분', '보습'],
    '색소침착': ['미백', '브라이트닝', '비타민', '피부톤', '투명','트러블케어','피부재생','피부보호','스팟','저자극','진정']
    }

# 7. 피부타입 별 제외 조건 정의
exclude_keywords = {
    '지성': ['페이스오일', '멀티밤', '보습크림', '나이트크림'],
    '건성': ['워터토너', '브라이트닝'],
    '복합건성': [],
    '복합지성': [],
    '중성': []
}

# 8. 점수 계산 함수 (가중치 적용)
def score_product(row):
    tags = str(row['태그'])
    detail = str(row.get('세부', ''))
    score = 0

    # 제외 조건
    for block_word in exclude_keywords.get(skin_type, []):
        if block_word in detail or block_word in tags:
            return -1

    # 고민별 가중치 부여
    weights = [3, 2, 1]  # 1순위 3점, 2순위 2점, 그 외 1점
    for idx, concern in enumerate(user_concerns):
        weight = weights[idx] if idx < len(weights) else 1
        for keyword in concern_keywords.get(concern, []):
            if keyword in tags:
                score += weight
    return score

# 9. 제품 스코어 계산 및 추천
products['score'] = products.apply(score_product, axis=1)
recommended = products[products['score'] >= 0].sort_values(by='score', ascending=False).head(5)

# 10. 시각화 출력
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('📌 추천 제품 Top 5', fontsize=18)

for idx, (_, row) in enumerate(recommended.iterrows()):
    try:
        response = requests.get(row['이미지'])
        img = mpimg.imread(BytesIO(response.content), format='jpg')
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"{row['제품명'][:12]}\n({row['브랜드']})", fontsize=9)
    except:
        axes[idx].text(0.5, 0.5, 'No Image', ha='center', va='center')
        axes[idx].axis('off')

plt.show()

# 11. 표 출력
print("\n📄 추천 제품 리스트:")
print(recommended[['브랜드', '제품명', '용량/가격', '별점']])
