from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from analyze_image_skin_type import run_analysis
import json

app = FastAPI()

@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = run_analysis(image_bytes)

    # ✅ NaN 직렬화 방지 처리
    try:
        safe_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
        return JSONResponse(content=json.loads(safe_json))  # 다시 dict로 변환
    except Exception as e:
        print("🚨 JSON 직렬화 실패:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
