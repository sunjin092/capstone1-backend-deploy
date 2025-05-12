from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from analyze_image_skin_type import run_analysis

app = FastAPI()

@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = run_analysis(image_bytes)
    return JSONResponse(content=result)
