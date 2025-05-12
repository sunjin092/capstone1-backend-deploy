import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image, ImageOps
import torch.nn as nn
import cv2
import mediapipe as mp
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 상대경로로 수정
regression_ckpt = os.path.join("checkpoint", "regression")
regression_num_output = [1, 2, 0, 0, 0, 3, 3, 0, 2]

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 결과 복원 기준
restore_stats = {
    1: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9)},
    5: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9), "모공 개수": "log"},
    6: {"수분": (60.2, 9.6), "탄력": (49.3, 12.1), "모공 개수": "log"},
    8: {"수분": (61.3, 10.0), "탄력": (47.5, 12.0)},
    0: {"색소침착 개수": 300}
}
area_label = {
    0: "전체", 1: "이마", 2: "미간", 3: "왼쪽 눈가", 4: "오른쪽 눈가",
    5: "왼쪽 볼", 6: "오른쪽 볼", 7: "입술", 8: "턱"
}
reg_desc = {
    0: ["색소침착 개수"],
    1: ["수분", "탄력"],
    5: ["수분", "탄력", "모공 개수"],
    6: ["수분", "탄력", "모공 개수"],
    8: ["수분", "탄력"]
}

# Mediapipe 얼굴 영역 crop
mp_face_mesh = mp.solutions.face_mesh
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

def crop_regions_by_ratio(pil_img, visualize=False):
    img = np.array(pil_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    regions = [None] * 9
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
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

# 회귀 모델만 불러오기
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

# ✅ 분석 함수

def run_analysis(image_bytes):
    result = {}
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    try:
        regions = crop_regions_by_ratio(image, visualize=False)
    except Exception as e:
        return {"error": f"얼굴 인식 실패: {str(e)}"}

    region_results = {}
    for idx in range(9):
        if reg_models[idx] is None or regions[idx] is None or idx in [3, 4]:
            continue
        crop_tensor = transform(regions[idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            reg_out = reg_models[idx](crop_tensor).squeeze().cpu().numpy()
        if reg_out.ndim == 0:
            reg_out = [reg_out]
        area_name = area_label[idx]
        values = {}
        for i, val in enumerate(reg_out):
            label = reg_desc[idx][i]
            if label == "모공 개수" and restore_stats[idx].get(label) == "log":
                val = np.clip(np.exp(val) - 1, 0, 2500)
            elif label in restore_stats[idx] and isinstance(restore_stats[idx][label], tuple):
                mean, std = restore_stats[idx][label]
                val = val * std + mean
            elif label == "색소침착 개수":
                val *= 300
            values[label] = round(float(val), 2)
        region_results[area_name] = values

    result["regions"] = region_results
    return result
