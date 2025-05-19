# ✅ analyze_image_skin_type.py (새 CV 모델 기반)

import os, torch, numpy as np
from PIL import Image, ImageOps
import pandas as pd
from torchvision import transforms, models
import torch.nn as nn
import cv2
import mediapipe as mp


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

skin_label_names = ["건성", "복합건성", "중성", "복합지성", "지성"]

# Dummy skin type model (for structure)
skin_model = models.resnet18(weights=None)
skin_model.fc = nn.Linear(skin_model.fc.in_features, len(skin_label_names))
skin_model.eval()


def get_restore_stats():
    return {
        1: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9)},
        5: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9), "모공 개수": "log"},
        6: {"수분": (60.2, 9.6), "탄력": (49.3, 12.1), "모공 개수": "log"},
        8: {"수분": (61.3, 10.0), "탄력": (47.5, 12.0)},
        0: {"색소침착 개수": 300}
    }

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

def compute_z_score(value, mean, std):
    if std == 0: return 0
    return round((value - mean) / std, 2)

# ✅ 모델 불러오기
reg_models = []
restore_stats = get_restore_stats()
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

# ✅ 메인 함수: 이미지 + 성별_연령대 → 결과 리턴
def model_image(image: Image.Image, gender_age: str, average_data_path="average_data.csv") -> dict:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    regions = crop_regions_by_ratio(image)
    result = {"regions": {}, "z_scores": {}, "skin_type": None, "priority_concern": None}

    df = pd.read_csv(average_data_path, encoding='cp949')
    avg_data = {}
    for _, row in df.iterrows():
        key = row["성별"]
        avg_data[key] = {
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
                "색소침착 개수_스팟개수": 130
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
                "색소침착 개수_스팟개수": row["색소침착 개수_스팟개수_표준편차"]
            }
        }

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
            z_key = f"{label}_{area_label[idx]}" if label != "색소침착 개수" else "스팟개수"
            if z_key in mean_dict:
                z = compute_z_score(val, mean_dict[z_key], std_dict[z_key])
                sub_zscore[label] = z
                z_score_list.append((label, area_label[idx], z))
        result["regions"][area_label[idx]] = sub_result
        if sub_zscore:
            result["z_scores"][area_label[idx]] = sub_zscore

    if regions[0] is not None:
        overall_tensor = transform(regions[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(skin_model(overall_tensor), dim=1).item()
            result["skin_type"] = skin_label_names[pred]

    concerns = []
    for label, area, z in z_score_list:
        if label in ["수분", "탄력"] and z < -0.2:
            concerns.append((label, area, z))
        elif label in ["모공 개수", "색소침착 개수"] and z > 0.2:
            concerns.append((label, area, z))
    if concerns:
        result["priority_concern"] = sorted(concerns, key=lambda x: abs(x[2]), reverse=True)[0]

    return result
