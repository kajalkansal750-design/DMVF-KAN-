"""
Synthetic Benchmark Generator for Multi-Label Chest X-Ray Datasets
Generates structured datasets matching CheXpert, PadChest, JSRT, and COVID distributions
to allow out-of-the-box execution and testing of the entire DMVF-KAN++ pipeline.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from configs.config import CHEST_XRAY_14_LABELS


def generate_synthetic_cxr_image(
    filename: str,
    has_cardiomegaly: bool = False,
    has_pneumonia: bool = False,
    has_effusion: bool = False,
    size: Tuple[int, int] = (224, 224)
) -> str:
    """
    Renders a realistic synthetic Chest X-Ray image with thoracic ribcage,
    cardiac silhouette, and pathology opacities.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img = Image.new("L", size, color=20)
    draw = ImageDraw.Draw(img)
    w, h = size
    
    # Draw chest cavity boundary
    draw.ellipse([int(w * 0.15), int(h * 0.1), int(w * 0.85), int(h * 0.9)], fill=60)
    
    # Left and Right Lung Fields (darker translucent air)
    draw.ellipse([int(w * 0.2), int(h * 0.2), int(w * 0.45), int(h * 0.8)], fill=30)
    draw.ellipse([int(w * 0.55), int(h * 0.2), int(w * 0.8), int(h * 0.8)], fill=30)
    
    # Mediastinum and Spine (denser white center)
    draw.rectangle([int(w * 0.45), int(h * 0.1), int(w * 0.55), int(h * 0.9)], fill=150)
    
    # Cardiac silhouette (heart shadow in lower left chest)
    heart_scale = 1.35 if has_cardiomegaly else 1.0
    heart_box = [
        int(w * 0.42),
        int(h * 0.45),
        int(w * (0.42 + 0.25 * heart_scale)),
        int(h * (0.45 + 0.35 * heart_scale))
    ]
    draw.ellipse(heart_box, fill=160)
    
    # Pathology artifacts
    if has_pneumonia:
        # Patchy consolidation / opacities in lung field
        draw.ellipse([int(w * 0.25), int(h * 0.35), int(w * 0.4), int(h * 0.55)], fill=130)
        
    if has_effusion:
        # Blunted costophrenic angle / fluid level at lower lung bases
        draw.polygon(
            [(int(w * 0.18), int(h * 0.7)), (int(w * 0.45), int(h * 0.82)), (int(w * 0.18), int(h * 0.82))],
            fill=140
        )
        
    # Ribs simulation (subtle horizontal arcs)
    for r in range(4):
        y_pos = int(h * (0.28 + r * 0.12))
        draw.arc([int(w * 0.2), y_pos - 10, int(w * 0.8), y_pos + 15], start=0, end=180, fill=85, width=3)
        
    # Add subtle Gaussian blur to simulate radiographic scattering
    img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    img.save(filename)
    return filename


def create_synthetic_benchmark_suite(data_root: str = "data_samples", num_samples: int = 150):
    """
    Creates complete mock datasets for CheXpert, PadChest, JSRT, and COVID
    following the exact patient-level splitting and metadata schemes of Table 1a.
    """
    os.makedirs(data_root, exist_ok=True)
    
    # 1. CheXpert train / val / test
    chexpert_dir = os.path.join(data_root, "chexpert")
    os.makedirs(os.path.join(chexpert_dir, "images"), exist_ok=True)
    
    records = []
    for i in range(num_samples):
        patient_id = f"patient_{i // 3:04d}"
        img_name = f"chexpert_{i:04d}.jpg"
        img_path = os.path.join(chexpert_dir, "images", img_name)
        
        sex = "Male" if i % 2 == 0 else "Female"
        age = 20 + (i * 7) % 75
        view = "PA" if (i % 3 != 0) else "AP"
        
        # Correlated labels
        cardiomegaly = 1 if (i % 4 == 0) else 0
        pneumonia = 1 if (i % 5 == 0) else 0
        effusion = 1 if (cardiomegaly and (i % 2 == 0)) or (i % 6 == 0) else 0
        
        generate_synthetic_cxr_image(
            img_path,
            has_cardiomegaly=bool(cardiomegaly),
            has_pneumonia=bool(pneumonia),
            has_effusion=bool(effusion)
        )
        
        row = {
            "image_path": img_path,
            "patient_id": patient_id,
            "sex": sex,
            "age": age,
            "view": view,
            "scanner": "GE_Optima" if i % 2 == 0 else "Siemens_YSIO",
            "institution": "stanford_chexpert",
            "Atelectasis": 1 if (i % 7 == 0) else 0,
            "Cardiomegaly": cardiomegaly,
            "Consolidation": 1 if (pneumonia and i % 2 == 0) else 0,
            "Edema": 1 if (cardiomegaly and effusion) else 0,
            "Pleural Effusion": effusion,
            "Emphysema": 1 if (i % 11 == 0) else 0,
            "Fibrosis": 1 if (i % 13 == 0) else 0,
            "Hernia": 1 if (i % 23 == 0) else 0,
            "Infiltration": 1 if (pneumonia) else 0,
            "Mass": 1 if (i % 17 == 0) else 0,
            "Nodule": 1 if (i % 9 == 0) else 0,
            "Pleural Thickening": 1 if (effusion and i % 3 == 0) else 0,
            "Pneumonia": pneumonia,
            "Pneumothorax": 1 if (i % 19 == 0) else 0
        }
        records.append(row)
        
    df_chexpert = pd.DataFrame(records)
    
    # Patient-level split 80:10:10 (Section 4.1)
    unique_patients = df_chexpert["patient_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    n_p = len(unique_patients)
    n_train = int(n_p * 0.8)
    n_val = int(n_p * 0.1)
    
    train_pts = set(unique_patients[:n_train])
    val_pts = set(unique_patients[n_train:n_train + n_val])
    test_pts = set(unique_patients[n_train + n_val:])
    
    df_chexpert[df_chexpert["patient_id"].isin(train_pts)].to_csv(os.path.join(chexpert_dir, "train.csv"), index=False)
    df_chexpert[df_chexpert["patient_id"].isin(val_pts)].to_csv(os.path.join(chexpert_dir, "val.csv"), index=False)
    df_chexpert[df_chexpert["patient_id"].isin(test_pts)].to_csv(os.path.join(chexpert_dir, "test.csv"), index=False)
    
    # 2. PadChest benchmark test split
    padchest_dir = os.path.join(data_root, "padchest")
    os.makedirs(os.path.join(padchest_dir, "images"), exist_ok=True)
    pad_records = []
    for i in range(50):
        img_name = f"padchest_{i:04d}.jpg"
        img_path = os.path.join(padchest_dir, "images", img_name)
        generate_synthetic_cxr_image(img_path, has_cardiomegaly=(i % 3 == 0), has_pneumonia=(i % 4 == 0))
        p_row = {
            "image_path": img_path,
            "patient_id": f"pad_patient_{i}",
            "sex": "Female" if i % 2 == 0 else "Male",
            "age": 30 + (i * 5) % 65,
            "view": "PA",
            "scanner": "Philips_DigitalDiagnost",
            "institution": "san_juan_padchest"
        }
        for lbl in CHEST_XRAY_14_LABELS:
            p_row[lbl] = 1 if (hash(lbl + str(i)) % 5 == 0) else 0
        pad_records.append(p_row)
    pd.DataFrame(pad_records).to_csv(os.path.join(padchest_dir, "test.csv"), index=False)
    
    # 3. JSRT Unseen benchmark
    jsrt_dir = os.path.join(data_root, "jsrt")
    os.makedirs(os.path.join(jsrt_dir, "images"), exist_ok=True)
    jsrt_records = []
    for i in range(30):
        img_name = f"jsrt_{i:04d}.jpg"
        img_path = os.path.join(jsrt_dir, "images", img_name)
        generate_synthetic_cxr_image(img_path)
        has_nodule = 1 if i % 2 == 0 else 0
        j_row = {
            "image_path": img_path,
            "patient_id": f"jsrt_{i}",
            "sex": "Male" if i % 2 == 0 else "Female",
            "age": 55,
            "view": "PA",
            "scanner": "Konica_Minolta",
            "institution": "jsrt_japan_us"
        }
        for lbl in CHEST_XRAY_14_LABELS:
            j_row[lbl] = has_nodule if lbl == "Nodule" else 0
        jsrt_records.append(j_row)
    pd.DataFrame(jsrt_records).to_csv(os.path.join(jsrt_dir, "test.csv"), index=False)
    
    print(f"Synthetic benchmark suite created successfully at: {data_root}")
    return chexpert_dir, padchest_dir, jsrt_dir
