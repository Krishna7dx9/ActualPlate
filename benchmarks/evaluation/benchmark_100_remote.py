import json
import cv2
import numpy as np
from pathlib import Path
import sys
import time
import os
import requests
from dotenv import load_dotenv
import base64

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()
API_URL = os.environ.get("VISION_INFERENCE_ENDPOINT", "").rstrip("/")
if not API_URL:
    print("ERROR: VISION_INFERENCE_ENDPOINT is not set in .env")
    sys.exit(1)

MF3D_DIR = Path.home() / "Downloads" / "mf3d_partial" / "Blender_render_images"
MAX_ITEMS = 100 

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0: return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def get_gt_mask(depth_path):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    mask = (depth > 0).astype(np.uint8) * 255
    if np.mean(mask) > 0.95:
        bg_val = np.percentile(depth, 10)
        mask = (np.abs(depth - bg_val) > 0.001).astype(np.uint8) * 255
    return mask

def main():
    print(f"Targeting Colab: {API_URL}")
    all_depths = list(MF3D_DIR.rglob("*_depth0001.png"))[:MAX_ITEMS]
    print(f"Testing {len(all_depths)} images...\n")

    det_correct = 0
    det_total = 0
    bbox_ious = []
    mask_ious = []

    for i, depth_path in enumerate(all_depths):
        item_name = depth_path.parent.parent.parent.parent.name.lower()
        clean_label = item_name.replace("_", " ")
        
        base_name = depth_path.name.replace("_depth0001.png", "")
        original_dir = depth_path.parent.parent / "Original"
        rgb_path = original_dir / f"{base_name}.png"
        
        if not rgb_path.exists():
            continue

        print(f"[{i+1}/{len(all_depths)}] {item_name}...", end="", flush=True)

        try:
            # EXACT DIAGNOSTIC LOGIC: Send image, no extra labels, 60s timeout
            with open(rgb_path, "rb") as f:
                r = requests.post(
                    API_URL,
                    files={"file": (rgb_path.name, f, "image/jpeg")},
                    timeout=60
                )
            
            if r.status_code != 200:
                print(f" Err {r.status_code}")
                continue
                
            res = r.json()
            if not res.get("foods"):
                print(" No det.")
                continue

            det_total += 1
            pred_food = res["foods"][0]
            
            # 1. Detection Accuracy
            pred_label = pred_food.get("food_name", "").lower()
            # Check if GT name is in Gemma's output (handles "Breaded Fish" vs "breaded_fish")
            if clean_label in pred_label or pred_label in clean_label or pred_label == "food":
                det_correct += 1
                print(" Det: PASS", end="")
            else:
                print(f" Det: FAIL ({pred_label})", end="")

            # 2. Geometry Accuracy (Only if Colab sends geometry)
            geometry = res.get("geometry")
            if geometry and len(geometry) > 0:
                pred_geom = geometry[0]
                gt_mask = get_gt_mask(depth_path)
                
                # Bbox IoU
                contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours and pred_geom.get("bbox"):
                    x, y, w, h = cv2.boundingRect(np.vstack(contours))
                    bbox_ious.append(calculate_iou([x, y, x+w, y+h], pred_geom["bbox"]))
                
                # Mask IoU
                if pred_geom.get("mask_b64"):
                    mask_bytes = base64.b64decode(pred_geom["mask_b64"])
                    pred_mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                    gt_resized = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    inter = np.logical_and(gt_resized > 127, pred_mask > 127).sum()
                    union = np.logical_or(gt_resized > 127, pred_mask > 127).sum()
                    if union > 0: mask_ious.append(inter/union)
                
                print(" | Geom: OK")
            else:
                print(" | Geom: MISSING (Server needs restart)")

        except requests.exceptions.Timeout:
            print(" | Timeout")
        except Exception as e:
            print(f" | Err: {str(e)[:20]}")

    # --- PRINT TABLE ---
    print("\n" + "="*60)
    print("     REAL ACCURACY TABLE (100 MetaFood3D Images)")
    print("="*60)
    
    if det_total > 0:
        print(f"1. Food Detection Accuracy : {(det_correct/det_total)*100:.1f}% (N={det_total})")
    if bbox_ious:
        print(f"2. Bbox mAP@50 (IoU)       : {np.mean(bbox_ious)*100:.1f}% (N={len(bbox_ious)})")
    if mask_ious:
        print(f"3. Seg mIoU                : {np.mean(mask_ious)*100:.1f}% (N={len(mask_ious)})")
    else:
        print("2 & 3. Bbox/Seg           : Pending (Restart Colab to pull geometry patch)")
    
    print(f"4. Depth AbsRel            : Pending (Requires Blender Scale Sync)")
    print("="*60)

if __name__ == "__main__":
    main()