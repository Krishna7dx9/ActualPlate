import os, csv, requests, numpy as np, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api.actualplate.tech/upload-image"
ROOT = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
CSV_PATH = ROOT / "labels.csv"
OUT_CSV = "diagnostic_3d_results.csv"

gt = {}
with open(CSV_PATH, "r") as f:
    for row in csv.DictReader(f):
        gt[row['image']] = {
            "label": row['label'].lower(),
            "volume": float(row['volume']),
            "weight": float(row['weight'])
        }

images = list(ROOT.rglob("*.jpg"))
print(f"🚀 Running 3D Math Diagnostic Benchmark on {len(images)} images...")

with open(OUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "True_Label", "Pred_Label", "True_Vol", "Pred_Vol", 
                     "Pred_Thick_cm", "Pred_Diam_cm", "Mask_Pixels", "BBox_Width_Px"])

    for i, img_path in enumerate(images, 1):
        if img_path.name not in gt: continue
        
        true_data = gt[img_path.name]
        print(f"[{i}/{len(images)}] {img_path.name}...", end="", flush=True)
        
        try:
            with open(img_path, "rb") as f:
                r = requests.post(API_URL, files={"file": (img_path.name, f, "image/jpeg")}, timeout=60).json()
                
            foods = r.get("foods") or []
            geom = r.get("geometry") or []
            vlm = r.get("vlm_output") or []
            
            if not foods or not geom or not vlm:
                print(" | FAIL (No detection)")
                writer.writerow([img_path.name, true_data['label'], "none", true_data['volume'], 0, 0, 0, 0, 0])
                continue
                
            pred_label = (foods[0].get("food_name") or "").lower()
            pred_vol = geom[0].get("volume_cm3", 0.0)
            
            # Extract diagnostic data
            pred_thick = vlm[0].get("estimated_thickness_cm", 0.0)
            pred_diam = vlm[0].get("estimated_diameter_cm", 0.0)
            mask_px = geom[0].get("mask_pixels", 0)
            
            bbox = geom[0].get("bbox", [0,0,0,0])
            bbox_w = int(bbox[2] - bbox[0]) if bbox else 0
            
            print(f" | Vol: {pred_vol:.1f} | Thick: {pred_thick:.1f} | Mask: {mask_px}")
            writer.writerow([img_path.name, true_data['label'], pred_label, true_data['volume'], pred_vol, 
                             pred_thick, pred_diam, mask_px, bbox_w])
            
        except Exception as e:
            print(f" | ERR: {str(e)[:20]}")
            writer.writerow([img_path.name, true_data['label'], "error", true_data['volume'], 0, 0, 0, 0, 0])
            time.sleep(2)

print("\n" + "="*50)
print("DIAGNOSTIC BENCHMARK COMPLETE")
print("="*50)
true_v, pred_v = [], []
with open(OUT_CSV, "r") as f:
    for row in csv.DictReader(f):
        if float(row["Pred_Vol"]) > 0 and float(row["True_Vol"]) > 0:
            true_v.append(float(row["True_Vol"]))
            pred_v.append(float(row["Pred_Vol"]))

if true_v:
    mape = np.mean(np.abs(np.array(pred_v) - np.array(true_v)) / np.array(true_v)) * 100
    print(f"Valid Images: {len(true_v)}")
    print(f"New 3D Volume MAPE: {mape:.2f}%")
    print(f"Results saved to: {OUT_CSV}")