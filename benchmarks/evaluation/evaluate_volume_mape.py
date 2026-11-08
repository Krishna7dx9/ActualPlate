import os
import csv
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ["VISION_INFERENCE_ENDPOINT"]

DATASET_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
LABELS_FILE = DATASET_DIR / "labels.csv"
RESULTS_FILE = Path("benchmarks/results/full_architecture_diagnostic.csv")

def run_full_diagnostic():
    print(f"Loading ground truth from: {LABELS_FILE}")
    
    # Load existing results to allow resuming if it crashes
    processed_images = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_images.add(row["image"])
        print(f"Resuming. {len(processed_images)} images already processed.")
    else:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # THE ULTIMATE HEADER
            writer.writerow([
                "image", "true_label", "true_volume", 
                "pred_volume", "abs_error_pct", "volume_status",
                "gemma_label", "gemma_diameter_cm", "gemma_thickness_cm", 
                "bbox_width_px", "scale_factor_applied"
            ])

    errors = []
    total_evaluated = 0
    total_skipped = 0
    
    with open(LABELS_FILE, mode='r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            image_name = row["image"]
            
            if image_name in processed_images:
                continue
                
            true_label = row["label"]
            true_volume = float(row["volume"])
            image_path = DATASET_DIR / image_name
            
            if not image_path.exists():
                continue
            
            print(f"[{idx+1}] {image_name} ({true_label})...", end="", flush=True)
            
            try:
                with open(image_path, "rb") as img:
                    response = requests.post(
                        API_URL,
                        files={"file": (image_name, img, "image/jpeg")},
                        timeout=300
                    )
                
                if response.status_code != 200:
                    print(f" | HTTP Error {response.status_code}")
                    continue
                    
                data = response.json()
                geometry = data.get("geometry", [])
                vlm_output = data.get("vlm_output", [])
                
                if not geometry:
                    print(" | FAIL (No geometry)")
                    total_skipped += 1
                    continue
                
                geo = geometry[0]
                vlm = vlm_output[0] if vlm_output else {}
                
                pred_volume = geo.get("volume_cm3")
                volume_status = geo.get("volume_status", "unknown")
                gemma_label = vlm.get("search_query", "N/A")
                gemma_diam = vlm.get("estimated_diameter_cm", "N/A")
                gemma_thick = vlm.get("estimated_thickness_cm", "N/A")
                scale_factor = geo.get("scale_factor_applied", "N/A")
                
                # Bbox width
                bbox = geo.get("bbox", [0,0,0,0])
                bbox_w = round(bbox[2] - bbox[0], 1) if bbox else 0
                
                if pred_volume is None:
                    print(f" | SKIP (Unreliable) | Gemma: {gemma_thick}cm thick")
                    error_str = "N/A"
                    total_skipped += 1
                else:
                    error = abs(pred_volume - true_volume) / true_volume
                    errors.append(error)
                    total_evaluated += 1
                    error_str = f"{error*100:.1f}%"
                    print(f" | Pred: {pred_volume:.0f}cm³ | Err: {error_str} | Gemma: {gemma_thick}cm")
                
                # Append to CSV immediately
                with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        image_name, true_label, true_volume, 
                        pred_volume if pred_volume else "None", 
                        error_str, volume_status,
                        gemma_label, gemma_diam, gemma_thick, 
                        bbox_w, scale_factor
                    ])
                
            except Exception as e:
                print(f" | ERROR: {str(e)[:50]}")
                total_skipped += 1

    print("\n" + "="*60)
    if errors:
        mape = (sum(errors) / len(errors)) * 100
        print(f"FINAL MAPE (N={total_evaluated} evaluated): {mape:.2f}%")
    else:
        print("No valid predictions collected.")
    
    print(f"Coverage: {total_evaluated} successful out of {total_evaluated + total_skipped} attempts")
    print(f"Full diagnostic CSV saved to: {RESULTS_FILE}")
    print("="*60)

if __name__ == "__main__":
    run_full_diagnostic()