import os
import csv
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ["VISION_INFERENCE_ENDPOINT"]

# Paths
DATASET_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
LABELS_FILE = DATASET_DIR / "labels.csv"

# Use the existing benchmarks/results folder
RESULTS_FILE = Path("benchmarks/results/baseline_volume_results.csv")

def evaluate_full_dataset():
    print(f"Loading ground truth from: {LABELS_FILE}")
    
    # Ensure results directory exists
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results to allow resuming if it crashes
    processed_images = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_images.add(row["image"])
        print(f"Resuming. {len(processed_images)} images already processed.")
    else:
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "true_volume", "pred_volume", "abs_error_pct"])

    errors = []
    total_processed = 0
    
    with open(LABELS_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            image_name = row["image"]
            
            if image_name in processed_images:
                continue
                
            true_volume = float(row["volume"])
            label = row["label"]
            image_path = DATASET_DIR / image_name
            
            if not image_path.exists():
                continue
            
            print(f"[{idx+1}] Testing {image_name} ({label})...", end="", flush=True)
            
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
                
                if not geometry:
                    print(" | FAIL (No geometry)")
                    continue
                
                pred_volume = geometry[0].get("volume_cm3")
                
                if pred_volume is None:
                    print(f" | FAIL (Unreliable)")
                    continue
                    
                error = abs(pred_volume - true_volume) / true_volume
                errors.append(error)
                total_processed += 1
                
                # Append to CSV immediately
                with open(RESULTS_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([image_name, label, true_volume, pred_volume, f"{error*100:.2f}"])
                
                print(f" | Pred: {pred_volume:.1f} cm³ | Err: {error*100:.1f}%")
                
            except Exception as e:
                print(f" | ERROR: {str(e)[:50]}")

    if errors:
        mape = (sum(errors) / len(errors)) * 100
        print("\n" + "="*50)
        print(f"FINAL BASELINE MAPE (N={len(errors)}): {mape:.2f}%")
        print(f"Results saved to: {RESULTS_FILE}")
        print("="*50)
    else:
        print("No valid predictions collected.")

if __name__ == "__main__":
    evaluate_full_dataset()