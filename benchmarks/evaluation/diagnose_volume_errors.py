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
DIAGNOSTIC_FILE = Path("benchmarks/results/diagnostic_xray.jsonl")

def run_diagnostics():
    print(f"Loading ground truth from: {LABELS_FILE}")
    
    # Clear the file to start fresh
    with open(DIAGNOSTIC_FILE, "w") as f:
        pass

    with open(LABELS_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        
        # We will test 3 of each: Apple (passes), Corn (fails), Chicken Nugget (fails)
        target_counts = {"apple": 3, "corn": 3, "chicken_nugget": 3}
        current_counts = {"apple": 0, "corn": 0, "chicken_nugget": 0}
        
        for row in reader:
            label = row["label"]
            if label not in target_counts or current_counts[label] >= target_counts[label]:
                continue
                
            image_name = row["image"]
            true_volume = float(row["volume"])
            image_path = DATASET_DIR / image_name
            
            if not image_path.exists():
                continue
                
            current_counts[label] += 1
            print(f"X-Ray {image_name} ({label})...", end="", flush=True)
            
            try:
                with open(image_path, "rb") as img:
                    response = requests.post(
                        API_URL,
                        files={"file": (image_name, img, "image/jpeg")},
                        timeout=300
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # The API returns the geometry as a list
                    geometry = data.get("geometry", [{}])
                    if not geometry:
                        print(" | No geometry returned")
                        continue
                    geo = geometry[0]
                    
                    vlm_output = data.get("vlm_output", [{}])
                    vlm = vlm_output[0] if vlm_output else {}
                    
                    diagnostic_record = {
                        "image": image_name,
                        "label": label,
                        "true_volume": true_volume,
                        
                        # Image metadata
                        "image_width": data.get("image_width"),
                        "image_height": data.get("image_height"),
                        
                        # Gemma's Guesses
                        "gemma_search_query": vlm.get("search_query"),
                        "gemma_diameter": vlm.get("estimated_diameter_cm"),
                        "gemma_thickness": vlm.get("estimated_thickness_cm"),
                        
                        # DINO/SAM2 Output
                        "bbox": geo.get("bbox"),
                        
                        # MoGe-2 Depth Output
                        "depth_min": geo.get("depth_m", {}).get("min"),
                        "depth_median": geo.get("depth_m", {}).get("median"),
                        "depth_max": geo.get("depth_m", {}).get("max"),
                        
                        # Final Result
                        "pred_volume": geo.get("volume_cm3"),
                        "volume_status": geo.get("volume_status"),
                        "scale_factor_applied": geo.get("scale_factor_applied"),
                        "volume_error": geo.get("volume_error")
                    }
                    
                    with open(DIAGNOSTIC_FILE, "a") as f:
                        f.write(json.dumps(diagnostic_record) + "\n")
                        
                    print(f" | Status: {diagnostic_record['volume_status']}")
                else:
                    print(f" | HTTP Error {response.status_code}")
                    
            except Exception as e:
                print(f" | ERROR: {str(e)[:50]}")

if __name__ == "__main__":
    run_diagnostics()