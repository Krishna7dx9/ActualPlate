import os, requests, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("VISION_INFERENCE_ENDPOINT")
img_path = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset" / "20230927_102352.jpg"

print(f"🚀 Sending ONE image to {API_URL}...")
with open(img_path, "rb") as f:
    response = requests.post(API_URL, files={"file": (img_path.name, f, "image/jpeg")}, timeout=120)

print(f"HTTP Status: {response.status_code}")
if response.status_code != 200:
    print(f"❌ SERVER ERROR: {response.text}")
    exit()

r = response.json()

print("\n--- 1. RAW GEOMETRY FROM BACKEND ---")
geom = r.get("geometry") or []
foods = r.get("foods") or []

if not geom:
    print("❌ CRITICAL: Backend returned no geometry.")
    exit()

g = geom[0]
print(f"DINO Label: {g.get('label')}")
print(f"Volume from backend: {g.get('volume_cm3')}")
print(f"Mask Pixels: {g.get('mask_pixels')}")
print(f"Bbox: {g.get('bbox')}")

has_scale = "scale_factor_applied" in g
print(f"Scale Factor Applied: {'✅ YES' if has_scale else '❌ NO'}")

if not has_scale:
    print("\n--- 2. WHY WAS THE SCALE SKIPPED? ---")
    
    # Check if the VLM output is even reaching the backend
    # (The pipeline merges it into the final JSON, but we might need to check if the pipeline returned an error)
    
    print("Checking if the DINO label matches the VLM search_query...")
    dino_label = g.get('label', '').lower()
    print(f"DINO Label: '{dino_label}'")
    
    # Check the foods array for the VLM output
    if foods:
        vlm_food_name = (foods[0].get('food_name') or '').lower()
        print(f"VLM Food Name (from FatSecret): '{vlm_food_name}'")
        
        # Test the exact matching logic from pipeline.py
        match_found = (vlm_food_name in dino_label) or (dino_label in vlm_food_name)
        print(f"Match found? {'✅ YES' if match_found else '❌ NO'}")
        
        if not match_found:
            print("❌ BUG FOUND: The VLM called it '{vlm_food_name}', but DINO called it '{dino_label}'. The string matching failed!")
    else:
        print("❌ No foods returned by VLM.")
        
    print("\n--- 3. FULL RAW JSON (For manual inspection) ---")
    print(json.dumps(g, indent=2))

else:
    print("\n✅ SUCCESS! The pixel math worked perfectly. You can now run `python test_2_local.py`.")