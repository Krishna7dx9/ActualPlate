import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load your local .env
load_dotenv()
url = os.environ.get("VISION_INFERENCE_ENDPOINT")

# Use the real image you provided
img_path = Path("C:/Users/Krish Sharma/OneDrive/Desktop/images.jpg")

print(f"🚀 Sending REAL image to live pipeline: {url}")
print(f"   File: {img_path.name}")

if not img_path.exists():
    print("❌ ERROR: Image not found at that path!")
    exit(1)

with open(img_path, "rb") as f:
    response = requests.post(
        url,
        files={"file": (img_path.name, f, "image/jpeg")},
        timeout=300
    )

print(f"\nHTTP Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    
    print("\n--- 1. GEMMA'S VLM OUTPUT (The Scale) ---")
    vlm = data.get("vlm_output", [])
    if not vlm:
        print("No foods detected by Gemma.")
    for item in vlm:
        print(f"Label: {item.get('search_query')}")
        print(f"  Diameter: {item.get('estimated_diameter_cm')} cm")
        print(f"  Thickness: {item.get('estimated_thickness_cm')} cm")

    print("\n--- 2. GEOMETRY & VOLUME (The Math) ---")
    geom = data.get("geometry", [])
    if not geom:
        print("DINO did not find any bounding boxes.")
        
    for item in geom:
        print(f"Label: {item.get('label')}")
        print(f"  Volume: {item.get('volume_cm3')} cm3")
        print(f"  Status: {item.get('volume_status')}")
        print(f"  Scale Applied: {item.get('scale_factor_applied')}")
        
    print("\n--- 3. NUTRITION ---")
    for food in data.get("foods", []):
        print(f"Food: {food.get('food_name')} | Mass: {food.get('mass_g')}g | Calories: {food.get('calories')}")
else:
    print("\n❌ ERROR:", response.text)