import os, sys, json
from pathlib import Path

sys.path.append(os.getcwd())
from vision_service.recognition.food_recognition import detect_food

ROOT = Path.home() / 'Downloads' / 'simple_food_45' / 'ordered_dataset'
images = list(ROOT.rglob('*.jpg'))[:5]

print(f"Testing {len(images)} REAL images using YOUR UPDATED CODE...\n")

for i, img_path in enumerate(images, 1):
    print(f"[{i}/5] Image: {img_path.name}")
    image_bytes = img_path.read_bytes()
    
    try:
        result = detect_food(image_bytes)
        print("Output:", json.dumps(result, indent=2))
        
        if result and "estimated_height_cm" in result[0]:
            print(f"? SUCCESS: Height output = {result[0]['estimated_height_cm']} cm\n")
        else:
            print(f"? FAILED: Height still missing.\n")
    except Exception as e:
        print(f"? ERROR: {str(e)}\n")
