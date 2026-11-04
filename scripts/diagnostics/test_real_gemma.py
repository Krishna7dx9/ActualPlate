import os, sys, base64, json, requests, re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get('OPENROUTER_API_KEY')

ROOT = Path.home() / 'Downloads' / 'simple_food_45' / 'ordered_dataset'
images = list(ROOT.rglob('*.jpg'))[:5]

SYSTEM_PROMPT = """You are an expert food scientist analyzing a photo.
Return ONLY a JSON array of objects. Do not include markdown.
Extract exactly three fields:
1. "search_query": simplest name
2. "density_g_cm3": density
3. "estimated_height_cm": best visual guess of physical height in cm (e.g. 6.5)
"""

for i, img_path in enumerate(images, 1):
    print(f'\n[{i}/5] Testing REAL image: {img_path.name}')
    image_b64 = base64.b64encode(img_path.read_bytes()).decode('utf-8')

    payload = {
        'model': 'google/gemini-1.5-flash',  # <--- CORRECTED MODEL NAME
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Analyze this image.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}}
            ]}
        ],
        'temperature': 0.1
    }

    try:
        r = requests.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': f'Bearer {API_KEY}'}, json=payload)
        res_json = r.json()
        
        if 'choices' not in res_json:
            print("API Error:", res_json)
            continue
            
        raw = res_json['choices'][0]['message']['content']
        raw = re.sub(r'```json|```', '', raw).strip()
        
        parsed = json.loads(raw)
        print('Output:', parsed)
        
        if 'estimated_height_cm' in parsed[0]:
            print(f'? SUCCESS: Height output = {parsed[0]["estimated_height_cm"]} cm')
        else:
            print('? FAILED: Gemma dropped the height field.')
            
    except Exception as e:
        print('Error:', e)
