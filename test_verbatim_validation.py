import urllib.request
import json
import base64
import os

def test_payload(img_path, ftype='Auto'):
    ext = os.path.splitext(img_path)[1].lower()
    mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    with open(img_path, 'rb') as f:
        b64 = f'data:{mime};base64,' + base64.b64encode(f.read()).decode()
    req_data = json.dumps({'image_base64': b64, 'field_type': ftype}).encode()
    req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=req_data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"{os.path.basename(img_path):28s} [{ftype:11s}] -> Text: '{res.get('text')}' | Raw: '{res.get('raw_ocr_text')}' | Layout: {res.get('layout_mode')}")

if __name__ == '__main__':
    print("=== TESTING USER CROPPED IMAGE (245326) ===")
    for ft in ['Auto', 'Date', 'Pin', 'Handwriting']:
        test_payload('exact_user_field_245326.png', ft)

    print("\n=== TESTING BENCHMARK FIELDS ===")
    test_payload('data/form_fields/field_0003_date.png', 'Auto')
    test_payload('data/form_fields/field_0005_code.png', 'Auto')
    test_payload('data/form_fields/field_0009_pin.png', 'Auto')
    test_payload('data/form_fields/field_0001_date.png', 'Date')

    print("\n=== TESTING DYSGRAPHIA HANDWRITING ===")
    dys_p = r'C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING\Low Potential Dysgraphia\LPD (1).jpg'
    if os.path.exists(dys_p):
        test_payload(dys_p, 'Auto')
