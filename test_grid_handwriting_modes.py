import urllib.request
import json
import base64
import os

def test_image(fpath, ftype='Auto', is_comb=False):
    with open(fpath, 'rb') as f:
        b64 = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
    payload = json.dumps({'image_base64': b64, 'field_type': ftype, 'is_comb_box': is_comb}).encode()
    req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode())
        print(f"Test: {os.path.basename(fpath)} (Requested Type: {ftype})")
        print(f"  Result Text: '{d.get('text')}' (Raw: '{d.get('raw_ocr_text')}')")
        print(f"  Confidence : {d.get('confidence')} ({d.get('conf_pct')}%)")
        print(f"  Has Grid   : {d.get('has_grid')} | Cells: {d.get('detected_cells')}")
        print(f"  Layout Mode: {d.get('layout_mode')} | Effective Type: {d.get('field_type')}")
        print(f"  Latency    : {d.get('pipeline_stages', {}).get('total_ms')} ms\n")

if __name__ == '__main__':
    print("=================================================================")
    print("TESTING FORM FIELDS WITH GRIDS, FREEFORM, AND HANDWRITING")
    print("=================================================================")

    # Test 1: Comb-Box Grid Date (Ground Truth: '11/05/2022')
    print("--- TEST 1: Comb-Box Grid Date with Auto-Detect ---")
    test_image('data/form_fields/field_0003_date.png', 'Auto', False)

    # Test 2: Comb-Box Grid Alphanumeric Code (Ground Truth: 'ELQ-7177')
    print("--- TEST 2: Comb-Box Grid Code with Auto-Detect ---")
    test_image('data/form_fields/field_0005_code.png', 'Auto', False)

    # Test 3: Comb-Box Grid PIN (Ground Truth: '209517')
    print("--- TEST 3: Comb-Box Grid PIN with Auto-Detect ---")
    test_image('data/form_fields/field_0009_pin.png', 'Auto', False)

    # Test 4: Freeform Date (Ground Truth: '02/06/1984')
    print("--- TEST 4: Freeform Date with Date Format ---")
    test_image('data/form_fields/field_0001_date.png', 'Date', False)

    # Test 5: Normal Handwriting Mode (Freeform Handwriting)
    print("--- TEST 5: Normal Handwriting Mode ---")
    test_image('data/form_fields/field_0001_date.png', 'Handwriting', False)
