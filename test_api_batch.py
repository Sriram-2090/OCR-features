import urllib.request
import json
import base64
import os

test_samples = [
    ('data/form_fields/field_0007_pin.png', 'PIN', '131437'),
    ('data/form_fields/field_0004_date.png', 'Date', '18-08-1972'),
    ('data/form_fields/field_0002_date.png', 'Date', '18/09/1987'),
    ('data/form_fields/field_0006_code.png', 'Code', 'ZL-9234'),
]

print("=================================================================")
print("BENCHMARKING TrOCR TOKEN-TO-INK SPATIAL ALIGNMENT VIA LIVE API")
print("=================================================================")

for fpath, ftype, gt in test_samples:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'rb') as f:
        img_b64 = 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')

    payload = json.dumps({
        'image_base64': img_b64,
        'field_type': ftype,
        'mode': 'trocr'
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/predict',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"\nImage: {os.path.basename(fpath)} (Type: {ftype}, Ground Truth: '{gt}')")
        print(f"  Recognized Text : '{res['text']}'")
        print(f"  Confidence      : Mean {res['mean_conf_pct']}% | Min {res['min_conf_pct']}% -> Status: {res['status']}")
        print(f"  Tokens Aligned  : {len(res['tokens'])} tokens in {res['latency_ms']} ms ({res['device']})")
        token_str = " ".join([f"{t['char']}[{t['conf_pct']}%@{t['bbox'][0]}]" for t in res['tokens'][:8]])
        print(f"  Spatial Layout  : {token_str}")
