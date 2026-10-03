import urllib.request
import json
import base64
import os

test_files = [
    ('data/form_fields/field_0001_date.png', '02/06/1984'),
    ('data/form_fields/field_0002_date.png', '18/09/1987'),
    ('data/form_fields/field_0002_pin.png', '131437'),
    ('data/form_fields/field_0003_date.png', '11/05/2022'),
    ('data/form_fields/field_0004_date.png', '18-08-1972'),
    ('data/form_fields/field_0005_code.png', 'ELQ-7177'),
    ('data/form_fields/field_0007_pin.png', '131437'),
]

for fpath, gt in test_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'rb') as f:
        b64 = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
    payload = json.dumps({'image_base64': b64, 'field_type': 'Auto'}).encode()
    req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode())
        pred = d.get('text')
        match = 'MATCH' if pred == gt else 'FAIL'
        print(f"{os.path.basename(fpath):20} [GT: {gt:10}] -> Pred: '{pred:12}' ({match}) | Grid: {d.get('has_grid')} | Mode: {d.get('layout_mode')}")
