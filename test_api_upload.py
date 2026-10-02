import urllib.request
import json
import base64

with open('data/form_fields/field_0007_pin.png', 'rb') as f:
    img_b64 = 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')

payload = json.dumps({
    'image_base64': img_b64,
    'field_type': 'PIN',
    'mode': 'trocr'
}).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/predict',
    data=payload,
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print("\n--- UPLOADED HANDWRITING INFERENCE ---")
    print(f"Text         : {res['text']}")
    print(f"Clean text   : {res['clean_text']}")
    print(f"Architecture : {res['architecture']}")
    print(f"Mean Conf    : {res['mean_conf_pct']}%")
    print(f"Min Conf     : {res['min_conf_pct']}%")
    print(f"Status       : {res['status']}")
    print(f"Syntax valid : {res['syntax_valid']}")
    print(f"Latency      : {res['latency_ms']} ms ({res['device']})")
    print(f"Total Tokens : {len(res['tokens'])}")
    for t in res['tokens']:
        print(f"  Token '{t['char']}': bbox={t['bbox']}, conf={t['conf_pct']}%, peak={t['spatial_peak']}")
