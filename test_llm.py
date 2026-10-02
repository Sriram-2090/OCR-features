import time
import requests
import json

t0 = time.time()
prompt = """You are an OCR post-correction system.
The OCR engine transcribed: "021061/3/3.4"
Expected format: DD/MM/YYYY
Return strictly a JSON object: {"corrected_text": "<date>", "reasoning": "<explanation>"}
"""

payload = {
    "model": "qwen2.5:7b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "num_predict": 60
    }
}

try:
    resp = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=25.0)
    print("Status:", resp.status_code)
    print("Elapsed:", round(time.time() - t0, 2), "s")
    raw = resp.json().get("response", "")
    print("Raw output:\n", raw)
except Exception as e:
    print("Error:", e)
