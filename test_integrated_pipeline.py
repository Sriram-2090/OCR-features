"""
Integrated Pipeline Test
========================
Uses HandwritingAnalyzer to run TrOCR + BHK dysgraphia analysis on handwriting images.
"""

import os
import sys

# IMPORTANT: Add Hackathon repo root FIRST so src.field_reader resolves correctly
HACKATHON_ROOT = os.path.abspath(os.path.dirname(__file__))
if HACKATHON_ROOT not in sys.path:
    sys.path.insert(0, HACKATHON_ROOT)

from src.field_reader.handwriting_analyzer import get_handwriting_analyzer  # noqa: E402

analyzer = get_handwriting_analyzer()


def analyze_image(img_path: str):
    """Run full analysis on a handwriting image."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(img_path)}")
    print(f"{'='*60}")

    result = analyzer.analyze(img_path, field_type="General")

    print(f"  Transcribed Text : {result['text']}")
    print(f"  OCR Mean Conf    : {result['mean_conf_pct']}%")
    print(f"  OCR Min Conf     : {result['min_conf_pct']}%")
    print(f"  Tokens           : {result['num_tokens']}")
    print(f"  Analysis Mode    : {result['analysis_mode']}")
    print(f"  Total Latency    : {result['total_latency_ms']} ms")

    risk = result.get("dysgraphia_risk", {})
    print(f"\n  Dysgraphia Risk  : {risk.get('risk_level', 'N/A')} (score={risk.get('risk_score', 0.0):.3f})")
    for flag in risk.get("flags", []):
        print(f"    [!] {flag}")

    bhk_summary = risk.get("bhk_summary", {})
    if bhk_summary:
        print(f"\n  BHK Summary:")
        for k, v in bhk_summary.items():
            print(f"    {k}: {v}")

    return result


# Test images — order: dysgraphia samples first, then benchmark form fields
test_files = [
    r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\scraped_candidates\images\ENG_CAND_058.jpg",
    r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING\Potential Dysgraphia\PD (1).jpg",
    "data/form_fields/field_0007_pin.png",
    "data/form_fields/field_0001_date.png",
]

for tf in test_files:
    full_path = tf if os.path.isabs(tf) else os.path.join(HACKATHON_ROOT, tf)
    if os.path.exists(full_path):
        analyze_image(full_path)
    else:
        print(f"[SKIP] Not found: {tf}")

