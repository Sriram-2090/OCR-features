# -*- coding: utf-8 -*-
"""
Full Pipeline Test: TrOCR replacing CRNN in Dysgraphia Analysis
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from src.field_reader.handwriting_analyzer import HandwritingAnalyzer

analyzer = HandwritingAnalyzer()

# Test images from both repos
DYS_ROOT = r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection"
HACK_ROOT = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"

tests = [
    ("Potential Dysgraphia",
     os.path.join(DYS_ROOT, "DATASET DYSGRAPHIA HANDWRITING", "Potential Dysgraphia", "PD (1).jpg")),
    ("Low Potential Dysgraphia",
     os.path.join(DYS_ROOT, "DATASET DYSGRAPHIA HANDWRITING", "Low Potential Dysgraphia", "LPD (1).jpg")),
    ("Form Field (Date)",
     os.path.join(HACK_ROOT, "data", "form_fields", "field_0001_date.png")),
]

for label, img_path in tests:
    if not os.path.exists(img_path):
        print(f"SKIP {label}: file not found at {img_path}")
        continue

    t0 = time.time()
    result = analyzer.analyze(img_path)
    elapsed = (time.time() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"  Category: {label}")
    print(f"  File: {os.path.basename(img_path)}")
    print(f"{'='*60}")

    # TrOCR Aligner (primary)
    print(f"  TrOCR Aligner Text: '{result['text']}'")
    print(f"  TrOCR Aligner Conf: {result['mean_conf_pct']}%")

    # TrOCR OCR Engine (replaces CRNN)
    crnn = result.get("iam_crnn", {})
    if crnn.get("available"):
        print(f"  TrOCR-Engine Text:  '{crnn['text']}'")
        print(f"  TrOCR-Engine Conf:  {crnn['mean_conf_pct']}%")
        print(f"  Lines: {crnn.get('total_lines', 0)}, Words: {crnn.get('total_words', 0)}")
        sigs = crnn.get("ocr_dysgraphia_signals", {})
        if sigs:
            print("  OCR Dysgraphia Signals:")
            for k, v in sigs.items():
                print(f"    {k}: {v}")
    else:
        print("  TrOCR-Engine: NOT AVAILABLE")

    # BHK Features
    bhk = result.get("bhk_features", {})
    if bhk:
        print(f"  BHK Features ({len(bhk)} metrics extracted)")
        for k in ["baseline_drift_slope", "letter_size_cv", "inter_component_gap_cv",
                   "stroke_tremor_high_freq", "letter_collision_ratio", "slant_angle_std"]:
            if k in bhk:
                print(f"    {k}: {bhk[k]}")

    # Dysgraphia Risk
    risk = result.get("dysgraphia_risk", {})
    print(f"  Risk Level: {risk.get('risk_level')}")
    print(f"  Probability: {risk.get('dysgraphia_probability')}%")
    print(f"  Prediction: {risk.get('prediction_label')}")
    flags = risk.get("flags", [])
    if flags:
        print("  Flags:")
        for f in flags[:6]:
            print(f"    - {f}")

    print(f"  Total Latency: {elapsed:.0f}ms")
    print(f"  Models: {result.get('models_loaded')}")

print("\n\nDone!")
