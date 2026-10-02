# -*- coding: utf-8 -*-
"""
Benchmark: Full Handwriting Analysis across all image categories.
Tests: form fields (numbers/dates), full handwriting pages (text), dysgraphia samples.
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath('.'))

from src.field_reader.handwriting_analyzer import get_handwriting_analyzer

analyzer = get_handwriting_analyzer()

DYSGRAPHIA_BASE = r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection"
HACKATHON_BASE  = os.path.abspath('.')

# ----------------------------------------------------------------
# Image groups to test
# ----------------------------------------------------------------
test_groups = {
    "Potential Dysgraphia (full-page text)": [
        os.path.join(DYSGRAPHIA_BASE, "DATASET DYSGRAPHIA HANDWRITING", "Potential Dysgraphia", f"PD ({i}).jpg")
        for i in [1, 5, 10, 20, 30]
    ],
    "Low Potential Dysgraphia (full-page text)": [
        os.path.join(DYSGRAPHIA_BASE, "DATASET DYSGRAPHIA HANDWRITING", "Low Potential Dysgraphia", f"LPD ({i}).jpg")
        for i in [1, 5, 10]
    ],
    "Scraped candidates (general handwriting)": [
        os.path.join(DYSGRAPHIA_BASE, "scraped_candidates", "images", "ENG_CAND_058.jpg"),
        os.path.join(DYSGRAPHIA_BASE, "scraped_candidates", "images", "ENG_CAND_001.jpg"),
    ],
    "Form fields - Dates (structured numbers)": [
        os.path.join(HACKATHON_BASE, "data", "form_fields", f"field_{i:04d}_date.png")
        for i in range(1, 6)
    ],
    "Form fields - PIN codes (digits only)": [
        os.path.join(HACKATHON_BASE, "data", "form_fields", f"field_{i:04d}_pin.png")
        for i in range(7, 12)
    ],
}

results_by_group = {}
grand_latencies = []

for group_name, paths in test_groups.items():
    print(f"\n{'='*65}")
    print(f"GROUP: {group_name}")
    print(f"{'='*65}")
    group_results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {os.path.basename(path)}")
            continue
        try:
            res = analyzer.analyze(path, field_type="General")
            risk = res.get("dysgraphia_risk", {})
            group_results.append({
                "file": os.path.basename(path),
                "text": res["text"][:60],
                "mean_conf": res["mean_conf_pct"],
                "tokens": res["num_tokens"],
                "risk": risk.get("risk_level", "N/A"),
                "risk_score": risk.get("risk_score", 0),
                "mode": res.get("analysis_mode", "?"),
                "latency_ms": res.get("total_latency_ms", 0),
            })
            grand_latencies.append(res.get("total_latency_ms", 0))
            print(f"  {os.path.basename(path)[:35]:35s} | text='{res['text'][:25]}' | conf={res['mean_conf_pct']:5.1f}% | tokens={res['num_tokens']:3d} | risk={risk.get('risk_level','N/A'):8s} ({risk.get('risk_score',0):.3f}) | {res.get('total_latency_ms',0):.0f}ms")
        except Exception as e:
            print(f"  [ERR] {os.path.basename(path)}: {e}")
    results_by_group[group_name] = group_results

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print(f"\n{'='*65}")
print("BENCHMARK SUMMARY")
print(f"{'='*65}")
total_images = sum(len(v) for v in results_by_group.values())
print(f"Total images analyzed : {total_images}")
if grand_latencies:
    print(f"Avg latency           : {sum(grand_latencies)/len(grand_latencies):.0f} ms")
    print(f"Min latency           : {min(grand_latencies):.0f} ms")
    print(f"Max latency           : {max(grand_latencies):.0f} ms")

for group, items in results_by_group.items():
    if not items:
        continue
    avg_conf = sum(r["mean_conf"] for r in items) / len(items)
    risk_dist = {}
    for r in items:
        risk_dist[r["risk"]] = risk_dist.get(r["risk"], 0) + 1
    print(f"\n  [{group}]")
    print(f"    Tested: {len(items)} | Avg OCR conf: {avg_conf:.1f}% | Risk dist: {risk_dist}")
