"""
Comprehensive Evaluation & Comparative Benchmark for Handwritten Form Field Reader.
Compares:
  Mode 1: Raw Character CNN Baseline (Track B Mandatory Deliverable)
  Mode 2: Grammar & Dictionary FSM Decoder
  Mode 3: SOTA Tri-Engine Pipeline (CNN + Clean Morphology + FSM + Semantic Lattice)
"""

from __future__ import annotations

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import cv2

repo_root = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"
sys.path.insert(0, repo_root)

from src.field_reader.pipeline import FormReaderPipeline


def run_comparative_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running Comparative Benchmark on {device}...")

    ckpt_path = os.path.join(repo_root, "models", "field_cnn.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Model not found at {ckpt_path}")
        return

    pipeline = FormReaderPipeline(model_path=ckpt_path, device=device)

    # Load 150 benchmark test fields
    csv_path = os.path.join(repo_root, "data", "form_fields", "metadata.csv")
    test_df = pd.read_csv(csv_path)
    print(f"Loaded {len(test_df)} benchmark test fields from: {csv_path}\n")

    modes = ["raw_cnn", "grammar_fsm", "tri_engine"]
    mode_names = {
        "raw_cnn": "Baseline: Raw Character CNN",
        "grammar_fsm": "Tier 1: CNN + Grammar/FSM Decoder",
        "tri_engine": "Tier 2: SOTA Tri-Engine Pipeline"
    }

    mode_results = {m: {"exact": 0, "chars_correct": 0, "total_chars": 0, "fields": []} for m in modes}
    confusion_pairs = {}

    for idx, row in test_df.iterrows():
        img_path = row["image_path"]
        gt = str(row["ground_truth"]).strip()
        f_type = row["field_type"]
        is_comb = bool(row["is_comb_box"])
        n_chars = int(row["num_chars"])

        img = cv2.imread(img_path)
        if img is None:
            continue

        for m in modes:
            res = pipeline.process(
                img,
                field_type=f_type,
                is_comb_box=is_comb,
                expected_cells=n_chars,
                mode=m,
                conf_threshold=0.85
            )

            pred_text = res["text"]
            is_exact = (pred_text == gt)
            if is_exact:
                mode_results[m]["exact"] += 1

            # Char-level accuracy
            for i in range(min(len(pred_text), len(gt))):
                mode_results[m]["total_chars"] += 1
                if pred_text[i] == gt[i]:
                    mode_results[m]["chars_correct"] += 1
                elif m == "raw_cnn":
                    pair = f"'{gt[i]}' -> '{pred_text[i]}'"
                    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

            mode_results[m]["fields"].append({
                "field_id": int(row["field_id"]),
                "type": f_type,
                "gt": gt,
                "pred": pred_text,
                "min_conf": res["min_conf"],
                "exact": is_exact
            })

    # Summary Display
    total_fields = len(test_df)
    print("="*75)
    print("COMPARATIVE BENCHMARK LEADERBOARD (150 TEST FIELDS, 1,157 GLYPHS)")
    print("="*75)
    print(f"{'Architecture / Model Mode':<36} | {'Char Acc':^10} | {'Field Exact Match':^18}")
    print("-" * 75)

    summary_table = []
    for m in modes:
        c_acc = (mode_results[m]["chars_correct"] / max(mode_results[m]["total_chars"], 1)) * 100.0
        f_acc = (mode_results[m]["exact"] / total_fields) * 100.0
        print(f"{mode_names[m]:<36} | {c_acc:8.2f}% | {f_acc:8.2f}% ({mode_results[m]['exact']}/{total_fields})")
        summary_table.append({
            "mode": m,
            "name": mode_names[m],
            "char_accuracy": round(c_acc, 2),
            "complete_field_accuracy": round(f_acc, 2),
            "exact_matches": mode_results[m]["exact"],
            "total_fields": total_fields,
        })
    print("="*75)

    # Confidence Gating Trade-Off Table for SOTA Tri-Engine
    print("\n" + "="*75)
    print("SOTA TRI-ENGINE CONFIDENCE TRIGGER & AUTOMATED INGESTION TABLE")
    print("="*75)
    print(f"{'Threshold θ':^12} | {'Auto-Accept Rate':^18} | {'Auto-Accepted Acc':^19} | {'Human Review Rate':^18}")
    print("-" * 75)

    sota_fields = mode_results["tri_engine"]["fields"]
    trigger_analysis = []
    for theta in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        auto_accepted = [f for f in sota_fields if f["min_conf"] >= theta]
        flagged = [f for f in sota_fields if f["min_conf"] < theta]

        auto_rate = (len(auto_accepted) / total_fields) * 100.0
        flag_rate = (len(flagged) / total_fields) * 100.0
        auto_acc = (sum(1 for f in auto_accepted if f["exact"]) / max(len(auto_accepted), 1)) * 100.0

        print(f"  θ = {theta:.2f}     |      {auto_rate:5.1f}%      |        {auto_acc:5.1f}%         |      {flag_rate:5.1f}%")
        trigger_analysis.append({
            "threshold": theta,
            "auto_accept_rate": round(auto_rate, 1),
            "auto_accepted_accuracy": round(auto_acc, 1),
            "human_review_rate": round(flag_rate, 1),
        })

    # Save full report
    report = {
        "summary": summary_table,
        "tri_engine_triggers": trigger_analysis,
        "top_confusions": sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    report_path = os.path.join(repo_root, "models", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Full comparative benchmark report saved to: {report_path}")


if __name__ == "__main__":
    run_comparative_benchmark()
