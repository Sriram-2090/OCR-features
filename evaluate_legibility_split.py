"""
Legibility-Stratified Accuracy & Economic Cost-Optimal Threshold Benchmark.
Form Field Reader (Track B Rubric Deliverable).

Partitions collected samples into:
  1. Group A: Clearly Legible Handwriting (Comb-box constrained & isolated glyphs)
  2. Group B: Genuinely Difficult Handwriting (Touching cursive, ligatures, stroke collisions)

Evaluates:
  - Character-Level Accuracy (Char Acc % / CER %)
  - Whole-Field Accuracy (Exact Match %)
  - Confidence Distributions
  - Economic Cost Model: C_error ($25.00) vs C_review ($0.04) -> 625:1 Cost Ratio
  - Cost Curve & Mathematical Justification for Optimal Confidence Threshold (theta*)
"""

from __future__ import annotations

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import pandas as pd
import numpy as np
import torch

repo_root = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"
sys.path.insert(0, repo_root)

from src.field_reader.pipeline import FormReaderPipeline

DATASET_CSV = os.path.join(repo_root, "data", "form_fields", "metadata.csv")
MODEL_CKPT = os.path.join(repo_root, "models", "field_cnn.pth")
REPORT_OUT = os.path.join(repo_root, "models", "legibility_split_report.json")

# Operational Cost Model Constants
C_HUMAN_REVIEW = 0.04    # $0.04 per field (~6 sec operator review at $24/hr)
C_WRONG_CONFIDENT = 25.00 # $25.00 per undetected silent error (banking/KYC/re-shipping penalty)


def run_legibility_split_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("   OC&HCR: LEGIBILITY-STRATIFIED ACCURACY & COST-OPTIMAL BENCHMARK")
    print("=" * 80)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Dataset Source : {DATASET_CSV}")
    print(f"[*] Model Checkpoint: {MODEL_CKPT}")
    print(f"[*] Economic Matrix: C_review = ${C_HUMAN_REVIEW:.2f} | C_error = ${C_WRONG_CONFIDENT:.2f} (Ratio {C_WRONG_CONFIDENT/C_HUMAN_REVIEW:.0f}:1)\n")

    if not os.path.exists(MODEL_CKPT):
        print(f"Error: Model weights not found at {MODEL_CKPT}")
        return

    pipeline = FormReaderPipeline(model_path=MODEL_CKPT, device=device)
    df = pd.read_csv(DATASET_CSV)

    records = []
    print(f"[*] Processing {len(df)} collected form field samples across Tri-Engine Pipeline...")

    for idx, row in df.iterrows():
        p = row["image_path"]
        img = cv2.imread(p)
        if img is None:
            continue

        gt = str(row["ground_truth"]).strip()
        gt_len = len(gt)
        f_type = row["field_type"]
        is_comb = bool(row["is_comb_box"])
        n_chars = int(row["num_chars"])

        # Morphological Analysis for Legibility Classification
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        valid_boxes = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > 15 and s[cv2.CC_STAT_HEIGHT] > 8]
        touching_count = max(0, gt_len - len(valid_boxes))

        # Stratification Criteria:
        # Group A (Clearly Legible): Clean stroke boundaries with zero character collisions
        # Group B (Genuinely Difficult): Touching strokes, cursive ligatures, or stroke collisions
        if touching_count == 0:
            primary_group = "Group A: Clearly Legible Handwriting"
            is_difficult = False
            sub_category = "Comb-Box Clean" if is_comb else "Freeform Isolated"
        else:
            primary_group = "Group B: Genuinely Difficult Handwriting"
            is_difficult = True
            sub_category = "Comb-Box Border Noise" if is_comb else "Freeform Touching Cursive"

        # Inference through Tri-Engine
        res = pipeline.process(
            img,
            field_type=f_type,
            is_comb_box=is_comb,
            expected_cells=n_chars,
            mode="tri_engine",
            conf_threshold=0.85
        )

        pred = res["text"]
        exact = (pred == gt)

        # Character-level matching
        match_c = sum(1 for c1, c2 in zip(gt, pred) if c1 == c2)
        tot_c = max(len(gt), len(pred))
        char_acc = match_c / tot_c
        cer = 1.0 - char_acc

        # Extract calibrated minimum character confidence
        glyphs = res.get("glyphs", [])
        if glyphs:
            min_char_conf = min(g["conf"] for g in glyphs)
            mean_char_conf = float(np.mean([g["conf"] for g in glyphs]))
        else:
            min_char_conf = res["min_conf"]
            mean_char_conf = res["mean_conf"]

        records.append({
            "field_id": int(row["field_id"]),
            "field_type": f_type,
            "is_comb_box": is_comb,
            "ground_truth": gt,
            "predicted_text": pred,
            "exact_match": exact,
            "char_acc": char_acc,
            "cer": cer,
            "num_chars": tot_c,
            "chars_correct": match_c,
            "min_conf": min_char_conf,
            "mean_conf": mean_char_conf,
            "primary_group": primary_group,
            "is_difficult": is_difficult,
            "sub_category": sub_category,
            "touching_glyphs": touching_count
        })

    res_df = pd.DataFrame(records)

    # 1. Performance Summary by Group
    print("\n" + "=" * 80)
    print("1. EMPIRICAL ACCURACY RESULTS: CLEARLY LEGIBLE VS GENUINELY DIFFICULT")
    print("=" * 80)
    print(f"{'Metric':<34} | {'Group A: Clearly Legible':^20} | {'Group B: Difficult':^18}")
    print("-" * 80)

    legible_df = res_df[~res_df["is_difficult"]]
    diff_df = res_df[res_df["is_difficult"]]

    def calc_metrics(sub):
        n = len(sub)
        exact_cnt = sub["exact_match"].sum()
        exact_pct = (exact_cnt / n) * 100.0 if n > 0 else 0.0
        tot_chars = sub["num_chars"].sum()
        corr_chars = sub["chars_correct"].sum()
        char_acc_pct = (corr_chars / tot_chars) * 100.0 if tot_chars > 0 else 0.0
        cer_pct = 100.0 - char_acc_pct
        mean_c = sub["mean_conf"].mean() * 100.0
        min_c = sub["min_conf"].mean() * 100.0
        return {
            "n": n,
            "exact_cnt": int(exact_cnt),
            "exact_pct": exact_pct,
            "char_acc_pct": char_acc_pct,
            "cer_pct": cer_pct,
            "mean_conf": mean_c,
            "min_conf": min_c
        }

    m_a = calc_metrics(legible_df)
    m_b = calc_metrics(diff_df)

    print(f"{'Sample Count (N)':<34} | {m_a['n']:^20} | {m_b['n']:^18}")
    print(f"{'Whole-Field Exact Match Rate':<34} | {m_a['exact_pct']:>6.2f}% ({m_a['exact_cnt']}/{m_a['n']})    | {m_b['exact_pct']:>6.2f}% ({m_b['exact_cnt']}/{m_b['n']})")
    print(f"{'Character-Level Accuracy':<34} | {m_a['char_acc_pct']:>6.2f}%              | {m_b['char_acc_pct']:>6.2f}%")
    print(f"{'Character Error Rate (CER)':<34} | {m_a['cer_pct']:>6.2f}%              | {m_b['cer_pct']:>6.2f}%")
    print(f"{'Mean Character Confidence':<34} | {m_a['mean_conf']:>6.2f}%              | {m_b['mean_conf']:>6.2f}%")
    print(f"{'Average Minimum Confidence':<34} | {m_a['min_conf']:>6.2f}%              | {m_b['min_conf']:>6.2f}%")
    print("=" * 80)

    # 2. Granular Sub-Category Breakdown
    print("\n" + "=" * 80)
    print("2. GRANULAR SUB-CATEGORY PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(f"{'Sub-Category':<28} | {'N':^5} | {'Field Exact':^14} | {'Char Acc':^10} | {'CER':^8} | {'Min Conf':^10}")
    print("-" * 80)
    sub_report = []
    for cat, sub in res_df.groupby("sub_category"):
        m = calc_metrics(sub)
        print(f"{cat:<28} | {m['n']:^5} | {m['exact_pct']:>6.2f}% ({m['exact_cnt']}/{m['n']}) | {m['char_acc_pct']:>6.2f}%  | {m['cer_pct']:>5.2f}% | {m['min_conf']:>6.2f}%")
        sub_report.append({"category": cat, **m})
    print("=" * 80)

    # 3. Confidence Threshold Sweep & Economic Cost Analysis
    print("\n" + "=" * 80)
    print("3. ECONOMIC COST CURVE & CONFIDENCE THRESHOLD (θ) OPTIMIZATION")
    print(f"   Formula: Total Cost = N_flagged * ${C_HUMAN_REVIEW:.2f} + N_silent_error * ${C_WRONG_CONFIDENT:.2f}")
    print("=" * 80)
    print(f"{'Threshold θ':^11} | {'Review Rate':^13} | {'Auto-Accept':^13} | {'Silent Errors':^14} | {'Cost/Field':^11} | {'Total Cost ($)':^13}")
    print("-" * 80)

    cost_sweep = []
    best_theta = 0.85
    min_cost = float("inf")
    total_samples = len(res_df)

    thresholds = [round(t, 2) for t in np.arange(0.60, 0.99, 0.02)]
    for theta in thresholds:
        flagged = res_df["min_conf"] < theta
        n_flagged = int(flagged.sum())
        flag_rate = (n_flagged / total_samples) * 100.0

        auto_accepted = res_df[~flagged]
        n_auto = len(auto_accepted)
        auto_rate = (n_auto / total_samples) * 100.0

        # Silent errors = Auto-accepted but not exact match
        silent_errs = auto_accepted[~auto_accepted["exact_match"]]
        n_silent = len(silent_errs)
        silent_rate = (n_silent / total_samples) * 100.0

        # Economic calculation
        cost_review = n_flagged * C_HUMAN_REVIEW
        cost_error = n_silent * C_WRONG_CONFIDENT
        total_cost = cost_review + cost_error
        cost_per_field = total_cost / total_samples

        if total_cost < min_cost:
            min_cost = total_cost
            best_theta = theta

        cost_sweep.append({
            "threshold": theta,
            "review_rate_pct": round(flag_rate, 2),
            "auto_accept_pct": round(auto_rate, 2),
            "silent_error_count": n_silent,
            "silent_error_pct": round(silent_rate, 2),
            "cost_per_field": round(cost_per_field, 4),
            "total_cost": round(total_cost, 2),
            "cost_review": round(cost_review, 2),
            "cost_error": round(cost_error, 2)
        })

    # Print representative checkpoints
    display_thetas = [0.60, 0.70, 0.75, 0.80, 0.84, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
    for row in cost_sweep:
        if row["threshold"] in display_thetas or row["threshold"] == best_theta:
            star = " <--- OPTIMAL" if row["threshold"] == best_theta else ""
            print(f"  θ = {row['threshold']:.2f}    |  {row['review_rate_pct']:>5.1f}% ({int(row['review_rate_pct']*total_samples/100):2d}) |  {row['auto_accept_pct']:>5.1f}% ({int(row['auto_accept_pct']*total_samples/100):3d}) |  {row['silent_error_pct']:>4.1f}% ({row['silent_error_count']:2d})   |  ${row['cost_per_field']:>6.4f}   |  ${row['total_cost']:>7.2f}{star}")

    print("=" * 80)
    print(f"\n[★] MATHEMATICALLY OPTIMAL OPERATING THRESHOLD: θ* = {best_theta:.2f}")
    print(f"    - Minimizes Expected Operational Loss to ${min_cost/total_samples:.4f} / field (Total ${min_cost:.2f} / {total_samples} fields)")
    print(f"    - Cost Ratio Asymmetry: ${C_WRONG_CONFIDENT:.2f} / ${C_HUMAN_REVIEW:.2f} = {C_WRONG_CONFIDENT/C_HUMAN_REVIEW:.0f}:1")
    print(f"    - Human Review Routing Rate at θ* : {cost_sweep[[r['threshold'] for r in cost_sweep].index(best_theta)]['review_rate_pct']:.1f}%")
    print(f"    - Silent Error Rate at θ*          : {cost_sweep[[r['threshold'] for r in cost_sweep].index(best_theta)]['silent_error_pct']:.1f}%")

    # 4. Multi-Regime Economic Sensitivity Analysis
    print("\n" + "=" * 80)
    print("4. COST SENSITIVITY ANALYSIS ACROSS OPERATIONAL REGIMES")
    print("=" * 80)
    print(f"{'Operational Domain':<26} | {'C_error':^9} | {'C_review':^10} | {'Ratio':^8} | {'Optimal θ*':^11} | {'Review Rate':^12}")
    print("-" * 80)

    regimes = [
        {"domain": "Low-Risk Archival", "c_err": 5.00, "c_rev": 0.04},
        {"domain": "Standard Logistics/Postal", "c_err": 15.00, "c_rev": 0.04},
        {"domain": "Govt/Tax Administration", "c_err": 25.00, "c_rev": 0.04},
        {"domain": "High-Assurance Banking/KYC", "c_err": 50.00, "c_rev": 0.04},
        {"domain": "Critical Legal/Defense", "c_err": 100.00, "c_rev": 0.04},
    ]

    sensitivity_results = []
    for reg in regimes:
        ce = reg["c_err"]
        cr = reg["c_rev"]
        ratio = ce / cr
        best_t = 0.85
        lowest_c = float("inf")

        for row in cost_sweep:
            t = row["threshold"]
            flagged = res_df["min_conf"] < t
            n_flg = int(flagged.sum())
            n_slt = int((res_df[~flagged]["exact_match"] == False).sum())
            c_tot = (n_flg * cr) + (n_slt * ce)
            if c_tot < lowest_c:
                lowest_c = c_tot
                best_t = t

        flg_rate = (res_df["min_conf"] < best_t).mean() * 100.0
        print(f"{reg['domain']:<26} | ${ce:>7.2f} | ${cr:>8.2f} | {ratio:>6.0f}:1 |    θ*={best_t:.2f}   |   {flg_rate:>5.1f}%")
        sensitivity_results.append({**reg, "ratio": ratio, "optimal_theta": best_t, "review_rate": flg_rate})

    print("=" * 80)

    # 5. Save structured report
    report_data = {
        "metadata": {
            "title": "Legibility-Stratified Accuracy & Economic Cost-Optimal Threshold Report",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_samples": total_samples,
            "cost_parameters": {
                "c_review": C_HUMAN_REVIEW,
                "c_error": C_WRONG_CONFIDENT,
                "cost_ratio": C_WRONG_CONFIDENT / C_HUMAN_REVIEW
            },
            "optimal_threshold": best_theta,
            "min_cost_per_field": round(min_cost / total_samples, 4)
        },
        "group_a_clearly_legible": m_a,
        "group_b_genuinely_difficult": m_b,
        "sub_categories": sub_report,
        "cost_curve": cost_sweep,
        "sensitivity_analysis": sensitivity_results
    }

    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[✓] Structured JSON benchmark report successfully written to:\n    {REPORT_OUT}")


if __name__ == "__main__":
    run_legibility_split_benchmark()
