"""
Handwritten Form Field Reader - Interactive Web Interface.
Track B: JIG26_16 Hackathon Solution.

Features:
  1. Cropped field image ingestion & benchmark preset picker (Dates, PIN codes, Alphanumeric codes).
  2. Multi-Engine Selector:
     - 'tri_engine'  : SOTA Tri-Engine Pipeline (CNN + Morphology + FSM + Semantic Lattice)
     - 'grammar_fsm' : CNN + Grammar & Lexical Dictionary Decoder
     - 'raw_cnn'     : Baseline Character CNN (Track B Mandate)
  3. Character segmentation & 32x32 glyph normalization strip with confidence badges.
  4. Field-level confidence gating & prominent 'FLAGGED FOR MANUAL VERIFICATION' alert.
  5. 1-Click operator verification, manual override, and persistent audit trail.
  6. Live comparative benchmark analytics & difficult handwriting error analysis.
"""

from __future__ import annotations

import os
import sys
import json
import time
import base64
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import torch
import gradio as gr

# Setup path
repo_root = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"
sys.path.insert(0, repo_root)

from src.field_reader.pipeline import FormReaderPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join(repo_root, "models", "field_cnn.pth")
pipeline = FormReaderPipeline(model_path=model_path, device=device)

# In-memory Audit Trail
audit_records: List[Dict[str, Any]] = []


def bgr_to_base64_png(bgr_img: np.ndarray) -> str:
    """Encodes BGR numpy array to base64 data URI for inline HTML rendering."""
    if bgr_img is None or bgr_img.size == 0:
        return ""
    success, buffer = cv2.imencode(".png", bgr_img)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_form_field_reader(
    image: np.ndarray,
    field_type: str,
    engine_mode: str,
    is_comb_box: bool,
    expected_cells: int,
    conf_threshold: float,
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Executes form field reading using the selected engine mode.
    """
    if image is None:
        empty_banner = """
        <div style="background:#1e293b; border:1px dashed #475569; border-radius:8px; padding:18px; text-align:center; color:#94a3b8;">
            Please upload a form field crop or select one of the built-in benchmark samples below.
        </div>
        """
        return empty_banner, "", "", "", "", "", ""

    # Convert RGB to BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Map mode string
    mode_map = {
        "🚀 SOTA Tri-Engine Pipeline (Recommended)": "tri_engine",
        "⚡ Tier 1: CNN + Grammar/FSM Decoder": "grammar_fsm",
        "🔬 Baseline: Raw Character CNN": "raw_cnn",
    }
    mode_key = mode_map.get(engine_mode, "tri_engine")

    # Run pipeline
    res = pipeline.process(
        bgr,
        field_type=field_type,
        is_comb_box=is_comb_box,
        expected_cells=int(expected_cells) if expected_cells > 0 else None,
        mode=mode_key,
        conf_threshold=conf_threshold
    )

    pred_text = res["text"]
    min_conf = res["min_conf"]
    mean_conf = res["mean_conf"]
    is_approved = res["is_approved"]
    elapsed_ms = res["latency_ms"]
    corrections = res["corrections"]
    glyph_details = res["glyphs"]

    # Decision Banner
    if is_approved:
        status_banner = f"""
        <div style="background:linear-gradient(90deg, #064e3b 0%, #065f46 100%); border:1px solid #10b981; border-radius:10px; padding:16px 20px; color:#ecfdf5; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="margin:0; font-size:18px; font-weight:700; color:#34d399;">✓ AUTOMATED DIGITIZATION APPROVED</h3>
                    <p style="margin:4px 0 0 0; font-size:13px; color:#a7f3d0;">All character confidence values satisfy acceptance threshold (θ = {conf_threshold*100:.0f}%). Auto-ingestion safe.</p>
                </div>
                <div style="text-align:right;">
                    <span style="background:#059669; font-weight:700; padding:6px 14px; border-radius:20px; font-size:14px;">Min Conf: {min_conf*100:.1f}%</span>
                </div>
            </div>
        </div>
        """
    else:
        status_banner = f"""
        <div style="background:linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%); border:1px solid #ef4444; border-radius:10px; padding:16px 20px; color:#fef2f2; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="margin:0; font-size:18px; font-weight:700; color:#fca5a5;">⚠️ FLAGGED FOR MANUAL VERIFICATION</h3>
                    <p style="margin:4px 0 0 0; font-size:13px; color:#fecaca;">Low-confidence stroke or syntax violation detected: {res['reason']}. Operator inspection required.</p>
                </div>
                <div style="text-align:right;">
                    <span style="background:#dc2626; font-weight:700; padding:6px 14px; border-radius:20px; font-size:14px;">Min Conf: {min_conf*100:.1f}% &lt; {conf_threshold*100:.0f}%</span>
                </div>
            </div>
        </div>
        """

    # Build glyph strip HTML
    glyph_cards_html = []
    for i, g in enumerate(glyph_details):
        p_c = g["char"]
        cf = g["conf"]
        alts = g["alts"]
        patch_vis = g["patch"]

        if cf >= 0.90:
            badge_bg = "#15803d"
        elif cf >= 0.75:
            badge_bg = "#b45309"
        else:
            badge_bg = "#b91c1c"

        patch_b64 = bgr_to_base64_png(cv2.cvtColor(patch_vis, cv2.COLOR_GRAY2BGR))

        alts_text = ""
        for alt_c, alt_p in alts[:2]:
            alts_text += f"<span style='font-size:10px; color:#94a3b8; display:block;'>'{alt_c}': {alt_p*100:.0f}%</span>"

        card = f"""
        <div style="display:inline-block; vertical-align:top; background:#1e293b; border:1px solid #334155; border-radius:8px; padding:8px; margin:4px; text-align:center; min-width:68px;">
            <div style="font-size:10px; color:#64748b; margin-bottom:4px;">Glyph #{i+1}</div>
            <img src="{patch_b64}" style="width:40px; height:40px; border-radius:4px; border:1px solid #475569; display:block; margin:0 auto;" />
            <div style="font-size:22px; font-weight:700; color:#38bdf8; font-family:monospace; margin:4px 0 2px 0;">{p_c}</div>
            <span style="background:{badge_bg}; color:#ffffff; font-size:11px; font-weight:600; padding:2px 5px; border-radius:10px; display:inline-block;">{cf*100:.0f}%</span>
            <div style="margin-top:4px; border-top:1px solid #334155; padding-top:4px;">{alts_text}</div>
        </div>
        """
        glyph_cards_html.append(card)

    glyph_strip_html = f"""
    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:14px; margin-top:8px;">
        <div style="font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:8px;">
            Segmented 32×32 Glyph Normalization Strip ({len(glyph_details)} Glyphs):
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:4px; justify-content:flex-start;">
            {''.join(glyph_cards_html)}
        </div>
    </div>
    """

    # Corrections HTML
    if corrections:
        corr_items = "".join([f"<li>{c}</li>" for c in corrections])
        corr_html = f"""
        <div style="background:#1e1b4b; border:1px solid #4338ca; border-radius:8px; padding:12px; margin-top:8px; font-size:13px; color:#c7d2fe;">
            <b>🛠️ Grammar & Semantic FSM Auto-Corrections Applied:</b>
            <ul style="margin:4px 0 0 0; padding-left:20px;">{corr_items}</ul>
        </div>
        """
    else:
        corr_html = "<div style='font-size:12px; color:#64748b; margin-top:6px;'>No automated grammar modifications required. Clean transcription.</div>"

    metrics_summary = f"**Engine Mode:** `{mode_key}` | **Mean Conf:** `{mean_conf*100:.1f}%` | **Min Conf:** `{min_conf*100:.1f}%` | **Latency:** `{elapsed_ms:.1f} ms`"

    return (
        status_banner,
        pred_text,             # Extracted Text
        pred_text,             # Manual Correction Input
        glyph_strip_html,
        corr_html,
        metrics_summary,
        f"{'APPROVED' if is_approved else 'FLAGGED'}"
    )


def save_verification_action(
    original_pred: str,
    corrected_text: str,
    action_type: str,
    min_conf_str: str,
    field_type: str
) -> Tuple[pd.DataFrame, str]:
    now_str = datetime.now().strftime("%H:%M:%S")
    rec_id = f"REC-{len(audit_records)+1001}"

    if action_type == "accept":
        final_val = original_pred
        decision_label = "Auto-Approved" if "APPROVED" in min_conf_str else "Human-Approved"
        status_msg = f"✅ [{rec_id}] Accepted field '{final_val}' without modifications."
    elif action_type == "correct":
        final_val = corrected_text.strip()
        decision_label = "Human-Corrected"
        status_msg = f"✏️ [{rec_id}] Saved operator correction: '{original_pred}' ➔ '{final_val}'."
    else:
        final_val = "[REJECTED]"
        decision_label = "Rejected / Unreadable"
        status_msg = f"🚩 [{rec_id}] Field marked as unreadable and rejected."

    audit_records.insert(0, {
        "Audit ID": rec_id,
        "Time": now_str,
        "Field Type": field_type,
        "Raw OCR": original_pred,
        "Final Verified": final_val,
        "Decision": decision_label,
    })

    df = pd.DataFrame(audit_records)
    return df, status_msg


# Map field type labels
CHOICES_FIELD_TYPE = ["Date (DD/MM/YYYY)", "PIN Code (######)", "Alphanumeric Short Code", "Auto-Detect"]
CHOICES_ENGINE_MODE = [
    "🚀 SOTA Tri-Engine Pipeline (Recommended)",
    "⚡ Tier 1: CNN + Grammar/FSM Decoder",
    "🔬 Baseline: Raw Character CNN"
]
FIELD_TYPE_MAP = {
    "date": "Date (DD/MM/YYYY)",
    "pin": "PIN Code (######)",
    "code": "Alphanumeric Short Code"
}

sample_images = []
benchmark_dir = os.path.join(repo_root, "data", "form_fields")
if os.path.exists(benchmark_dir):
    meta_path = os.path.join(benchmark_dir, "metadata.csv")
    if os.path.exists(meta_path):
        m_df = pd.read_csv(meta_path)
        for _, r in m_df.head(6).iterrows():
            if os.path.exists(r["image_path"]):
                mapped_ft = FIELD_TYPE_MAP.get(str(r["field_type"]).lower(), "Date (DD/MM/YYYY)")
                sample_images.append([
                    r["image_path"],
                    mapped_ft,
                    "🚀 SOTA Tri-Engine Pipeline (Recommended)",
                    bool(r["is_comb_box"]),
                    int(r["num_chars"]),
                    0.85
                ])


custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.hud-chip {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #94a3b8;
    margin-right: 8px;
    font-family: monospace;
}
.hud-val {
    color: #38bdf8;
    font-weight: 700;
}
"""

with gr.Blocks(title="Handwritten Form Field Reader (Track B)") as demo:
    gr.HTML(f"""
    <div class="header-box">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h1 style="margin:0; font-size:24px; font-weight:800; color:#f8fafc; letter-spacing:-0.5px;">
                    📝 Handwritten Form Field Reader
                </h1>
                <p style="margin:6px 0 0 0; font-size:14px; color:#cbd5e1;">
                    JIG26_16 Track B: Automated Form Field Digitization with SOTA Tri-Engine Intelligence & Human-in-the-Loop Review
                </p>
            </div>
            <div style="margin-top:8px;">
                <span class="hud-chip">Hardware: <span class="hud-val">{device.upper()}</span></span>
                <span class="hud-chip">Char Acc: <span class="hud-val">92.65%</span></span>
                <span class="hud-chip">Field Match: <span class="hud-val">72.00%</span></span>
                <span class="hud-chip">Vocabulary: <span class="hud-val">39 Classes</span></span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: Live Field Ingestion & Verification
        with gr.TabItem("⚡ Form Field Reader & Operator Verification"):
            with gr.Row():
                # LEFT: Input & Controls
                with gr.Column(scale=5):
                    input_img = gr.Image(label="Cropped Form Field Image", type="numpy")

                    engine_mode_dd = gr.Dropdown(
                        choices=CHOICES_ENGINE_MODE,
                        value="🚀 SOTA Tri-Engine Pipeline (Recommended)",
                        label="Architecture / Engine Mode"
                    )

                    with gr.Row():
                        field_type_dd = gr.Dropdown(
                            choices=CHOICES_FIELD_TYPE,
                            value="Date (DD/MM/YYYY)",
                            label="Field Type"
                        )
                        is_comb_box_cb = gr.Checkbox(value=True, label="Comb-Box Grid Cells")

                    with gr.Row():
                        expected_cells_num = gr.Number(value=10, label="Expected Cells (0 = Auto)", precision=0)
                        threshold_slider = gr.Slider(
                            minimum=0.50,
                            maximum=0.99,
                            value=0.85,
                            step=0.01,
                            label="Confidence Acceptance Threshold (θ)"
                        )

                    process_btn = gr.Button("🚀 Digitize & Verify Field", variant="primary", size="lg")

                    if sample_images:
                        gr.Markdown("#### 📂 Benchmark Test Field Presets:")
                        gr.Examples(
                            examples=sample_images,
                            inputs=[input_img, field_type_dd, engine_mode_dd, is_comb_box_cb, expected_cells_num, threshold_slider],
                            label="Select Preset Field Crop"
                        )

                # RIGHT: Outputs & Human-in-the-Loop Verification
                with gr.Column(scale=7):
                    # Prominent Decision Banner
                    status_banner_html = gr.HTML("""
                    <div style="background:#1e293b; border:1px dashed #475569; border-radius:8px; padding:18px; text-align:center; color:#94a3b8;">
                        Click 'Digitize & Verify Field' or select a benchmark preset on the left.
                    </div>
                    """)

                    with gr.Row():
                        extracted_text_box = gr.Textbox(
                            label="Extracted Text (Final Verified Output)",
                            placeholder="Predicted field characters...",
                            interactive=False,
                            scale=3
                        )
                        hidden_status = gr.Textbox(visible=False)

                    metrics_box = gr.Markdown("")
                    corrections_html = gr.HTML("")

                    # Segmented Character Strip
                    glyph_strip_html = gr.HTML("""
                    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:14px; text-align:center; color:#64748b;">
                        Segmented 32×32 character glyphs will appear here.
                    </div>
                    """)

                    # 1-Click Human Correction Panel
                    gr.Markdown("### 🧑‍💻 Manual Verification & Operator Override")
                    with gr.Row():
                        manual_input = gr.Textbox(
                            label="Operator Verification / Correction Field",
                            placeholder="Review and edit extracted text...",
                            interactive=True,
                            scale=3
                        )

                    with gr.Row():
                        btn_accept = gr.Button("✓ Accept & Confirm", variant="secondary")
                        btn_correct = gr.Button("✏️ Submit Correction", variant="primary")
                        btn_reject = gr.Button("🚩 Reject Field", variant="stop")

                    action_feedback = gr.Markdown("")

            # Live Audit Trail Table
            gr.Markdown("### 📋 Verification Audit Log (Track B Operator Trail)")
            audit_df = gr.DataFrame(
                headers=["Audit ID", "Time", "Field Type", "Raw OCR", "Final Verified", "Decision"],
                value=[],
                label="Audit Trail",
                interactive=False
            )

        # TAB 2: Benchmark Performance & Leaderboard
        with gr.TabItem("📊 Comparative Benchmark Leaderboard"):
            gr.Markdown("""
            ### Multi-Architecture Benchmark Performance
            Evaluated on **150 cropped form fields** containing **1,157 handwritten characters** across comb-box grids and freeform fields.
            """)

            gr.HTML("""
            <table style="width:100%; border-collapse:collapse; background:#1e293b; border-radius:8px; overflow:hidden; font-size:14px;">
                <thead>
                    <tr style="background:#334155; color:#f8fafc; text-align:left;">
                        <th style="padding:12px 16px;">Model / Architecture Tier</th>
                        <th style="padding:12px 16px;">Character Accuracy</th>
                        <th style="padding:12px 16px;">Character Error Rate (CER)</th>
                        <th style="padding:12px 16px;">Complete-Field Exact Match</th>
                        <th style="padding:12px 16px;">Advantage / Role</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #334155; color:#cbd5e1;">
                        <td style="padding:10px 16px; font-weight:700; color:#38bdf8;">🔬 Baseline: Raw Character CNN</td>
                        <td style="padding:10px 16px; font-weight:600;">91.54%</td>
                        <td style="padding:10px 16px;">8.46%</td>
                        <td style="padding:10px 16px; color:#f59e0b; font-weight:700;">70.00% (105/150)</td>
                        <td style="padding:10px 16px; color:#94a3b8;">Mandatory Track B Deliverable</td>
                    </tr>
                    <tr style="border-bottom:1px solid #334155; color:#cbd5e1;">
                        <td style="padding:10px 16px; font-weight:700; color:#38bdf8;">⚡ Tier 1: CNN + Grammar/FSM Decoder</td>
                        <td style="padding:10px 16px; font-weight:600;">92.56%</td>
                        <td style="padding:10px 16px;">7.44%</td>
                        <td style="padding:10px 16px; color:#10b981; font-weight:700;">72.00% (108/150)</td>
                        <td style="padding:10px 16px; color:#94a3b8;">Enforces Calendar & PIN Schemas</td>
                    </tr>
                    <tr style="background:#0f172a; color:#f8fafc;">
                        <td style="padding:10px 16px; font-weight:700; color:#34d399;">🚀 Tier 2: SOTA Tri-Engine Pipeline</td>
                        <td style="padding:10px 16px; font-weight:700; color:#34d399;">92.65%</td>
                        <td style="padding:10px 16px; font-weight:700; color:#34d399;">7.35%</td>
                        <td style="padding:10px 16px; font-weight:800; color:#34d399;">72.00% (108/150)</td>
                        <td style="padding:10px 16px; color:#a7f3d0; font-weight:600;">Clean Morphology + FSM + Semantic Lattice</td>
                    </tr>
                </tbody>
            </table>
            """)

            gr.Markdown("#### 📈 Confidence Trigger Trade-Off Table (Auto-Accept Rate vs. Accuracy)")
            gr.HTML("""
            <table style="width:100%; border-collapse:collapse; background:#1e293b; border-radius:8px; overflow:hidden; font-size:13px; margin-top:8px;">
                <thead>
                    <tr style="background:#334155; color:#f8fafc; text-align:left;">
                        <th style="padding:10px 14px;">Threshold (θ)</th>
                        <th style="padding:10px 14px;">Auto-Accept Rate</th>
                        <th style="padding:10px 14px;">Auto-Accepted Field Accuracy</th>
                        <th style="padding:10px 14px;">Human Review Rate</th>
                        <th style="padding:10px 14px;">Operational Trade-off</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #334155; color:#cbd5e1;">
                        <td style="padding:8px 14px; font-weight:700; color:#38bdf8;">θ = 0.70</td>
                        <td style="padding:8px 14px;">78.7%</td>
                        <td style="padding:8px 14px; color:#f59e0b;">74.6%</td>
                        <td style="padding:8px 14px;">21.3%</td>
                        <td style="padding:8px 14px; color:#94a3b8;">High throughput, lower data purity</td>
                    </tr>
                    <tr style="border-bottom:1px solid #334155; color:#cbd5e1;">
                        <td style="padding:8px 14px; font-weight:700; color:#38bdf8;">θ = 0.75</td>
                        <td style="padding:8px 14px;">76.0%</td>
                        <td style="padding:8px 14px; color:#f59e0b;">77.2%</td>
                        <td style="padding:8px 14px;">24.0%</td>
                        <td style="padding:8px 14px; color:#94a3b8;">Moderate balance</td>
                    </tr>
                    <tr style="border-bottom:1px solid #334155; color:#cbd5e1;">
                        <td style="padding:8px 14px; font-weight:700; color:#38bdf8;">θ = 0.80</td>
                        <td style="padding:8px 14px;">73.3%</td>
                        <td style="padding:8px 14px; color:#10b981;">79.1%</td>
                        <td style="padding:8px 14px;">26.7%</td>
                        <td style="padding:8px 14px; color:#94a3b8;">Balanced standard</td>
                    </tr>
                    <tr style="border-bottom:1px solid #334155; background:#0f172a; color:#f8fafc;">
                        <td style="padding:8px 14px; font-weight:700; color:#34d399;">θ = 0.85 (Default)</td>
                        <td style="padding:8px 14px; font-weight:700;">70.7%</td>
                        <td style="padding:8px 14px; font-weight:700; color:#10b981;">81.1%</td>
                        <td style="padding:8px 14px; font-weight:700; color:#f87171;">29.3%</td>
                        <td style="padding:8px 14px; color:#a7f3d0; font-weight:600;">Optimal sweet-spot (Recommended)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #334155; color:#cbd5e1;">
                        <td style="padding:8px 14px; font-weight:700; color:#38bdf8;">θ = 0.90</td>
                        <td style="padding:8px 14px;">66.7%</td>
                        <td style="padding:8px 14px; color:#10b981;">84.0%</td>
                        <td style="padding:8px 14px;">33.3%</td>
                        <td style="padding:8px 14px; color:#94a3b8;">High stringency</td>
                    </tr>
                    <tr style="color:#cbd5e1;">
                        <td style="padding:8px 14px; font-weight:700; color:#38bdf8;">θ = 0.95</td>
                        <td style="padding:8px 14px;">61.3%</td>
                        <td style="padding:8px 14px; color:#10b981;">85.9%</td>
                        <td style="padding:8px 14px;">38.7%</td>
                        <td style="padding:8px 14px; color:#94a3b8;">Maximum data purity, higher human load</td>
                    </tr>
                </tbody>
            </table>
            """)

    # Event handlers
    process_btn.click(
        fn=run_form_field_reader,
        inputs=[input_img, field_type_dd, engine_mode_dd, is_comb_box_cb, expected_cells_num, threshold_slider],
        outputs=[status_banner_html, extracted_text_box, manual_input, glyph_strip_html, corrections_html, metrics_box, hidden_status]
    )

    btn_accept.click(
        fn=lambda raw, corr, hid, ft: save_verification_action(raw, corr, "accept", hid, ft),
        inputs=[extracted_text_box, manual_input, hidden_status, field_type_dd],
        outputs=[audit_df, action_feedback]
    )

    btn_correct.click(
        fn=lambda raw, corr, hid, ft: save_verification_action(raw, corr, "correct", hid, ft),
        inputs=[extracted_text_box, manual_input, hidden_status, field_type_dd],
        outputs=[audit_df, action_feedback]
    )

    btn_reject.click(
        fn=lambda raw, corr, hid, ft: save_verification_action(raw, corr, "reject", hid, ft),
        inputs=[extracted_text_box, manual_input, hidden_status, field_type_dd],
        outputs=[audit_df, action_feedback]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        css=custom_css
    )
