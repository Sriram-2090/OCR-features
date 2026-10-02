"""
Integrated Handwriting Analyzer — Full Multi-Model Suite
=========================================================
Unifies:
  1. TrOCR Vision-Language Transformer (microsoft/trocr-base-handwritten)
     - Line-level recognition with Token-to-Ink Spatial Alignment & Cross-Attention Heatmaps
  2. TrOCR Context-Aware OCR Engine (replaces legacy CRNN)
     - Multi-line segmentation via line detection
     - Per-token stroke primitive extraction (ascenders, descenders, loops)
     - OCR-Derived Dysgraphia Diagnostic Signals (confidence variance,
       context rescue rate, stroke agreement, visual-language disagreement)
  3. BHK Biomechanical Feature Engine (22+ Clinical Metrics)
     - Multi-line baseline drift, letter size variability, inter-component gap CV,
       stroke tremor, collision ratio, slant standard deviation, spatial & motor dysgraphia scores
     - Explainability visualization overlay (bounding boxes + fitted baselines)
  4. Clinical Dysgraphia Ensemble Classifier (Random Forest + XGBoost + SVM)
     - Calibrated screening probability, risk classification, and actionable flags
"""

from __future__ import annotations

import os
import sys
import time
import base64
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

import cv2
import numpy as np

# ── Ensure unified path setup across Hackathon and Dysgraphia repos ────────────
HACKATHON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DYSGRAPHIA_ROOT = os.path.abspath(os.path.join(HACKATHON_ROOT, "..", "Dysgraphia", "Dysgraphia-Detection"))

if HACKATHON_ROOT not in sys.path:
    sys.path.insert(0, HACKATHON_ROOT)

import src  # noqa
dys_src = os.path.join(DYSGRAPHIA_ROOT, "src")
if os.path.exists(dys_src) and dys_src not in src.__path__:
    src.__path__.append(dys_src)

# ── Load Dysgraphia repo modules ──────────────────────────────────────────────
_bhk_available = False
_crnn_available = False
_clf_available = False

extract_bhk_features = None
generate_feature_visualization = None
preprocess_handwriting_image = None
ContextAwareOCRPipeline = None
ConfidenceTier = None

try:
    from src.bhk_features import (
        extract_bhk_features,
        generate_feature_visualization,
        FEATURE_NAMES,
    )
    from src.preprocessing import preprocess_handwriting_image
    _bhk_available = True
    print("[HandwritingAnalyzer] BHK Biomechanical Engine: AVAILABLE")
except Exception as e:
    print(f"[HandwritingAnalyzer] BHK Engine init failed: {e}")

try:
    from src.field_reader.trocr_ocr_engine import TrOCROCRPipeline
    from src.ocr.utils import ConfidenceTier
    _crnn_available = True
    print("[HandwritingAnalyzer] TrOCR OCR Engine (replaces CRNN): AVAILABLE")
except Exception as e:
    print(f"[HandwritingAnalyzer] TrOCR OCR Engine init failed: {e}")

# ── Load Trained Dysgraphia Ensemble Classifier ────────────────────────────────
_clf_bundle = None
CLASSIFIER_CANDIDATES = [
    os.path.join(HACKATHON_ROOT, "models", "dysgraphia_classifier.pkl"),
    os.path.join(DYSGRAPHIA_ROOT, "model_bundle.pkl"),
]

for p in CLASSIFIER_CANDIDATES:
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                _clf_bundle = pickle.load(f)
            _clf_available = True
            meta = _clf_bundle.get("metadata", {})
            print(f"[HandwritingAnalyzer] Dysgraphia Classifier loaded: {os.path.basename(p)}")
            print(f"  Version: {meta.get('version', 'v2')}, Samples: {meta.get('total_samples', 'N/A')}")
            break
        except Exception as e:
            print(f"[HandwritingAnalyzer] Classifier load failed ({p}): {e}")

# ── Load TrOCR Aligner ────────────────────────────────────────────────────────
from src.field_reader.trocr_aligner import get_trocr_aligner  # noqa


class HandwritingAnalyzer:
    """
    Unified multi-model handwriting analyzer:
      - TrOCR (Vision-Language Transformer)
      - IAM CRNN (CTC + Stroke Primitives + Context Rescue)
      - BHK (Biomechanical Diagnostics)
      - Ensemble (Dysgraphia Risk Classification)
    """

    def __init__(self):
        # 1. TrOCR
        self.trocr = get_trocr_aligner()

        # 2. TrOCR OCR Pipeline (replaces legacy CRNN)
        self.crnn_pipe: Optional[Any] = None
        if _crnn_available:
            try:
                self.crnn_pipe = TrOCROCRPipeline(
                    conf_threshold=0.85,
                    beam_width=15,
                    stroke_weight=0.15,
                )
                print("[HandwritingAnalyzer] TrOCR OCR Pipeline initialized (replaces CRNN)")
            except Exception as e:
                print(f"[HandwritingAnalyzer] Error initializing TrOCR OCR pipeline: {e}")

        # Status
        models_active = ["TrOCR-Base-Handwritten"]
        if self.crnn_pipe is not None:
            models_active.append("TrOCR-OCR-Engine")
        if _bhk_available:
            models_active.append("BHK-Biomechanical-Engine")
        if _clf_available:
            models_active.append("Ensemble-Dysgraphia-Classifier")
        self.models_active = models_active
        print(f"[HandwritingAnalyzer] Active Models: {', '.join(models_active)}")

    def analyze(
        self,
        image_input: Any,
        field_type: str = "General",
        conf_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Run end-to-end multi-model analysis on any handwriting image.
        Returns full OCR transcription, letter/word hypotheses, token-to-ink alignment,
        stroke primitives, 22 BHK features, and dysgraphia risk classification.
        """
        t_start = time.time()
        img_bgr = self._decode_to_bgr(image_input)

        # ── 1. TrOCR Vision-Language Inference & Spatial Alignment ────────────
        trocr_res = self.trocr.predict_and_align(
            img_bgr, field_type=field_type, conf_threshold=conf_threshold
        )

        # ── 2. Preprocessing & BHK Biomechanical Feature Extraction ──────────
        bhk_features: Dict[str, float] = {}
        bhk_annotated_b64: Optional[str] = None
        binary_mask: Optional[np.ndarray] = None

        if _bhk_available and preprocess_handwriting_image and extract_bhk_features:
            try:
                binary_mask, _ = preprocess_handwriting_image(img_bgr)
                raw_bhk, _ = extract_bhk_features(binary_mask)
                bhk_features = {k: round(float(v), 5) for k, v in raw_bhk.items()}

                # Generate explainability visualization overlay
                if generate_feature_visualization:
                    try:
                        vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        overlay = generate_feature_visualization(binary_mask, vis_rgb)
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        _, buf = cv2.imencode(".png", overlay_bgr)
                        bhk_annotated_b64 = f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"
                    except Exception as e:
                        print(f"[HandwritingAnalyzer] BHK visualizer warning: {e}")
            except Exception as e:
                print(f"[HandwritingAnalyzer] BHK feature extraction warning: {e}")

        # ── 3. IAM-Trained CRNN Transcription & Stroke Diagnostics ────────────
        crnn_data: Dict[str, Any] = {
            "available": False,
            "text": "",
            "mean_conf_pct": 0.0,
            "lines": [],
            "ocr_dysgraphia_signals": {},
        }

        if self.crnn_pipe is not None:
            try:
                crnn_trans = self.crnn_pipe.transcribe(img_bgr)
                crnn_lines = []
                for li, line in enumerate(crnn_trans.lines):
                    line_words = []
                    for wi, w in enumerate(line.words):
                        # Extract stroke analysis if available
                        stroke_prims = []
                        n_asc = n_desc = n_loops = 0
                        sa = w.metadata.get("stroke_analysis") if hasattr(w, "metadata") and isinstance(w.metadata, dict) else None
                        if sa and hasattr(sa, "primitives"):
                            stroke_prims = [p.value if hasattr(p, "value") else str(p) for p in sa.primitives]
                            n_asc = getattr(sa, "n_ascenders", 0)
                            n_desc = getattr(sa, "n_descenders", 0)
                            n_loops = getattr(sa, "n_loops", 0)

                        # Letter hypotheses
                        char_hyps = []
                        if hasattr(w, "char_hypotheses") and w.char_hypotheses:
                            for ch in w.char_hypotheses:
                                alts = [{"char": a[0], "conf": round(float(a[1]) * 100, 1)} for a in ch.alternatives[:3]] if hasattr(ch, "alternatives") else []
                                char_hyps.append({
                                    "char": ch.char,
                                    "conf": round(float(ch.confidence) * 100, 1),
                                    "alts": alts,
                                })

                        tier_str = getattr(w.tier, "value", str(w.tier)) if hasattr(w, "tier") else "MEDIUM"
                        bbox_coords = [int(v) for v in w.bbox] if getattr(w, "bbox", None) is not None else None

                        line_words.append({
                            "word_idx": wi + 1,
                            "text": w.text,
                            "confidence": round(float(w.confidence) * 100, 1),
                            "tier": tier_str,
                            "bbox": bbox_coords,
                            "context_rescued": bool(w.metadata.get("context_rescued")) if hasattr(w, "metadata") and isinstance(w.metadata, dict) else False,
                            "stroke_primitives": stroke_prims,
                            "ascenders": n_asc,
                            "descenders": n_desc,
                            "loops": n_loops,
                            "char_hypotheses": char_hyps,
                            "visual_score": round(float(w.visual_score), 3) if hasattr(w, "visual_score") else None,
                            "language_score": round(float(w.language_score), 3) if hasattr(w, "language_score") else None,
                        })

                    crnn_lines.append({
                        "line_idx": li + 1,
                        "raw_text": line.raw_text,
                        "line_confidence": round(float(line.line_confidence) * 100, 1),
                        "words": line_words,
                    })

                # OCR Dysgraphia features
                ocr_feats = crnn_trans.metadata.get("ocr_dysgraphia_features")
                signals = {}
                if ocr_feats:
                    for attr in [
                        "mean_word_confidence",
                        "fraction_low_confidence_words",
                        "word_confidence_variance",
                        "character_substitution_rate",
                        "context_rescue_rate",
                        "visual_language_disagreement",
                        "mean_stroke_agreement",
                    ]:
                        val = getattr(ocr_feats, attr, None)
                        if val is not None:
                            signals[attr] = round(float(val), 4)

                crnn_data = {
                    "available": True,
                    "text": crnn_trans.full_text,
                    "mean_conf_pct": round(float(crnn_trans.mean_confidence) * 100, 1),
                    "total_lines": len(crnn_lines),
                    "total_words": crnn_trans.total_words,
                    "lines": crnn_lines,
                    "ocr_dysgraphia_signals": signals,
                }
            except Exception as e:
                print(f"[HandwritingAnalyzer] CRNN transcription error: {e}")

        # ── 4. Clinical Dysgraphia Risk Classification ───────────────────────
        trocr_conf_norm = trocr_res.get("mean_conf", 0.85)
        dysgraphia_risk = self._classify_dysgraphia(bhk_features, trocr_conf_norm, crnn_data.get("ocr_dysgraphia_signals"))

        # ── 5. Primary Synthesis (Document-Level Text Selection) ─────────────
        # For multi-line text where CRNN has high confidence, offer synthesis
        primary_text = trocr_res["text"]
        if crnn_data["available"] and crnn_data["total_lines"] > 1 and crnn_data["mean_conf_pct"] > 70.0:
            # Multi-line document with high CRNN confidence
            if len(crnn_data["text"]) > len(primary_text):
                primary_text = crnn_data["text"]

        total_ms = round((time.time() - t_start) * 1000, 2)

        return {
            # Backward compatibility keys for frontend
            "text": primary_text,
            "mean_conf": trocr_res.get("mean_conf", 0.0),
            "mean_conf_pct": trocr_res.get("mean_conf_pct", 0.0),
            "min_conf_pct": trocr_res.get("min_conf_pct", 0.0),
            "num_tokens": trocr_res.get("num_tokens", 0),
            "tokens": trocr_res.get("tokens", []),
            "annotated_image_b64": trocr_res.get("annotated_image_b64"),
            "field_type": field_type,
            "total_latency_ms": total_ms,
            "analysis_mode": "full_multimodal",
            "models_loaded": self.models_active,

            # Complete Sub-Engine Payloads
            "trocr": {
                "text": trocr_res["text"],
                "mean_conf_pct": trocr_res["mean_conf_pct"],
                "min_conf_pct": trocr_res["min_conf_pct"],
                "tokens": trocr_res["tokens"],
                "annotated_image_b64": trocr_res.get("annotated_image_b64"),
            },
            "iam_crnn": crnn_data,
            "bhk_features": bhk_features,
            "bhk_annotated_image_b64": bhk_annotated_b64,
            "dysgraphia_risk": dysgraphia_risk,
        }

    def _classify_dysgraphia(
        self,
        bhk: Dict[str, float],
        trocr_conf: float,
        ocr_signals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Classifies dysgraphia risk using the trained ensemble model (RF + XGB + SVM)
        grounded by BHK biomechanical metrics and OCR-derived signals.
        """
        bhk_summary = {
            "baseline_drift_slope": round(bhk.get("baseline_drift_slope", 0.0), 4),
            "baseline_waviness": round(bhk.get("baseline_drift_residual_norm", 0.0), 4),
            "letter_size_cv": round(bhk.get("letter_size_cv", 0.0), 4),
            "spacing_cv": round(bhk.get("inter_component_gap_cv", 0.0), 4),
            "collision_ratio": round(bhk.get("letter_collision_ratio", 0.0), 4),
            "stroke_tremor": round(bhk.get("stroke_tremor_high_freq", 0.0), 5),
            "slant_std": round(bhk.get("slant_angle_std", 18.0), 2),
            "spatial_dysgraphia_score": round(max(0.0, bhk.get("spatial_dysgraphia_score", 0.0)), 3),
            "motor_dysgraphia_score": round(max(0.0, bhk.get("motor_dysgraphia_score", 0.0)), 3),
            "cursive_index": round(bhk.get("cursive_index", 0.0), 3),
            "is_cursive": bool(bhk.get("is_cursive", 0.0) > 0.5),
            "line_count": int(bhk.get("line_count", 1)),
        }

        # Collect clinical warning flags
        flags: List[str] = []
        if bhk.get("baseline_drift_slope", 0) > 0.10:
            flags.append("Severe baseline drift detected (BHK #3)")
        elif bhk.get("baseline_drift_slope", 0) > 0.05:
            flags.append("Mild baseline drift detected")

        if bhk.get("letter_size_cv", 0) > 0.38:
            flags.append("High letter size variability (BHK #8)")
        if bhk.get("inter_component_gap_cv", 0) > 0.75:
            flags.append("Irregular inter-character spacing (BHK #4)")
        if bhk.get("letter_collision_ratio", 0) > 0.10:
            flags.append("Frequent letter collisions / overlapping (BHK #7)")
        if bhk.get("stroke_tremor_high_freq", 0) > 0.02:
            flags.append("Neuromotor stroke tremor detected")
        if bhk.get("slant_angle_std", 18) > 28.0:
            flags.append("Inconsistent stroke slant angles")

        if ocr_signals:
            if ocr_signals.get("mean_stroke_agreement", 1.0) < 0.60:
                flags.append("Low stroke-to-character agreement (impaired motor control)")
            if ocr_signals.get("context_rescue_rate", 0.0) > 0.60:
                flags.append("High contextual rescue rate (letters illegible in isolation)")

        if trocr_conf < 0.55:
            flags.append(f"Reduced legibility (Vision OCR confidence: {trocr_conf:.0%})")

        # ── 1. If trained ML ensemble is loaded, use it ──────────────────────
        if _clf_available and _clf_bundle is not None:
            try:
                feat_names = _clf_bundle["feature_names"]
                scaler = _clf_bundle["scaler"]
                model = _clf_bundle["ensemble_model"]
                threshold = _clf_bundle.get("optimal_threshold", 0.45)

                x = np.array([[bhk.get(k, 0.0) for k in feat_names]], dtype=np.float32)
                x_sc = scaler.transform(x)
                prob = float(model.predict_proba(x_sc)[0][1])
                pred = int(prob >= threshold)

                if prob < 0.35:
                    risk_level = "Low"
                elif prob < 0.60:
                    risk_level = "Moderate"
                else:
                    risk_level = "High"

                if not flags:
                    flags.append("No significant dysgraphia indicators detected")

                return {
                    "prediction": pred,
                    "prediction_label": "Dysgraphic" if pred == 1 else "Control",
                    "dysgraphia_probability": round(prob * 100, 1),
                    "risk_level": risk_level,
                    "risk_score": round(prob, 3),
                    "threshold": threshold,
                    "flags": flags,
                    "bhk_summary": bhk_summary,
                    "classifier": _clf_bundle.get("metadata", {}).get("version", "Ensemble-RF-XGB-SVM"),
                    "trained_on": _clf_bundle.get("metadata", {}).get("train_dataset", "Malay + Slovak"),
                }
            except Exception as e:
                print(f"[HandwritingAnalyzer] ML prediction warning: {e}")

        # ── 2. Rule-based fallback ───────────────────────────────────────────
        score = 0.0
        if bhk.get("baseline_drift_slope", 0) > 0.10: score += 0.22
        if bhk.get("letter_size_cv", 0) > 0.38: score += 0.20
        if bhk.get("inter_component_gap_cv", 0) > 0.75: score += 0.15
        if bhk.get("stroke_tremor_high_freq", 0) > 0.02: score += 0.18
        if bhk.get("letter_collision_ratio", 0) > 0.10: score += 0.12
        if trocr_conf < 0.55: score += 0.13
        score = min(1.0, score)

        level = "Low" if score < 0.25 else ("Moderate" if score < 0.55 else "High")
        if not flags:
            flags.append("Writing patterns within normal clinical range")

        return {
            "prediction": 1 if score >= 0.50 else 0,
            "prediction_label": "Dysgraphic" if score >= 0.50 else "Control",
            "dysgraphia_probability": round(score * 100, 1),
            "risk_level": level,
            "risk_score": round(score, 3),
            "flags": flags,
            "bhk_summary": bhk_summary,
            "classifier": "Rule-Based Clinical Heuristic",
        }

    def _decode_to_bgr(self, image_input: Any) -> np.ndarray:
        """Robust multi-format image decoder."""
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                return image_input
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)

        if isinstance(image_input, str):
            if os.path.exists(image_input):
                img = cv2.imread(image_input)
                if img is not None:
                    return img
            # Attempt base64 decode
            raw = image_input.split(",", 1)[-1] if "," in image_input else image_input
            nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img

        try:
            from PIL import Image as PILImage
            if isinstance(image_input, PILImage.Image):
                return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise ValueError(f"Unable to decode image of type: {type(image_input)}")


_global_analyzer: Optional[HandwritingAnalyzer] = None


def get_handwriting_analyzer() -> HandwritingAnalyzer:
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = HandwritingAnalyzer()
    return _global_analyzer
