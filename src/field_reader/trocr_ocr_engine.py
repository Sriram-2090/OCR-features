"""
TrOCR Context-Aware OCR Engine
===============================
Implements the Dysgraphia Detection OCR Pipeline interface with
Line-Level Vision-Language Transformer (TrOCR) replacing the legacy CRNN.

Provides:
  1. TrOCR ViT + RoBERTa recognition with Token-to-Ink Spatial Alignment
  2. Multi-line segmentation & line-level inference
  3. Topological stroke feature extraction (ascenders, descenders, closed loops)
  4. Per-token / per-word hypothesis generation with ConfidenceTier
  5. OCR-derived dysgraphia diagnostic signals (mean confidence, variance,
     stroke agreement, context rescue rate)
  6. Direct compatibility with Dysgraphia-Detection's TranscriptionResult contracts
"""

from __future__ import annotations

import os
import sys
import time
import base64
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2

# Path setup
HACKATHON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DYSGRAPHIA_ROOT = os.path.abspath(os.path.join(HACKATHON_ROOT, "..", "Dysgraphia", "Dysgraphia-Detection"))

if HACKATHON_ROOT not in sys.path:
    sys.path.insert(0, HACKATHON_ROOT)

import src  # noqa
dys_src = os.path.join(DYSGRAPHIA_ROOT, "src")
if os.path.exists(dys_src) and dys_src not in src.__path__:
    src.__path__.append(dys_src)

from src.field_reader.trocr_aligner import get_trocr_aligner, TrOCRTokenToInkAligner
from src.preprocessing import preprocess_handwriting_image
from src.ocr.segmentation import segment_lines, LineRegion
from src.ocr.stroke_features import extract_stroke_features, compute_char_prior
from src.ocr.utils import (
    ConfidenceTier,
    assign_confidence_tier,
    CharHypothesis,
    WordHypothesis,
    LineResult,
    TranscriptionResult,
    OCRDysgraphiaFeatures,
    StrokeAnalysis,
    StrokePrimitive,
    render_confidence_overlay,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stroke Primitive Verification Table
# ---------------------------------------------------------------------------
ASCENDER_CHARS = set("bdfhkl1t")
DESCENDER_CHARS = set("gjpqy")
LOOP_CHARS = set("abdegopq0689")


def _compute_token_stroke_agreement(char: str, stroke: Optional[StrokeAnalysis]) -> float:
    """
    Computes consistency between the predicted character and detected stroke primitives.
    Returns agreement score in [0.0, 1.0].
    """
    if stroke is None or not char:
        return 0.70  # Neutral baseline

    c = char[0].lower()
    score = 0.50

    has_asc = stroke.n_ascenders > 0
    has_desc = stroke.n_descenders > 0
    has_loop = stroke.n_loops > 0

    # Ascender agreement
    if c in ASCENDER_CHARS:
        score += 0.25 if has_asc else -0.15
    else:
        score += 0.10 if not has_asc else -0.10

    # Descender agreement
    if c in DESCENDER_CHARS:
        score += 0.25 if has_desc else -0.15
    else:
        score += 0.10 if not has_desc else -0.10

    # Loop agreement
    if c in LOOP_CHARS:
        score += 0.25 if has_loop else -0.10

    return float(np.clip(score, 0.05, 1.0))


# ---------------------------------------------------------------------------
# TrOCR Context-Aware Pipeline
# ---------------------------------------------------------------------------

class TrOCROCRPipeline:
    """
    State-of-the-Art OCR Pipeline replacing CRNN with TrOCR.
    Produces identical TranscriptionResult contracts with rich token-to-ink alignment.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        conf_threshold: float = 0.85,
        beam_width: int = 15,
        stroke_weight: float = 0.15,
    ):
        self.trocr: TrOCRTokenToInkAligner = get_trocr_aligner()
        self.conf_threshold = conf_threshold
        self.beam_width = beam_width
        self.stroke_weight = stroke_weight
        logger.info("[TrOCROCRPipeline] Initialized with TrOCR Vision-Language Transformer.")

    def preprocess(self, image_input: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and precomputes binary ink mask."""
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                bgr = cv2.imread(image_input)
            else:
                b64 = image_input.split(",", 1)[-1] if "," in image_input else image_input
                bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            bgr = image_input.copy()
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if bgr is None:
            raise ValueError("Failed to decode image.")

        binary_mask, _ = preprocess_handwriting_image(bgr)
        return bgr, binary_mask

    def transcribe(self, image_input: Union[str, np.ndarray]) -> TranscriptionResult:
        """
        Transcribes the handwriting image using TrOCR.
        Handles multi-line document segmentation or single-line field crops,
        extracts stroke primitives, and compiles full OCRDysgraphiaFeatures.
        """
        t0 = time.time()
        bgr, binary_mask = self.preprocess(image_input)
        h, w = bgr.shape[:2]

        # Step 1: Detect lines
        lines: List[LineRegion] = segment_lines(binary_mask)

        processed_lines: List[LineResult] = []
        all_stroke_agreements: List[float] = []
        all_word_confs: List[float] = []
        rescued_count = 0
        total_word_count = 0

        # If document has multiple well-separated lines (>1), process line by line
        if len(lines) > 1:
            for li, line in enumerate(lines):
                lx, ly, lw, lh = line.bbox_in_image
                # Padding around line for TrOCR ascenders/descenders
                pad_y = max(8, int(lh * 0.25))
                pad_x = max(12, int(lw * 0.10))
                y1 = max(0, ly - pad_y)
                y2 = min(h, ly + lh + pad_y)
                x1 = max(0, lx - pad_x)
                x2 = min(w, lx + lw + pad_x)

                line_crop = bgr[y1:y2, x1:x2]
                mask_crop = binary_mask[y1:y2, x1:x2]

                line_res = self._transcribe_line_crop(
                    line_crop,
                    mask_crop,
                    line_offset=(x1, y1),
                    line_idx=li,
                )
                if line_res.words:
                    processed_lines.append(line_res)
                    for w_hyp in line_res.words:
                        all_word_confs.append(w_hyp.confidence)
                        total_word_count += 1
                        if w_hyp.was_context_rescued:
                            rescued_count += 1
                        agr = w_hyp.metadata.get("stroke_agreement")
                        if agr is not None:
                            all_stroke_agreements.append(agr)
        else:
            # Single line or freeform document — feed full image to TrOCR
            line_res = self._transcribe_line_crop(
                bgr,
                binary_mask,
                line_offset=(0, 0),
                line_idx=0,
            )
            if line_res.words:
                processed_lines.append(line_res)
                for w_hyp in line_res.words:
                    all_word_confs.append(w_hyp.confidence)
                    total_word_count += 1
                    if w_hyp.was_context_rescued:
                        rescued_count += 1
                    agr = w_hyp.metadata.get("stroke_agreement")
                    if agr is not None:
                        all_stroke_agreements.append(agr)

        # Full document text
        full_text = "\n".join(l.raw_text for l in processed_lines if l.raw_text).strip()
        mean_conf = float(np.mean(all_word_confs)) if all_word_confs else 0.85
        conf_var = float(np.var(all_word_confs)) if len(all_word_confs) > 1 else 0.001
        frac_low = float(np.mean([1.0 if c < 0.50 else 0.0 for c in all_word_confs])) if all_word_confs else 0.0
        rescue_rate = float(rescued_count / max(total_word_count, 1))
        mean_stroke_agr = float(np.mean(all_stroke_agreements)) if all_stroke_agreements else 0.70

        # Construct OCR-derived dysgraphia features
        ocr_dysgraphia = OCRDysgraphiaFeatures(
            mean_word_confidence=round(mean_conf, 4),
            fraction_low_confidence_words=round(frac_low, 4),
            word_confidence_variance=round(conf_var, 4),
            character_substitution_rate=round(frac_low * 0.8, 4),
            character_deletion_rate=0.0,
            character_insertion_rate=0.0,
            context_rescue_rate=round(rescue_rate, 4),
            visual_language_disagreement=round(abs(mean_conf - mean_stroke_agr), 4),
            mean_stroke_agreement=round(mean_stroke_agr, 4),
            phonetically_plausible_error_rate=0.0,
        )

        total_ms = (time.time() - t0) * 1000.0

        transcription = TranscriptionResult(
            lines=processed_lines,
            processing_time_ms=round(total_ms, 2),
            model_info="TrOCR-Base-Handwritten (Vision-Language Transformer + Spatial Alignment)",
            image_shape=bgr.shape,
            metadata={
                "ocr_dysgraphia_features": ocr_dysgraphia,
                "total_tokens": sum(len(l.words) for l in processed_lines),
            },
            full_text=full_text,
            mean_confidence=mean_conf,
        )
        return transcription

    def _transcribe_line_crop(
        self,
        crop_bgr: np.ndarray,
        mask_crop: np.ndarray,
        line_offset: Tuple[int, int],
        line_idx: int,
    ) -> LineResult:
        """
        Runs TrOCR on a single line crop, extracts token-to-ink alignment,
        extracts stroke primitives, and compiles into a LineResult.
        """
        ox, oy = line_offset
        ch, cw = crop_bgr.shape[:2]

        trocr_out = self.trocr.predict_and_align(crop_bgr, field_type="General")
        line_text = trocr_out.get("text", "")
        tokens = trocr_out.get("tokens", [])  # Each token has: char, conf, bbox, alts, ...

        words: List[WordHypothesis] = []

        if not tokens:
            # Fallback if no tokens parsed
            return LineResult(raw_text=line_text, line_confidence=0.5, words=[], bbox=(ox, oy, cw, ch))

        # Group tokens into words based on whitespace and spatial proximity
        current_word_tokens = []
        for tok in tokens:
            t_char = tok.get("char", "")
            # Check if token starts with space or is whitespace
            if (t_char.startswith(" ") or t_char == " ") and current_word_tokens:
                words.append(self._build_word_hypothesis(current_word_tokens, mask_crop, (ox, oy)))
                current_word_tokens = []
            current_word_tokens.append(tok)

        if current_word_tokens:
            words.append(self._build_word_hypothesis(current_word_tokens, mask_crop, (ox, oy)))

        line_conf = float(np.mean([w.confidence for w in words])) if words else 0.85
        return LineResult(
            raw_text=line_text,
            line_confidence=line_conf,
            words=words,
            bbox=(ox, oy, cw, ch),
        )

    def _build_word_hypothesis(
        self,
        word_tokens: List[Dict[str, Any]],
        mask_crop: np.ndarray,
        line_offset: Tuple[int, int],
    ) -> WordHypothesis:
        """
        Compiles a list of TrOCR tokens into a full WordHypothesis with
        CharHypothesis list, stroke analysis, and confidence tier.
        """
        ox, oy = line_offset
        ch, cw = mask_crop.shape[:2]

        char_hyps: List[CharHypothesis] = []
        confs = []
        min_x, min_y, max_x, max_y = 1e6, 1e6, -1, -1

        token_chars = []
        for tok in word_tokens:
            raw_char = tok.get("char", "").strip()
            if not raw_char:
                continue
            token_chars.append(raw_char)
            conf = float(tok.get("conf", 0.85))
            confs.append(conf)

            # bbox is [ymin, xmin, ymax, xmax] in line crop coordinates
            tb = tok.get("bbox", [0, 0, 10, 10])
            ymin, xmin, ymax, xmax = tb[0], tb[1], tb[2], tb[3]
            min_x = min(min_x, xmin)
            min_y = min(min_y, ymin)
            max_x = max(max_x, xmax)
            max_y = max(max_y, ymax)

            # Global image coordinates
            char_bbox = (int(xmin + ox), int(ymin + oy), int(xmax - xmin), int(ymax - ymin))

            char_hyps.append(CharHypothesis(
                char=raw_char,
                confidence=conf,
                bbox=char_bbox,
                alternatives=[],
            ))

        word_text = "".join(token_chars)
        word_conf = float(np.mean(confs)) if confs else 0.85

        if min_x < 1e6:
            wx = int(min_x + ox)
            wy = int(min_y + oy)
            ww = int(max(1, max_x - min_x))
            wh = int(max(1, max_y - min_y))
            word_bbox = (wx, wy, ww, wh)

            # Extract ink patch for stroke analysis
            py1 = max(0, int(min_y))
            py2 = min(ch, int(max_y))
            px1 = max(0, int(min_x))
            px2 = min(cw, int(max_x))
            patch = mask_crop[py1:py2, px1:px2]
        else:
            word_bbox = (ox, oy, 20, 20)
            patch = np.zeros((20, 20), dtype=np.uint8)

        # Topological stroke feature extraction
        stroke_analysis: Optional[StrokeAnalysis] = None
        stroke_agr = 0.70
        try:
            if patch.size > 20 and np.count_nonzero(patch) > 5:
                stroke_analysis = extract_stroke_features(patch)
                stroke_agr = _compute_token_stroke_agreement(word_text, stroke_analysis)
        except Exception:
            pass

        tier = assign_confidence_tier(word_conf)
        was_rescued = bool(word_conf >= 0.75 and stroke_agr < 0.45)

        return WordHypothesis(
            text=word_text,
            visual_score=round(float(np.log(max(word_conf, 1e-4))), 3),
            language_score=-1.0,
            stroke_agreement=round(stroke_agr, 4),
            fused_score=round(word_conf, 3),
            confidence=round(word_conf, 4),
            confidence_tier=tier,
            bbox=word_bbox,
            char_hypotheses=char_hyps,
            alternatives=[],
            metadata={
                "stroke_analysis": stroke_analysis,
                "stroke_agreement": round(stroke_agr, 4),
                "context_rescued": was_rescued,
            },
        )

    def render_overlay(
        self,
        original_image: np.ndarray,
        transcription: TranscriptionResult,
    ) -> np.ndarray:
        """Renders interactive confidence overlay on top of original handwriting image."""
        return render_confidence_overlay(original_image, transcription)


_global_trocr_ocr_pipeline: Optional[TrOCROCRPipeline] = None


def get_trocr_ocr_pipeline() -> TrOCROCRPipeline:
    global _global_trocr_ocr_pipeline
    if _global_trocr_ocr_pipeline is None:
        _global_trocr_ocr_pipeline = TrOCROCRPipeline()
    return _global_trocr_ocr_pipeline
