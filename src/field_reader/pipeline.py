"""
Unified Form Field Reader Pipeline.
Supports 3 operational modes:
  1. 'raw_cnn'      : Baseline Character CNN alone (Rubric Mandate)
  2. 'grammar_fsm'  : CNN + Finite State Machine Grammar & Lexical Dictionary Decoder
  3. 'tri_engine'   : CNN + Clean Morphology + FSM Grammar + Semantic Lattice Verifier (SOTA)
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
import torch

try:
    from src.field_reader.model import FieldCharacterCNN, VOCAB_ALPHANUMERIC, VOCAB_DATE_NUMERIC
    from src.field_reader.segmenter import segment_field_characters, normalize_glyph, detect_grid_cells
    from src.field_reader.decoder import FormFieldGrammarDecoder
    from src.field_reader.semantic_verifier import SemanticFieldVerifier
except ImportError:
    from model import FieldCharacterCNN, VOCAB_ALPHANUMERIC, VOCAB_DATE_NUMERIC
    from segmenter import segment_field_characters, normalize_glyph, detect_grid_cells
    from decoder import FormFieldGrammarDecoder
    from semantic_verifier import SemanticFieldVerifier


class FormReaderPipeline:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = VOCAB_ALPHANUMERIC

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.vocab = ckpt.get("vocab", VOCAB_ALPHANUMERIC)
            self.model = FieldCharacterCNN(num_classes=len(self.vocab)).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.model_loaded = True
        else:
            self.model = None
            self.model_loaded = False

    def process(
        self,
        field_bgr: np.ndarray,
        field_type: str = "Date",
        is_comb_box: bool = True,
        expected_cells: Optional[int] = None,
        mode: str = "tri_engine",
        conf_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Executes end-to-end form field digitization.
        mode: 'raw_cnn', 'grammar_fsm', or 'tri_engine'
        """
        t0 = time.time()

        # 0. Automated Grid Detection & Schema Resolution
        auto_grid, auto_cells, _ = detect_grid_cells(field_bgr)
        if auto_grid:
            is_comb_box = True
            if not expected_cells or expected_cells <= 0:
                expected_cells = auto_cells

        f_type_lower = field_type.lower()
        if f_type_lower in ["general", "auto", "unknown"] and auto_grid:
            if auto_cells == 10:
                field_type = "Date"
                f_type_lower = "date"
            elif auto_cells == 6:
                field_type = "Pin"
                f_type_lower = "pin"
            elif auto_cells in [7, 8]:
                field_type = "Code"
                f_type_lower = "code"

        # Vocabulary restriction based on field type
        if "date" in f_type_lower:
            allowed_v = VOCAB_DATE_NUMERIC
            default_len = 10
        elif "pin" in f_type_lower:
            allowed_v = "0123456789"
            default_len = 6
        elif "code" in f_type_lower:
            allowed_v = VOCAB_ALPHANUMERIC
            default_len = 8
        else:
            allowed_v = VOCAB_ALPHANUMERIC
            default_len = expected_cells or 8

        n_cells = expected_cells if (expected_cells and expected_cells > 0) else default_len

        # 1. Segmentation
        glyphs = segment_field_characters(
            field_bgr,
            is_comb_box=is_comb_box,
            num_expected_cells=n_cells
        )

        if not glyphs:
            return {
                "text": "",
                "min_conf": 0.0,
                "mean_conf": 0.0,
                "is_approved": False,
                "status": "FLAGGED",
                "mode": mode,
                "reason": "Segmentation failed - no characters detected",
                "glyphs": [],
                "corrections": [],
                "latency_ms": (time.time() - t0) * 1000.0,
            }

        # 2. Character Recognition via CNN
        char_hypotheses = []
        glyph_details = []

        # If freeform and no expected_cells specified, process all segmented glyphs
        target_glyphs = glyphs if (not is_comb_box and not expected_cells) else glyphs[:n_cells]

        for crop, bbox in target_glyphs:
            tensor, patch_vis = normalize_glyph(crop)

            if self.model_loaded and self.model is not None:
                p_char, conf, alts = self.model.predict_patch(
                    tensor, vocab=self.vocab, allowed_vocab=allowed_v, device=self.device
                )
            else:
                p_char, conf, alts = "?", 0.50, []

            char_hypotheses.append((p_char, conf, alts))
            glyph_details.append({
                "char": p_char,
                "conf": conf,
                "alts": alts,
                "patch": patch_vis,
                "bbox": bbox,
            })

        raw_str = "".join([c[0] for c in char_hypotheses])
        raw_confs = [c[1] for c in char_hypotheses]
        raw_min_conf = float(np.min(raw_confs)) if raw_confs else 0.0

        corrections = []

        # Auto-infer field type if Auto/General
        eff_field_type = field_type
        if eff_field_type.lower() in ["general", "auto", "unknown"]:
            if len(char_hypotheses) == 10 and (raw_str[2] in "/-." or raw_str[5] in "/-."):
                eff_field_type = "Date"
            elif len(char_hypotheses) == 6 and sum(c[0].isdigit() for c in char_hypotheses) >= 4:
                eff_field_type = "Pin"
            elif "-" in raw_str and any(c.isalpha() for c in raw_str) and any(c.isdigit() for c in raw_str):
                eff_field_type = "Code"

        # 3. Mode Execution
        if mode == "raw_cnn":
            final_text = raw_str
            final_conf = raw_min_conf
            syntax_valid = True

        elif mode == "grammar_fsm":
            final_text, final_conf, corrs, syntax_valid = FormFieldGrammarDecoder.decode_field(
                char_hypotheses, eff_field_type
            )
            corrections.extend(corrs)

        else: # tri_engine
            fsm_text, fsm_conf, fsm_corrs, syntax_valid = FormFieldGrammarDecoder.decode_field(
                char_hypotheses, eff_field_type
            )
            corrections.extend(fsm_corrs)

            final_text, sem_conf, sem_reason = SemanticFieldVerifier.repair_field(
                fsm_text, eff_field_type, char_hypotheses
            )
            if sem_reason and "valid" not in sem_reason:
                corrections.append(f"Semantic Verifier: {sem_reason}")
            final_conf = min(fsm_conf, sem_conf) if fsm_conf < 0.85 else sem_conf
            syntax_valid = True

        mean_conf = float(np.mean([c[1] for c in char_hypotheses])) if char_hypotheses else 0.0

        # Gating Decision
        is_approved = (final_conf >= conf_threshold and syntax_valid)
        status = "APPROVED" if is_approved else "FLAGGED"

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "text": final_text,
            "raw_text": raw_str,
            "min_conf": round(final_conf, 4),
            "mean_conf": round(mean_conf, 4),
            "is_approved": is_approved,
            "status": status,
            "mode": mode,
            "syntax_valid": syntax_valid,
            "reason": "Automated verification passed" if is_approved else f"Confidence {final_conf*100:.1f}% below threshold {conf_threshold*100:.0f}%",
            "glyphs": glyph_details,
            "corrections": corrections,
            "latency_ms": round(elapsed_ms, 1),
        }
