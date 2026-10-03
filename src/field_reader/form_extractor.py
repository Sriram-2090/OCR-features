"""
Full-Page Form Template Alignment, Field Extraction, and End-to-End OCR Pipeline.
Fulfills Track B Deliverable 1: "Prepare cropped field images and labels".

Workflow:
  1. Ingest full-page filled handwritten form.
  2. Register & align document to canonical template coordinates (1200x1650).
  3. Extract and crop individual form fields with margin padding.
  4. Perform multi-model OCR on each cropped field (Comb-Box Cell CNN + FSM + TrOCR).
  5. Package annotated full form with visual bounding boxes & structured field outputs.
  6. Export cropped field images and ground truth / predicted labels to CSV/JSON.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import json
import time
import base64
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import pandas as pd
import torch

try:
    from src.field_reader.model import FieldCharacterCNN, VOCAB_ALPHANUMERIC, VOCAB_DATE_NUMERIC
    from src.field_reader.segmenter import normalize_glyph
    from src.field_reader.decoder import FormFieldGrammarDecoder
    from src.field_reader.trocr_aligner import get_trocr_aligner, sanitize_verbatim_text
except ImportError:
    from model import FieldCharacterCNN, VOCAB_ALPHANUMERIC, VOCAB_DATE_NUMERIC
    from segmenter import normalize_glyph
    from decoder import FormFieldGrammarDecoder
    from trocr_aligner import get_trocr_aligner, sanitize_verbatim_text

repo_root = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"
TEMPLATE_PATH = os.path.join(repo_root, "data", "templates", "form_template_blank.png")
SCHEMA_PATH = os.path.join(repo_root, "data", "templates", "template_schema.json")
CROPS_OUTPUT_DIR = os.path.join(repo_root, "data", "extracted_crops")


class FormTemplateExtractor:
    """
    Automated document aligner and field extraction engine for structured handwritten forms.
    """

    def __init__(self, schema_path: str = SCHEMA_PATH, template_path: str = TEMPLATE_PATH):
        self.schema_path = schema_path
        self.template_path = template_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load schema definition
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            self.schema = {"canvas_width": 1200, "canvas_height": 1650, "fields": []}

        self.canonical_w = self.schema.get("canvas_width", 1200)
        self.canonical_h = self.schema.get("canvas_height", 1650)
        self.fields = self.schema.get("fields", [])

        # Load blank template for alignment reference if available
        self.blank_template = cv2.imread(template_path) if os.path.exists(template_path) else None

        # Load trained Character CNN for comb-box cells
        ckpt_path = os.path.join(repo_root, "models", "field_cnn.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.vocab = ckpt.get("vocab", VOCAB_ALPHANUMERIC)
            self.cnn = FieldCharacterCNN(num_classes=len(self.vocab)).to(self.device)
            self.cnn.load_state_dict(ckpt["model_state_dict"])
            self.cnn.eval()
            self.cnn_loaded = True
        else:
            self.cnn = None
            self.vocab = VOCAB_ALPHANUMERIC
            self.cnn_loaded = False

        self.trocr = None  # Lazy-loaded when needed

    def align_form_document(self, form_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Aligns incoming form image to canonical template dimensions (1200x1650).
        Uses fiducial/feature matching with fallback to high-precision aspect-ratio scaling.
        """
        h, w = form_bgr.shape[:2]

        if w == self.canonical_w and h == self.canonical_h:
            return form_bgr.copy(), 1.0

        # Attempt ORB feature matching if template exists and size differs significantly
        if self.blank_template is not None and abs(w - self.canonical_w) > 50:
            try:
                orb = cv2.ORB_create(nfeatures=1500)
                kp1, des1 = orb.detectAndCompute(cv2.cvtColor(self.blank_template, cv2.COLOR_BGR2GRAY), None)
                kp2, des2 = orb.detectAndCompute(cv2.cvtColor(form_bgr, cv2.COLOR_BGR2GRAY), None)

                if des1 is not None and des2 is not None and len(kp2) >= 15:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

                    if len(matches) >= 15:
                        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if H is not None:
                            aligned = cv2.warpPerspective(form_bgr, H, (self.canonical_w, self.canonical_h))
                            return aligned, 0.98
            except Exception as e:
                print(f"[FormExtractor] Homography alignment note: {e}")

        # Fallback: Clean smooth resize to canonical dimensions
        aligned = cv2.resize(form_bgr, (self.canonical_w, self.canonical_h), interpolation=cv2.INTER_CUBIC)
        return aligned, 0.95

    def extract_fields(
        self,
        form_bgr: np.ndarray,
        save_crops: bool = True,
        form_id: str = "form_001"
    ) -> List[Dict[str, Any]]:
        """
        Crops all individual fields defined in template schema.
        Returns array of cropped field images with coordinates and metadata.
        """
        aligned_form, align_conf = self.align_form_document(form_bgr)
        extracted = []

        if save_crops:
            form_crops_dir = os.path.join(CROPS_OUTPUT_DIR, form_id)
            os.makedirs(form_crops_dir, exist_ok=True)
        else:
            form_crops_dir = None

        for field_def in self.fields:
            fid = field_def["id"]
            fname = field_def["name"]
            ftype = field_def["type"]
            is_comb = field_def.get("is_comb_box", True)
            n_cells = field_def.get("num_cells", 1)

            x, y, w, h = field_def["box"]

            # Tight box crop
            pad = 2
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(self.canonical_w, x + w + pad)
            y_max = min(self.canonical_h, y + h + pad)

            crop = aligned_form[y_min:y_max, x_min:x_max].copy()

            crop_file_path = None
            if save_crops and form_crops_dir is not None:
                crop_fname = f"{fid}.png"
                crop_file_path = os.path.join(form_crops_dir, crop_fname)
                cv2.imwrite(crop_file_path, crop)

            # Encode crop to base64
            _, buf = cv2.imencode(".png", crop)
            crop_b64 = f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

            extracted.append({
                "field_id": fid,
                "field_name": fname,
                "field_type": ftype,
                "is_comb_box": is_comb,
                "num_cells": n_cells,
                "box": [x, y, w, h],
                "crop_img": crop,
                "crop_b64": crop_b64,
                "crop_path": crop_file_path,
                "description": field_def.get("description", "")
            })

        return extracted

    def _ocr_comb_box_cells(
        self,
        aligned_form: np.ndarray,
        field_def: Dict[str, Any],
        conf_threshold: float = 0.85
    ) -> Tuple[str, float, float, List[Dict[str, Any]], List[str]]:
        """
        Slices each individual cell inside the comb-box directly from the aligned form,
        classifies with FieldCharacterCNN, and performs FSM grammar verification.
        """
        x, y, w, h = field_def["box"]
        n_cells = field_def["num_cells"]
        cell_w = field_def.get("cell_w", w // n_cells)
        ftype = field_def.get("type", field_def.get("field_type", "General")).lower()

        char_hypotheses = []
        glyphs = []

        for i in range(n_cells):
            # Inset by 4px from cell border to eliminate printed divider lines
            cx1 = x + i * cell_w + 4
            cx2 = x + (i + 1) * cell_w - 4
            cy1 = y + 4
            cy2 = y + h - 4

            cell_patch = aligned_form[cy1:cy2, cx1:cx2]
            gray = cv2.cvtColor(cell_patch, cv2.COLOR_BGR2GRAY)
            _, bin_cell = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Robust blank cell check using pixel contrast & standard deviation
            gray_std = float(gray.std())
            gray_range = int(gray.max()) - int(gray.min())
            if gray_std < 8.0 or gray_range < 30:
                char_hypotheses.append((" ", 1.0, [(" ", 1.0)]))
                glyphs.append({
                    "char": " ",
                    "conf": 1.0,
                    "bbox": [cx1, cy1, cx2 - cx1, cy2 - cy1],
                    "alternatives": [(" ", 1.0)]
                })
                continue

            tensor, _ = normalize_glyph(bin_cell)
            with torch.no_grad():
                logits = self.cnn(tensor.to(self.device))
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                # Restrict vocabulary based on field type
                if "pin" in ftype or "phone" in ftype:
                    for idx, c in enumerate(self.vocab):
                        if not c.isdigit():
                            probs[idx] = 0.0
                elif "date" in ftype:
                    if i in (2, 5):
                        for idx, c in enumerate(self.vocab):
                            if c not in ("/", "-", "."):
                                probs[idx] = 0.0
                    else:
                        for idx, c in enumerate(self.vocab):
                            if not c.isdigit():
                                probs[idx] = 0.0
                elif "name" in ftype:
                    for idx, c in enumerate(self.vocab):
                        if not c.isalpha():
                            probs[idx] = 0.0

                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum

                top_vals, top_idxs = torch.topk(probs, k=min(3, len(self.vocab)))
                top_char = self.vocab[top_idxs[0].item()]
                top_conf = float(top_vals[0].item())
                alts = [(self.vocab[idx.item()], round(float(v.item()), 3)) for v, idx in zip(top_vals, top_idxs)]

            char_hypotheses.append((top_char, top_conf, alts))
            glyphs.append({
                "char": top_char,
                "conf": round(top_conf, 3),
                "bbox": [cx1, cy1, cx2 - cx1, cy2 - cy1],
                "alternatives": alts
            })

        # FSM and Semantic Grammar Verification
        corrections = []
        if "name" in ftype:
            raw_text = " ".join("".join(c[0] for c in char_hypotheses).split())
            fsm_text = raw_text
            fsm_conf = float(np.mean([c[1] for c in char_hypotheses if c[0] != " "])) if char_hypotheses else 0.92
            syntax_valid = len(fsm_text) >= 2
        elif "phone" in ftype:
            raw_digits = "".join(c[0] for c in char_hypotheses if c[0].isdigit())[:10]
            fsm_text = raw_digits
            fsm_conf = float(np.mean([c[1] for c in char_hypotheses[:10]])) if char_hypotheses else 0.95
            syntax_valid = (len(fsm_text) == 10)
        elif "date" in ftype:
            fsm_text, fsm_conf, fsm_corrs, syntax_valid = FormFieldGrammarDecoder.decode_date(char_hypotheses)
            corrections.extend(fsm_corrs)
        elif "code" in ftype:
            fsm_text, fsm_conf, fsm_corrs, syntax_valid = FormFieldGrammarDecoder.decode_code(char_hypotheses)
            corrections.extend(fsm_corrs)
        elif "pin" in ftype:
            fsm_text, fsm_conf, fsm_corrs, syntax_valid = FormFieldGrammarDecoder.decode_pin(char_hypotheses)
            corrections.extend(fsm_corrs)
        else:
            raw_text = "".join(c[0] for c in char_hypotheses).strip()
            fsm_text = raw_text
            fsm_conf = float(np.mean([c[1] for c in char_hypotheses])) if char_hypotheses else 0.90
            syntax_valid = True

        non_blank_confs = [c[1] for c in char_hypotheses if c[0] != " "]
        mean_c = float(np.mean(non_blank_confs)) if non_blank_confs else 0.90
        min_c = float(min(non_blank_confs)) if non_blank_confs else 0.85

        return fsm_text, round(mean_c, 4), round(min_c, 4), glyphs, corrections

    def process_form(
        self,
        form_bgr: np.ndarray,
        form_id: str = "form_001",
        conf_threshold: float = 0.85,
        save_crops: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end full form digitization:
          1. Align document
          2. Crop all fields
          3. Perform OCR on each crop
          4. Overlay visual bounding boxes on full form
          5. Package unified response
        """
        t0 = time.perf_counter()
        aligned_form, align_conf = self.align_form_document(form_bgr)
        extracted_fields = self.extract_fields(aligned_form, save_crops=save_crops, form_id=form_id)

        annotated_form = aligned_form.copy()
        processed_fields = []
        all_approved = True
        total_conf = 0.0

        for ef in extracted_fields:
            fid = ef["field_id"]
            ftype = ef["field_type"]
            is_comb = ef["is_comb_box"]
            crop = ef["crop_img"]
            bx, by, bw, bh = ef["box"]

            field_text = ""
            field_conf = 0.90
            min_c = 0.85
            field_status = "APPROVED"
            glyphs = []
            corrections = []

            if is_comb and self.cnn_loaded:
                # Direct cell slicing + Character CNN + FSM
                field_text, field_conf, min_c, glyphs, corrections = self._ocr_comb_box_cells(
                    aligned_form, ef, conf_threshold=conf_threshold
                )
                is_app = (min_c >= conf_threshold)
                field_status = "APPROVED" if is_app else "FLAGGED"
            else:
                # Freeform handwriting sentence (Declaration) -> Process via TrOCR
                if self.trocr is None:
                    self.trocr = get_trocr_aligner()

                res_tr = self.trocr.predict_and_align(crop, field_type="General", conf_threshold=conf_threshold)
                field_text = sanitize_verbatim_text(res_tr.get("text", ""))
                field_conf = float(res_tr.get("mean_conf", 0.88))
                min_c = max(0.5, field_conf - 0.15)
                field_status = "APPROVED" if field_conf >= conf_threshold else "FLAGGED"
                glyphs = res_tr.get("tokens", [])

            if field_status == "FLAGGED":
                all_approved = False

            total_conf += field_conf

            # Draw visual bounding box and label chip on annotated full form
            box_color = (40, 180, 60) if field_status == "APPROVED" else (0, 140, 255)  # Green or Amber
            cv2.rectangle(annotated_form, (bx, by), (bx + bw, by + bh), box_color, 3)

            # Draw floating tag badge above box
            tag_text = f"{ef['field_name']}: {field_text} ({int(field_conf*100)}%)"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cv2.rectangle(annotated_form, (bx, by - 26), (bx + tw + 14, by), box_color, -1)
            cv2.putText(annotated_form, tag_text, (bx + 7, by - 8), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            processed_fields.append({
                "field_id": fid,
                "field_name": ef["field_name"],
                "field_type": ftype,
                "is_comb_box": is_comb,
                "box": ef["box"],
                "text": field_text,
                "confidence": round(field_conf, 4),
                "conf_pct": round(field_conf * 100, 1),
                "min_conf": round(min_c, 4),
                "status": field_status,
                "is_approved": (field_status == "APPROVED"),
                "crop_b64": ef["crop_b64"],
                "crop_path": ef.get("crop_path"),
                "glyphs": glyphs,
                "corrections": corrections
            })

        mean_form_conf = total_conf / max(len(processed_fields), 1)

        # Encode full annotated form to base64
        _, ann_buf = cv2.imencode(".jpg", annotated_form, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(ann_buf).decode('utf-8')}"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "form_id": form_id,
            "overall_status": "APPROVED" if all_approved else "REVIEW_REQUIRED",
            "overall_confidence": round(mean_form_conf, 4),
            "overall_conf_pct": round(mean_form_conf * 100, 1),
            "total_fields": len(processed_fields),
            "approved_count": sum(1 for f in processed_fields if f["is_approved"]),
            "flagged_count": sum(1 for f in processed_fields if not f["is_approved"]),
            "fields": processed_fields,
            "annotated_form_b64": annotated_b64,
            "latency_ms": elapsed_ms
        }

    def export_crops_dataset(
        self,
        samples_dir: str = os.path.join(repo_root, "data", "sample_forms"),
        out_csv: str = os.path.join(CROPS_OUTPUT_DIR, "extracted_fields_metadata.csv")
    ) -> pd.DataFrame:
        """
        Processes a collection of full sample forms, extracts all cropped field images,
        and saves a clean ground-truth + OCR metadata CSV.
        Fulfills: 'Prepare cropped field images and labels'.
        """
        manifest_path = os.path.join(samples_dir, "sample_forms_manifest.json")
        if not os.path.exists(manifest_path):
            print(f"Manifest not found at {manifest_path}")
            return pd.DataFrame()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        records = []
        print(f"[*] Processing {len(manifest)} sample forms to prepare cropped fields and labels...")

        for item in manifest:
            fid = f"form_{item['form_id']:03d}"
            fpath = item["file_path"]
            gt = item["ground_truth"]
            diff = item.get("difficulty", "legible")

            form_img = cv2.imread(fpath)
            if form_img is None:
                continue

            res = self.process_form(form_img, form_id=fid, save_crops=True)

            for pf in res["fields"]:
                field_key = pf["field_id"]
                field_gt = str(gt.get(field_key, "")).strip()
                pred = pf["text"]
                is_match = (pred == field_gt)

                records.append({
                    "form_id": item["form_id"],
                    "form_name": item["filename"],
                    "difficulty": diff,
                    "field_id": field_key,
                    "field_name": pf["field_name"],
                    "field_type": pf["field_type"],
                    "is_comb_box": pf["is_comb_box"],
                    "ground_truth": field_gt,
                    "transcribed_text": pred,
                    "exact_match": is_match,
                    "confidence": pf["confidence"],
                    "status": pf["status"],
                    "crop_image_path": pf["crop_path"]
                })

        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        print(f"[OK] Extracted {len(df)} cropped field images and labels saved to:\n    {out_csv}")
        return df


_extractor_instance: Optional[FormTemplateExtractor] = None

def get_form_extractor() -> FormTemplateExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = FormTemplateExtractor()
    return _extractor_instance


if __name__ == "__main__":
    extractor = get_form_extractor()
    print("[*] Running End-to-End Form Extraction and Field Dataset Preparation...")
    df = extractor.export_crops_dataset()
    print("\nSummary of Extracted Cropped Fields:")
    print(df[["form_id", "field_name", "ground_truth", "transcribed_text", "confidence", "status"]].head(12))
