"""
Dataset Generator & Loader for Handwritten Form Fields & Character Classification.
Prepares cropped field images, ground-truth labels, and PyTorch Character Datasets.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from src.field_reader.model import VOCAB_ALPHANUMERIC, VOCAB_DATE_NUMERIC
    from src.field_reader.segmenter import normalize_glyph
except ImportError:
    from model import VOCAB_ALPHANUMERIC, VOCAB_DATE_NUMERIC
    from segmenter import normalize_glyph

DATASET_DIR = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon\data\form_fields"


class SyntheticGlyphGenerator:
    """
    Generates realistic synthetic handwritten character glyphs with authentic handwriting perturbations:
    stroke thickness variation, slight rotation, shear, faint pencil, ink bleed.
    """

    FONTS = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    ]

    @classmethod
    def generate_glyph(cls, char: str, size: Tuple[int, int] = (40, 40)) -> np.ndarray:
        """Renders a single character glyph on a black background (white ink)."""
        canvas = np.zeros(size, dtype=np.uint8)
        font = random.choice(cls.FONTS)

        if char == ".":
            scale = random.uniform(0.40, 0.60)
            thickness = random.choice([1, 2])
        elif char == "-":
            scale = random.uniform(0.70, 1.00)
            thickness = random.choice([1, 2])
        elif char == "/":
            scale = random.uniform(0.75, 1.00)
            thickness = random.choice([1, 2])
        else:
            scale = random.uniform(0.70, 0.95)
            thickness = random.choice([1, 2])

        (tw, th), baseline = cv2.getTextSize(char, font, scale, thickness)
        x = max(2, (size[1] - tw) // 2 + random.randint(-2, 2))
        y = max(th + 2, (size[0] + th) // 2 + random.randint(-2, 2))

        cv2.putText(canvas, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

        # Apply random handwriting affine distortion
        angle = random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1.0)
        canvas = cv2.warpAffine(canvas, M, size, flags=cv2.INTER_LINEAR, borderValue=0)

        # Random faint stroke or noise
        if random.random() < 0.20:
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        return canvas


class FormFieldDatasetGenerator:
    """
    Generates realistic cropped form field images and labels for benchmark evaluation.
    Supports Dates, PIN codes, and Alphanumeric short codes in both comb-box and freeform styles.
    """

    @classmethod
    def generate_form_field(
        cls,
        field_type: str = "date",
        is_comb_box: bool = True
    ) -> Tuple[np.ndarray, str]:
        """
        Creates a realistic cropped form field image.
        field_type: 'date', 'pin', or 'code'.
        Returns: (bgr_image, ground_truth_text)
        """
        if field_type == "date":
            d = random.randint(1, 31)
            m = random.randint(1, 12)
            y = random.randint(1970, 2026)
            sep = random.choice(["/", "-"])
            text = f"{d:02d}{sep}{m:02d}{sep}{y:04d}"
        elif field_type == "pin":
            text = f"{random.randint(100000, 999999)}"
        else:
            prefix = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.choice([2, 3])))
            suffix = f"{random.randint(1000, 9999)}"
            text = f"{prefix}-{suffix}"

        n_chars = len(text)
        cell_w = random.randint(34, 46)
        cell_h = random.randint(45, 55)
        pad_x = 12
        pad_y = 10
        fw = pad_x * 2 + n_chars * cell_w
        fh = pad_y * 2 + cell_h

        field_img = np.full((fh, fw, 3), random.randint(245, 255), dtype=np.uint8)

        if is_comb_box:
            grid_color = (random.randint(120, 160), random.randint(120, 160), random.randint(120, 160))
            for i in range(n_chars + 1):
                gx = pad_x + i * cell_w
                cv2.line(field_img, (gx, pad_y), (gx, pad_y + cell_h), grid_color, 1)
            cv2.line(field_img, (pad_x, pad_y), (pad_x + n_chars * cell_w, pad_y), grid_color, 1)
            cv2.line(field_img, (pad_x, pad_y + cell_h), (pad_x + n_chars * cell_w, pad_y + cell_h), grid_color, 1)
        else:
            line_color = (random.randint(140, 180), random.randint(140, 180), random.randint(140, 180))
            cv2.line(field_img, (pad_x, pad_y + cell_h - 4), (pad_x + n_chars * cell_w, pad_y + cell_h - 4), line_color, 1)

        ink_color = (random.randint(20, 45), random.randint(20, 40), random.randint(30, 60))
        for i, ch in enumerate(text):
            glyph_mask = SyntheticGlyphGenerator.generate_glyph(ch, (cell_h - 8, cell_w - 6))
            cx = pad_x + i * cell_w + 3
            cy = pad_y + 4
            gh, gw = glyph_mask.shape
            y_end = min(fh, cy + gh)
            x_end = min(fw, cx + gw)
            gh_act = y_end - cy
            gw_act = x_end - cx
            if gh_act > 0 and gw_act > 0:
                cell_roi = field_img[cy:y_end, cx:x_end]
                mask_inv = (glyph_mask[:gh_act, :gw_act] > 80)
                cell_roi[mask_inv] = ink_color

        field_img = cv2.GaussianBlur(field_img, (3, 3), 0)
        return field_img, text

    @classmethod
    def build_benchmark_dataset(cls, n_samples: int = 150, out_dir: str = DATASET_DIR) -> pd.DataFrame:
        """Builds and saves a labeled benchmark dataset of form fields."""
        os.makedirs(out_dir, exist_ok=True)
        records = []

        types = ["date", "pin", "code"]
        for idx in range(n_samples):
            f_type = random.choice(types)
            is_comb = random.choice([True, False])
            img, label = cls.generate_form_field(field_type=f_type, is_comb_box=is_comb)

            filename = f"field_{idx+1:04d}_{f_type}.png"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, img)

            records.append({
                "field_id": idx + 1,
                "field_type": f_type,
                "ground_truth": label,
                "is_comb_box": is_comb,
                "image_path": path,
                "num_chars": len(label),
            })

        df = pd.DataFrame(records)
        csv_path = os.path.join(out_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"Generated {n_samples} benchmark form fields in: {out_dir}")
        return df


class CharacterDataset(Dataset):
    """
    PyTorch Dataset providing normalized 32x32 character glyphs and class labels.
    Uses normalize_glyph so training distribution matches inference exactly.
    """

    def __init__(self, vocab: str = VOCAB_ALPHANUMERIC, samples_per_class: int = 500):
        self.vocab = vocab
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.samples_per_class = samples_per_class

        self.data: List[Tuple[torch.Tensor, int]] = []
        self._build_dataset()

    def _build_dataset(self):
        for char, idx in self.char_to_idx.items():
            for _ in range(self.samples_per_class):
                raw = SyntheticGlyphGenerator.generate_glyph(char, size=(44, 44))
                tensor, _ = normalize_glyph(raw)
                # tensor is shape (1, 1, 32, 32), we store (1, 32, 32)
                self.data.append((tensor.squeeze(0), idx))

        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx]
