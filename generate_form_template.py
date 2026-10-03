"""
Form Template Generator & Full-Page Handwritten Form Creator.
Produces:
  1. data/templates/form_template_blank.png (Clean blank form template)
  2. data/templates/template_schema.json    (Canonical field bounding boxes & metadata)
  3. data/sample_forms/                    (Realistic filled handwritten forms)
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import json
import random
import numpy as np
import cv2

repo_root = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"
TEMPLATE_DIR = os.path.join(repo_root, "data", "templates")
SAMPLES_DIR = os.path.join(repo_root, "data", "sample_forms")
CROPS_DIR = os.path.join(repo_root, "data", "extracted_crops")

os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# Standard A4-like canvas at 150 DPI: 1200 x 1650
CANVAS_W = 1200
CANVAS_H = 1650

# Canonical Field Definitions (Labels safely ABOVE each box to eliminate text collision)
TEMPLATE_FIELDS = [
    {
        "id": "applicant_name",
        "name": "Applicant Full Name (Block Letters)",
        "type": "Name",
        "is_comb_box": True,
        "num_cells": 14,
        "box": [120, 315, 840, 60],  # x, y, w, h
        "cell_w": 60,
        "cell_h": 60,
        "description": "Applicant full legal name in block letters"
    },
    {
        "id": "date_of_birth",
        "name": "Date of Birth (DD/MM/YYYY)",
        "type": "Date",
        "is_comb_box": True,
        "num_cells": 10,
        "box": [120, 445, 600, 60],
        "cell_w": 60,
        "cell_h": 60,
        "description": "Date of birth in DD/MM/YYYY format"
    },
    {
        "id": "postal_pin",
        "name": "Postal PIN Code",
        "type": "Pin",
        "is_comb_box": True,
        "num_cells": 6,
        "box": [120, 575, 360, 60],
        "cell_w": 60,
        "cell_h": 60,
        "description": "6-digit postal PIN code"
    },
    {
        "id": "application_code",
        "name": "Application Tracking Code",
        "type": "Code",
        "is_comb_box": True,
        "num_cells": 8,
        "box": [120, 705, 480, 60],
        "cell_w": 60,
        "cell_h": 60,
        "description": "Alphanumeric code in format [A-Z]{3}-\\d{4}"
    },
    {
        "id": "phone_number",
        "name": "Primary Contact Number",
        "type": "Phone",
        "is_comb_box": True,
        "num_cells": 10,
        "box": [120, 835, 600, 60],
        "cell_w": 60,
        "cell_h": 60,
        "description": "10-digit primary mobile phone number"
    },
    {
        "id": "declaration_text",
        "name": "Handwritten Declaration & Confirmation",
        "type": "Handwriting",
        "is_comb_box": False,
        "num_cells": 1,
        "box": [120, 995, 960, 100],
        "cell_w": 960,
        "cell_h": 100,
        "description": "Handwritten confirmation and verification statement"
    }
]


def render_blank_template() -> np.ndarray:
    """Renders high-resolution crisp form template with fiducial registration markers."""
    img = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)

    # 1. Outer Border & Fiducial Alignment Markers (Corner L-shapes for homography)
    marker_size = 40
    marker_thick = 8
    # Top-Left
    cv2.line(img, (40, 40), (40 + marker_size, 40), (20, 20, 20), marker_thick)
    cv2.line(img, (40, 40), (40, 40 + marker_size), (20, 20, 20), marker_thick)
    # Top-Right
    cv2.line(img, (CANVAS_W - 40, 40), (CANVAS_W - 40 - marker_size, 40), (20, 20, 20), marker_thick)
    cv2.line(img, (CANVAS_W - 40, 40), (CANVAS_W - 40, 40 + marker_size), (20, 20, 20), marker_thick)
    # Bottom-Left
    cv2.line(img, (40, CANVAS_H - 40), (40 + marker_size, CANVAS_H - 40), (20, 20, 20), marker_thick)
    cv2.line(img, (40, CANVAS_H - 40), (40, CANVAS_H - 40 - marker_size), (20, 20, 20), marker_thick)
    # Bottom-Right
    cv2.line(img, (CANVAS_W - 40, CANVAS_H - 40), (CANVAS_W - 40 - marker_size, CANVAS_H - 40), (20, 20, 20), marker_thick)
    cv2.line(img, (CANVAS_W - 40, CANVAS_H - 40), (CANVAS_W - 40, CANVAS_H - 40 - marker_size), (20, 20, 20), marker_thick)

    # Outer Document Frame
    cv2.rectangle(img, (70, 70), (CANVAS_W - 70, CANVAS_H - 70), (60, 60, 60), 2)

    # 2. Form Header & Government / Administrative Branding
    cv2.putText(img, "GOVERNMENT ADMINISTRATIVE SERVICES", (260, 130), cv2.FONT_HERSHEY_DUPLEX, 1.05, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "OFFICIAL CITIZEN VERIFICATION & REGISTRATION FORM", (210, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(img, "FORM REF: JIG-2026-TRB / OC&HCR ENTERPRISE COMPLIANT", (310, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.line(img, (100, 240), (CANVAS_W - 100, 240), (40, 40, 40), 3)

    cv2.putText(img, "INSTRUCTIONS: Write in CLEAR BLOCK LETTERS. Each character must stay strictly within its box.", (115, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(img, (100, 290), (CANVAS_W - 100, 290), (200, 200, 200), 1)

    # 3. Draw Each Field (Label safely placed 10px ABOVE each box)
    for f in TEMPLATE_FIELDS:
        x, y, w, h = f["box"]
        # Field Label positioned safely above box
        cv2.putText(img, f["name"].upper(), (x + 2, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.60, (30, 30, 30), 1, cv2.LINE_AA)

        if f["is_comb_box"]:
            # Draw Comb-Box Outline and Internal Divider lines
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 2)
            for i in range(1, f["num_cells"]):
                cx = x + i * f["cell_w"]
                cv2.line(img, (cx, y), (cx, y + h), (120, 120, 120), 2)
        else:
            # Freeform underline / box
            cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 2)
            cv2.line(img, (x, y + h - 10), (x + w, y + h - 10), (160, 160, 160), 1)

    # Footer Section
    cv2.line(img, (100, 1480), (CANVAS_W - 100, 1480), (160, 160, 160), 1)
    cv2.putText(img, "APPLICANT SIGNATURE: ________________________", (120, 1535), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "VERIFIED DATE: ____/____/2026", (720, 1535), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "FOR OFFICIAL USE ONLY - PROCESSED VIA OC&HCR ENTERPRISE PIPELINE", (250, 1595), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1, cv2.LINE_AA)

    return img


def fill_handwritten_form(sample_id: int, difficulty: str = "legible") -> tuple[np.ndarray, dict]:
    """
    Fills the blank form template with authentic handwritten characters and words.
    difficulty: 'legible' (clean comb box digits) or 'difficult' (cursive, slant, faint ink).
    """
    blank = render_blank_template()
    filled = blank.copy()

    # Synthetic / Sample Ground Truth Profiles
    profiles = [
        {
            "applicant_name": "SRIRAM RAO",
            "date_of_birth": "15/08/1995",
            "postal_pin": "560034",
            "application_code": "APP-8402",
            "phone_number": "9845012345",
            "declaration_text": "I hereby verify that all provided details are authentic."
        },
        {
            "applicant_name": "AARTI SHARMA",
            "date_of_birth": "22/11/1988",
            "postal_pin": "110001",
            "application_code": "GOV-3914",
            "phone_number": "9123456789",
            "declaration_text": "All entries in this registration form are true and accurate."
        },
        {
            "applicant_name": "RAJESH KUMAR",
            "date_of_birth": "04/07/2001",
            "postal_pin": "400001",
            "application_code": "REG-7721",
            "phone_number": "9876543210",
            "declaration_text": "I confirm submission of official verification documents."
        },
        {
            "applicant_name": "MEERA IYER",
            "date_of_birth": "30/03/1979",
            "postal_pin": "600028",
            "application_code": "ADM-5026",
            "phone_number": "9444123890",
            "declaration_text": "The information given above is complete to my knowledge."
        },
        {
            "applicant_name": "VIKRAM JOSHI",
            "date_of_birth": "18/09/1992",
            "postal_pin": "500081",
            "application_code": "JIG-9180",
            "phone_number": "9000112233",
            "declaration_text": "I hereby certify my identity and postal location."
        }
    ]

    p = profiles[(sample_id - 1) % len(profiles)]
    ground_truth = {**p, "sample_id": sample_id, "difficulty": difficulty}

    # Font styles
    if difficulty == "legible":
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
        ink_color = (random.randint(20, 50), random.randint(20, 45), random.randint(30, 60))
    else:
        fonts = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_COMPLEX]
        ink_color = (random.randint(30, 80), random.randint(25, 75), random.randint(40, 90))

    for f in TEMPLATE_FIELDS:
        fid = f["id"]
        val = p[fid]
        bx, by, bw, bh = f["box"]

        if f["is_comb_box"]:
            # Write each character inside its cell centered
            for i, ch in enumerate(val):
                if i >= f["num_cells"]:
                    break
                if ch == " ":
                    continue

                cx = bx + i * f["cell_w"]
                cy = by

                font = random.choice(fonts)
                scale = 1.05 if difficulty == "legible" else 0.95
                thick = 2

                (tw, th), baseline = cv2.getTextSize(ch, font, scale, thick)
                px = cx + (f["cell_w"] - tw) // 2
                py = cy + (f["cell_h"] + th) // 2 - 2

                cv2.putText(filled, ch, (px, py), font, scale, ink_color, thick, cv2.LINE_AA)
        else:
            # Freeform sentence
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX if difficulty == "difficult" else cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.85 if difficulty == "difficult" else 0.78
            thick = 2
            px = bx + 25
            py = by + 60
            cv2.putText(filled, val, (px, py), font, scale, ink_color, thick, cv2.LINE_AA)

    # Add realistic subtle paper grain
    noise = np.random.normal(0, 1.2, filled.shape).astype(np.int16)
    noisy_img = np.clip(filled.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy_img, ground_truth


def build_full_form_suite():
    print("[*] Generating Canonical Form Template...")
    blank_template = render_blank_template()
    blank_path = os.path.join(TEMPLATE_DIR, "form_template_blank.png")
    cv2.imwrite(blank_path, blank_template)
    print(f"    Saved: {blank_path}")

    schema_path = os.path.join(TEMPLATE_DIR, "template_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({
            "template_name": "Official Citizen Verification Form",
            "canvas_width": CANVAS_W,
            "canvas_height": CANVAS_H,
            "fields": TEMPLATE_FIELDS
        }, f, indent=2)
    print(f"    Saved: {schema_path}")

    # Generate 5 sample filled forms (3 Clearly Legible, 2 Difficult)
    print("\n[*] Generating Sample Filled Handwritten Forms...")
    sample_manifest = []
    for i in range(1, 6):
        diff = "legible" if i <= 3 else "difficult"
        form_img, gt = fill_handwritten_form(i, difficulty=diff)
        fname = f"sample_form_{i:03d}_{diff}.png"
        fpath = os.path.join(SAMPLES_DIR, fname)
        cv2.imwrite(fpath, form_img)

        sample_manifest.append({
            "form_id": i,
            "filename": fname,
            "file_path": fpath,
            "difficulty": diff,
            "ground_truth": gt
        })
        print(f"    Created: {fname} (Difficulty: {diff})")

    manifest_path = os.path.join(SAMPLES_DIR, "sample_forms_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sample_manifest, f, indent=2)
    print(f"    Saved manifest: {manifest_path}")


if __name__ == "__main__":
    build_full_form_suite()
