import cv2
import numpy as np
from PIL import Image
import torch
from src.field_reader.trocr_aligner import get_trocr_aligner

trocr = get_trocr_aligner()

# Load field 1
img = cv2.imread('data/form_fields/field_0001_date.png')
h, w = img.shape[:2]
print(f"Original: {w}x{h}")

# Test 1: Direct as is
res1 = trocr.predict_and_align(img, field_type="Date")
print("1. Direct TrOCR:         ", res1.get("text"))

# Test 2: Aspect ratio padding
# Pad height so that ratio is 4:1 with white borders
target_h = max(h, int(w / 4))
pad_top = (target_h - h) // 2
pad_bottom = target_h - h - pad_top
pad_left = 30
pad_right = 30
padded = cv2.copyMakeBorder(img, pad_top + 20, pad_bottom + 20, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
res2 = trocr.predict_and_align(padded, field_type="Date")
print("2. Padded TrOCR:         ", res2.get("text"))

# Test 3: Vertical line suppression (remove comb dividers)
gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
_, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Detect vertical lines taller than 70% of character height with width <= 3px
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.65)))
v_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel)
# Subtract vertical lines
cleaned = cv2.subtract(bin_inv, v_lines)
cleaned_bgr = cv2.cvtColor(255 - cleaned, cv2.COLOR_GRAY2BGR)
res3 = trocr.predict_and_align(cleaned_bgr, field_type="Date")
print("3. Line-Suppressed TrOCR:", res3.get("text"))
