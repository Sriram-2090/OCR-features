"""
Character Segmentation & 32x32 Normalization Pipeline for Form Fields.
Supports both comb-box grid fields and freeform handwritten fields.
Includes morphological grid-border suppression and clean centroid centering.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch


def normalize_glyph(
    glyph_bgr: np.ndarray,
    target_size: Tuple[int, int] = (32, 32),
    pad: int = 4
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Normalizes a cropped handwritten glyph:
    1. Binarizes ink.
    2. Filters out straight horizontal and vertical grid-border line components.
    3. Crops tight bounding box around the actual character ink.
    4. Scales preserving aspect ratio (protecting dot and hyphen proportions).
    5. Centers by centroid into a 32x32 normalized float tensor [0, 1].

    Returns:
        - tensor: torch.Tensor of shape (1, 1, 32, 32)
        - vis_patch: uint8 numpy array (32, 32) for UI preview
    """
    if len(glyph_bgr.shape) == 3:
        gray = cv2.cvtColor(glyph_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = glyph_bgr.copy()

    median_val = float(np.median(gray))
    if median_val > 120:
        _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

    ch, cw = binary.shape

    # Connected Component Filtering: Remove straight grid border lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    char_mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        # Ignore horizontal or vertical grid border lines spanning > 60% of cell with thickness <= 3
        if (bw > cw * 0.60 and bh <= 3) or (bh > ch * 0.60 and bw <= 3):
            continue
        # Ignore outer 1px border lines
        if (by <= 1 and bh <= 2) or (by + bh >= ch - 1 and bh <= 2):
            continue
        if (bx <= 1 and bw <= 2) or (bx + bw >= cw - 1 and bw <= 2):
            continue
        char_mask[labels == i] = 255

    coords = cv2.findNonZero(char_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        tight = char_mask[y:y+h, x:x+w]
    else:
        coords_raw = cv2.findNonZero(binary)
        if coords_raw is not None:
            x, y, w, h = cv2.boundingRect(coords_raw)
            tight = binary[y:y+h, x:x+w]
        else:
            patch = np.zeros(target_size, dtype=np.uint8)
            tensor = torch.zeros((1, 1, target_size[0], target_size[1]), dtype=torch.float32)
            return tensor, patch

    th, tw = target_size
    inner_h = th - 2 * pad
    inner_w = tw - 2 * pad
    gh, gw = tight.shape

    is_dot = (max(gw, gh) <= 8 and min(gw, gh) >= 1)
    is_dash = (gh <= 6 and gw >= 8)

    if is_dot:
        scale = min(1.2, inner_w / max(gw, 1))
    elif is_dash:
        scale = min(inner_w / max(gw, 1), 2.0)
    else:
        scale = min(inner_w / max(gw, 1), inner_h / max(gh, 1))

    new_w = max(1, min(inner_w, int(round(gw * scale))))
    new_h = max(1, min(inner_h, int(round(gh * scale))))
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST)

    patch = np.zeros(target_size, dtype=np.uint8)
    dx = (tw - new_w) // 2
    dy = (th - new_h) // 2
    patch[dy:dy+new_h, dx:dx+new_w] = resized

    norm_float = patch.astype(np.float32) / 255.0
    tensor = torch.from_numpy(norm_float).unsqueeze(0).unsqueeze(0)
    return tensor, patch


def segment_field_characters(
    field_bgr: np.ndarray,
    is_comb_box: bool = False,
    num_expected_cells: Optional[int] = None
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segments individual character glyphs from a cropped form field.
    Supports both comb-box grid fields and freeform handwritten fields.
    """
    fh, fw = field_bgr.shape[:2]
    gray = cv2.cvtColor(field_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # Comb-box mode: Detect the actual grid boundaries or vertical dividers
    if is_comb_box and num_expected_cells and num_expected_cells > 0:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(fh * 0.4))))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        v_proj = np.sum(v_lines, axis=0)
        line_cols = np.where(v_proj > 0)[0]

        if len(line_cols) >= 2 and (line_cols[-1] - line_cols[0]) > fw * 0.5:
            grid_start = int(line_cols[0])
            grid_end = int(line_cols[-1])
        else:
            pad_est = max(4, int((fw % num_expected_cells) / 2))
            grid_start = pad_est
            grid_end = fw - pad_est

        grid_w = max(10, grid_end - grid_start)
        cell_w = grid_w / num_expected_cells

        glyphs = []
        for i in range(num_expected_cells):
            x1 = max(0, int(round(grid_start + i * cell_w)) + 2)
            x2 = min(fw, int(round(grid_start + (i + 1) * cell_w)) - 2)
            if x2 <= x1:
                x2 = min(fw, x1 + int(cell_w))
            cell_crop = field_bgr[:, x1:x2]
            glyphs.append((cell_crop, (x1, 0, x2 - x1, fh)))
        return glyphs

    # Freeform mode: remove underlines and segment using connected components
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(fw * 0.2)), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    ink_mask = cv2.bitwise_and(binary, cv2.bitwise_not(h_lines))

    border_cut = max(2, int(min(fh, fw) * 0.05))
    ink_mask[:border_cut, :] = 0
    ink_mask[-border_cut:, :] = 0
    ink_mask[:, :border_cut] = 0
    ink_mask[:, -border_cut:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        # Area threshold lowered to 6 to retain small hyphens and punctuation
        if area >= 6 and bh >= 3 and bw >= 2 and bw < fw * 0.85:
            boxes.append((bx, by, bw, bh))

    boxes.sort(key=lambda b: b[0])

    if not boxes:
        k = num_expected_cells or 6
        cell_w = fw / k
        return [(field_bgr[:, int(i*cell_w):int((i+1)*cell_w)], (int(i*cell_w), 0, int(cell_w), fh)) for i in range(k)]

    # Smart Box Merging: Unify split character strokes and dots
    typical_w = float(np.median([b[2] for b in boxes])) if boxes else 15.0
    est_pitch = fw / (num_expected_cells if num_expected_cells else max(len(boxes), 1))

    merged_boxes = []
    curr = list(boxes[0])
    for i in range(1, len(boxes)):
        bx, by, bw, bh = boxes[i]
        cx, cy, cw, ch = curr
        x_overlap = min(cx + cw, bx + bw) - max(cx, bx)
        x_dist = bx - (cx + cw)
        # Merge if horizontally overlapping or tiny gap within character stroke
        if x_overlap > 0 or (x_dist < 6 and (bx + bw - cx) <= max(typical_w * 1.3, est_pitch * 0.8)):
            new_x = min(cx, bx)
            new_y = min(cy, by)
            new_w = max(cx + cw, bx + bw) - new_x
            new_h = max(cy + ch, by + bh) - new_y
            curr = [new_x, new_y, new_w, new_h]
        else:
            merged_boxes.append(tuple(curr))
            curr = list(boxes[i])
    merged_boxes.append(tuple(curr))

    glyphs = []
    for bx, by, bw, bh in merged_boxes:
        pad_x = max(2, int(bw * 0.15))
        pad_y = max(2, int(bh * 0.15))
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(fw, bx + bw + pad_x)
        y2 = min(fh, by + bh + pad_y)
        crop = field_bgr[y1:y2, x1:x2]
        glyphs.append((crop, (bx, by, bw, bh)))

    return glyphs
