"""
prompt_strategies.py
====================
Two prompt-point strategies for SAM2 lesion segmentation.

STRATEGY A — Original (geometric, image-blind)
    strategy_3pos_invtri(x1, y1, x2, y2)
    Places 3 positive points at fixed angles (150°, 30°, 270°) inside
    the bounding box.  Does not look at the image at all.

STRATEGY B — New (intensity-guided, image-aware)
    strategy_3pos_intensity(image_gray, x1, y1, x2, y2)
    Finds the 3 brightest local peaks inside the bounding box.
    Works on the preprocessed (M_dark_gamma12) grayscale image so the
    lesion is already enhanced before we pick points.

Both functions return:  (pos_points, neg_points)
    pos_points : list of [x, y]  — 3 positive prompts
    neg_points : list of [x, y]  — 4 negative prompts (identical in both)

USAGE
-----
    from prompt_strategies import strategy_3pos_invtri, strategy_3pos_intensity
    import cv2, numpy as np

    img = cv2.cvtColor(cv2.imread("frame.jpg"), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x1, y1, x2, y2 = 100, 80, 200, 160   # from YOLO

    # Original
    pos_geo, neg = strategy_3pos_invtri(x1, y1, x2, y2)

    # New
    pos_int, neg = strategy_3pos_intensity(gray, x1, y1, x2, y2)
"""

import math
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY A — Original geometric (inverted triangle)
# Copied exactly from training/finetune.py — do not change
# ─────────────────────────────────────────────────────────────────────────────

def strategy_3pos_invtri(x1, y1, x2, y2):
    """
    Original strategy: 3 positive points at fixed geometric positions.

    Points are placed at angles 150°, 30°, 270° on a circle of radius
    0.5 * min(half_width, half_height) centred on the bounding box.
    This forms an inverted triangle:
        top-left  (150°)
        top-right  (30°)
        bottom     (270°)

    No image content is used.
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    r = min(bw / 2, bh / 2) * 0.5

    pos = [
        [cx + r * math.cos(math.radians(a)),
         cy - r * math.sin(math.radians(a))]
        for a in [150, 30, 270]
    ]

    mx, my = bw * 0.05, bh * 0.05
    neg = [
        [x1 + mx, y1 + my],   # top-left corner
        [x2 - mx, y1 + my],   # top-right corner
        [x1 + mx, y2 - my],   # bottom-left corner
        [x2 - mx, y2 - my],   # bottom-right corner
    ]
    return pos, neg


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY B — Intensity-guided (image-aware)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_3pos_intensity(image_gray, x1, y1, x2, y2,
                             n_pos: int = 3,
                             blur_ksize: int = 15,
                             suppress_radius_frac: float = 0.25):
    """
    New strategy: 3 positive points at the brightest local peaks inside
    the bounding box.

    HOW IT WORKS
    ------------
    1. Crop the grayscale image to the bounding box.
    2. Apply Gaussian blur to suppress noise and small artefacts.
    3. Iteratively find the brightest pixel (peak), record it as a
       positive point, then suppress a neighbourhood around it so the
       next peak is forced to be somewhere else.  Repeat n_pos times.
    4. Negative points are identical to the original strategy —
       4 points near the bounding-box corners.

    WHY THIS IS BETTER
    ------------------
    The inverted triangle places points at fixed geometric positions
    regardless of where the lesion actually is.  If the lesion is
    off-centre or has an unusual shape, some points may land outside
    the bright lesion region and confuse SAM2.

    The intensity-guided approach always lands on the brightest
    (most lesion-like) pixels, adapting to each frame individually.

    PARAMETERS
    ----------
    image_gray        : H×W uint8 grayscale image (apply M_dark_gamma12
                        preprocessing first for best results)
    x1,y1,x2,y2      : bounding box in pixel coordinates (from YOLO)
    n_pos             : number of positive points (default 3 to match original)
    blur_ksize        : Gaussian blur kernel size — larger = smoother peak
                        selection, less sensitive to noise  (default 15)
    suppress_radius_frac : neighbourhood to suppress around each found peak,
                        as a fraction of min(bbox_width, bbox_height)
                        (default 0.25 → 25% of the shorter bbox dimension)
    """
    img_h, img_w = image_gray.shape

    # ── Clamp bbox to image boundaries ──
    bx1 = max(0, int(x1))
    by1 = max(0, int(y1))
    bx2 = min(img_w, int(x2))
    by2 = min(img_h, int(y2))
    bw  = bx2 - bx1
    bh  = by2 - by1

    if bw <= 0 or bh <= 0:
        # Degenerate box — fall back to geometric strategy
        return strategy_3pos_invtri(x1, y1, x2, y2)

    # ── Crop and blur ──
    crop = image_gray[by1:by2, bx1:bx2].astype(np.float32)

    # Ensure blur kernel is odd
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(crop, (k, k), 0)

    # ── Iterative peak finding with neighbourhood suppression ──
    suppress_r = int(suppress_radius_frac * min(bw, bh))
    suppress_r = max(suppress_r, 3)   # at least 3px

    search = blurred.copy()
    pos = []

    for _ in range(n_pos):
        # Find current maximum
        _, _, _, max_loc = cv2.minMaxLoc(search)
        local_x, local_y = max_loc   # local_x = col, local_y = row in crop

        # Convert back to full-image coordinates
        global_x = bx1 + local_x
        global_y = by1 + local_y
        pos.append([float(global_x), float(global_y)])

        # Suppress neighbourhood so next peak is elsewhere
        y_lo = max(0, local_y - suppress_r)
        y_hi = min(bh, local_y + suppress_r + 1)
        x_lo = max(0, local_x - suppress_r)
        x_hi = min(bw, local_x + suppress_r + 1)
        search[y_lo:y_hi, x_lo:x_hi] = -1.0

    # ── Negative points — same as original (4 bbox corners) ──
    mx = (bx2 - bx1) * 0.05
    my = (by2 - by1) * 0.05
    neg = [
        [bx1 + mx, by1 + my],
        [bx2 - mx, by1 + my],
        [bx1 + mx, by2 - my],
        [bx2 - mx, by2 - my],
    ]

    return pos, neg


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helper (same as demo.py — apply before calling intensity strat)
# ─────────────────────────────────────────────────────────────────────────────

def dark_enhance_gamma12(img_rgb: np.ndarray) -> np.ndarray:
    """
    M_dark_gamma12 preprocessing from the original pipeline.
    Percentile stretch (2nd–98th) then gamma = 1.2.
    Apply this to the frame BEFORE extracting the grayscale for
    strategy_3pos_intensity — the lesion will be brighter and easier
    to localise.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray - p2) * 255 / (p98 - p2 + 1e-6), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    lut = np.array([((i / 255.0) ** 1.2) * 255 for i in range(256)], dtype=np.uint8)
    processed = cv2.LUT(out, lut)
    return processed


def preprocess_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Convenience: RGB frame → preprocessed grayscale ready for intensity strategy."""
    proc = dark_enhance_gamma12(img_rgb)
    return cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
