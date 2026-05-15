"""
visualize_prompts.py
====================
Side-by-side comparison of the two prompt strategies on real frames.
Runs on CPU only — does NOT require SAM2.

OUTPUT
------
For each sequence, saves one comparison image to:
    prompt_comparison/Ablation_1_prompt_comparison.png
    prompt_comparison/Ablation_2_prompt_comparison.png
    ...

Each image shows THREE columns per frame:
    Left   : Original frame + GEOMETRIC prompts (inverted triangle)
    Middle : Preprocessed frame + INTENSITY prompts (brightest peaks)
    Right  : GT mask overlay so you can see which points are better placed

USAGE
-----
    # From the root of your repo:
    python visualize_prompts.py

    # Choose which frame index to visualise per sequence (default: frame 0):
    python visualize_prompts.py --frame_idx 50

    # Visualise multiple frames at once:
    python visualize_prompts.py --frame_idx 0 25 50 75
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from prompt_strategies import (
    strategy_3pos_invtri,
    strategy_3pos_intensity,
    preprocess_gray,
    dark_enhance_gamma12,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCES    = ["Ablation_1", "Ablation_2", "Ablation_3", "Ablation_4"]
VIDEO_BASE   = "data/videos"
GT_BASE      = "data/gt_masks_smooth"
OUTPUT_DIR   = "prompt_comparison"

# Point drawing settings
POS_COLOR_GEO   = (0,   255,   0)    # green  — geometric positive
POS_COLOR_INT   = (255, 165,   0)    # orange — intensity positive
NEG_COLOR       = (255,   0,   0)    # red    — negative (same for both)
GT_COLOR        = (0,   200, 255)    # cyan   — GT mask outline
POINT_RADIUS    = 8
POINT_THICKNESS = -1   # filled circle


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_bbox_from_mask(mask, padding=0.15):
    """Get bounding box from GT mask (same as finetune.py)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bw, bh = x2 - x1, y2 - y1
    h, w = mask.shape
    return [
        max(0,   int(x1 - bw * padding)),
        max(0,   int(y1 - bh * padding)),
        min(w,   int(x2 + bw * padding)),
        min(h,   int(y2 + bh * padding)),
    ]


def draw_points(img, pos_points, neg_points, pos_color):
    """Draw prompt points on a copy of img."""
    out = img.copy()
    for p in pos_points:
        cv2.circle(out, (int(p[0]), int(p[1])), POINT_RADIUS, pos_color, POINT_THICKNESS)
        cv2.circle(out, (int(p[0]), int(p[1])), POINT_RADIUS, (255,255,255), 2)  # white border
    for p in neg_points:
        cv2.circle(out, (int(p[0]), int(p[1])), POINT_RADIUS, NEG_COLOR, POINT_THICKNESS)
        cv2.circle(out, (int(p[0]), int(p[1])), POINT_RADIUS, (255,255,255), 2)
    return out


def draw_bbox(img, bbox, color=(200, 200, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def overlay_mask_outline(img, mask, color=GT_COLOR, thickness=2):
    """Draw the GT mask as a coloured outline on img."""
    out = img.copy()
    if mask is None:
        return out
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def add_label(img, text, color=(255,255,255)):
    """Add a text label at the top of an image."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out


def add_point_legend(img):
    """Add a small legend at the bottom of the panel."""
    h, w = img.shape[:2]
    legend_h = 50
    canvas = np.zeros((h + legend_h, w, 3), dtype=np.uint8)
    canvas[:h] = img

    y = h + 15
    items = [
        ((0,255,0),   "Geometric +"),
        ((255,165,0), "Intensity +"),
        ((255,0,0),   "Negative  -"),
        ((0,200,255), "GT outline"),
    ]
    x = 10
    for color, label in items:
        cv2.circle(canvas, (x + 8, y + 8), 8, color, -1)
        cv2.putText(canvas, label, (x + 22, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)
        x += 160
    return canvas


def make_comparison_panel(img_rgb, gt_mask, bbox):
    """
    Build a single comparison panel for one frame.

    Returns an image with 3 columns:
        Col 1: Original + geometric prompts
        Col 2: Preprocessed + intensity prompts
        Col 3: Preprocessed + both prompt sets + GT outline
    """
    x1, y1, x2, y2 = bbox

    # ── Preprocessing ──
    proc_rgb  = dark_enhance_gamma12(img_rgb)
    gray_proc = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY)

    # ── Prompt generation ──
    pos_geo, neg = strategy_3pos_invtri(x1, y1, x2, y2)
    pos_int, _   = strategy_3pos_intensity(gray_proc, x1, y1, x2, y2)

    # ── Column 1: original frame + geometric prompts ──
    col1 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    col1 = draw_bbox(col1, bbox)
    col1 = draw_points(col1, pos_geo, neg, POS_COLOR_GEO)
    col1 = overlay_mask_outline(col1, gt_mask)
    col1 = add_label(col1, "GEOMETRIC (original)")

    # ── Column 2: preprocessed frame + intensity prompts ──
    col2 = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
    col2 = draw_bbox(col2, bbox)
    col2 = draw_points(col2, pos_int, neg, POS_COLOR_INT)
    col2 = overlay_mask_outline(col2, gt_mask)
    col2 = add_label(col2, "INTENSITY-GUIDED (new)", color=(255,165,0))

    # ── Column 3: preprocessed + both strategies overlaid ──
    col3 = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
    col3 = draw_bbox(col3, bbox)
    col3 = draw_points(col3, pos_geo, [], POS_COLOR_GEO)   # geometric (no neg)
    col3 = draw_points(col3, pos_int, neg, POS_COLOR_INT)  # intensity + neg
    col3 = overlay_mask_outline(col3, gt_mask)
    col3 = add_label(col3, "COMPARISON (green=geo, orange=int)")

    panel = np.hstack([col1, col2, col3])
    panel = add_point_legend(panel)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise geometric vs intensity-guided prompt strategies"
    )
    parser.add_argument("--frame_idx", type=int, nargs="+", default=[0, 25, 50],
                        help="Frame indices to visualise per sequence (default: 0 25 50)")
    parser.add_argument("--videos_dir", default=VIDEO_BASE)
    parser.add_argument("--gt_dir",     default=GT_BASE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Prompt Strategy Comparison Visualisation")
    print(f"{'='*60}")

    for seq in SEQUENCES:
        vid_dir = Path(args.videos_dir) / seq
        gt_dir  = Path(args.gt_dir)     / seq

        if not vid_dir.exists():
            print(f"  [SKIP] {seq} — not found at {vid_dir}")
            continue

        frames = sorted(vid_dir.glob("*.jpg"))
        if not frames:
            print(f"  [SKIP] {seq} — no .jpg files")
            continue

        print(f"\n  [{seq}]  {len(frames)} frames total")

        panels = []
        for idx in args.frame_idx:
            if idx >= len(frames):
                print(f"    [SKIP] frame {idx} — only {len(frames)} frames")
                continue

            frame_path = frames[idx]
            # Find matching GT mask by stem
            gt_path = gt_dir / (frame_path.stem + ".png")
            if not gt_path.exists():
                # Try 5-digit zero-padded index
                gt_path = gt_dir / f"{idx:05d}.png"
            if not gt_path.exists():
                print(f"    [WARN] No GT mask for frame {idx}")
                gt_mask = None
            else:
                gt_raw  = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_raw > 127).astype(np.uint8)

            img_bgr = cv2.imread(str(frame_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Get bbox from GT mask (same as finetune.py uses during training)
            if gt_mask is not None:
                bbox = get_bbox_from_mask(gt_mask)
            else:
                bbox = None

            if bbox is None:
                print(f"    [SKIP] frame {idx} — empty mask / no bbox")
                continue

            panel = make_comparison_panel(img_rgb, gt_mask, bbox)
            panels.append((idx, panel))
            print(f"    frame {idx:3d} — bbox {bbox} — panel built ✓")

        if not panels:
            continue

        # Stack all frames for this sequence vertically
        combined = np.vstack([p for _, p in panels])

        out_path = out_dir / f"{seq}_prompt_comparison.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved → {out_path}")

    print(f"\nAll visualisations saved to: {out_dir}/")
    print("\nWhat to look for:")
    print("  GREEN dots  = geometric strategy (fixed triangle — image-blind)")
    print("  ORANGE dots = intensity strategy  (brightest peaks — image-aware)")
    print("  CYAN line   = GT mask boundary")
    print("  A good prompt point sits INSIDE the cyan boundary.")
    print()


if __name__ == "__main__":
    main()
