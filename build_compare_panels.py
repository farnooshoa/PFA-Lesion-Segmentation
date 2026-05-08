"""Build 3-panel composites: Original | Manual GT | Best Prediction."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "data" / "videos"
GT = ROOT / "data" / "gt_masks_smooth"
PRED = ROOT / "results" / "masks"
OUT = ROOT / "compare_panels"
OUT.mkdir(parents=True, exist_ok=True)

GT_COLOR = np.array([0, 255, 0], dtype=np.float32)      # green
PRED_COLOR = np.array([255, 60, 60], dtype=np.float32)  # red
ALPHA = 0.45
GAP = 8           # px gap between panels
LABEL_H = 36      # px label strip height


def overlay(img_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = img_rgb.astype(np.float32).copy()
    m = mask > 127
    out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def add_label(img: Image.Image, text: str) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_H), (20, 20, 20))
    canvas.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (LABEL_H - th) // 2 - 2), text, fill=(255, 255, 255), font=font)
    return canvas


def build_panel(orig_path: Path, gt_path: Path, pred_path: Path) -> Image.Image:
    orig = np.array(Image.open(orig_path).convert("RGB"))
    gt_m = np.array(Image.open(gt_path).convert("L"))
    pr_m = np.array(Image.open(pred_path).convert("L"))

    panel_orig = Image.fromarray(orig)
    panel_gt = Image.fromarray(overlay(orig, gt_m, GT_COLOR, ALPHA))
    panel_pred = Image.fromarray(overlay(orig, pr_m, PRED_COLOR, ALPHA))

    panel_orig = add_label(panel_orig, "Original")
    panel_gt = add_label(panel_gt, "Manual (GT)")
    panel_pred = add_label(panel_pred, "Best Prediction (Dice 0.878)")

    w, h = panel_orig.size
    total_w = w * 3 + GAP * 2
    canvas = Image.new("RGB", (total_w, h), (20, 20, 20))
    canvas.paste(panel_orig, (0, 0))
    canvas.paste(panel_gt, (w + GAP, 0))
    canvas.paste(panel_pred, (2 * (w + GAP), 0))
    return canvas


def main():
    total = 0
    for seq in sorted(os.listdir(VIDEOS)):
        vdir = VIDEOS / seq
        gdir = GT / seq
        pdir = PRED / seq
        if not (vdir.is_dir() and gdir.is_dir() and pdir.is_dir()):
            continue
        out_dir = OUT / seq
        out_dir.mkdir(exist_ok=True)
        frames = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
        for f in frames:
            stem = f[:-4]
            orig_p = vdir / f
            gt_p = gdir / f"{stem}.png"
            pr_p = pdir / f"{stem}.png"
            if not (gt_p.exists() and pr_p.exists()):
                continue
            panel = build_panel(orig_p, gt_p, pr_p)
            panel.save(out_dir / f"{stem}.png", optimize=True)
            total += 1
        print(f"{seq}: {len(frames)} frames")
    print(f"Done. Wrote {total} panels to {OUT}")


if __name__ == "__main__":
    main()
