"""
eval_prompt_strategies.py
=========================
Compares two prompt strategies on Ablation_3 using the fine-tuned SAM2
weights (ft_Ablation_3.pt — trained with Ablation_3 as the held-out fold).

STRATEGIES COMPARED
-------------------
  A) Geometric    : 3 positive points at fixed angles (original method)
  B) Intensity    : 3 positive points at brightest local peaks (new method)

Both use the same 4 negative corner points and the same fine-tuned weights.
This isolates the effect of the prompt strategy alone.

USAGE
-----
    python eval_prompt_strategies.py

OUTPUT
------
    results/prompt_eval/
        summary.json          — mean Dice/IoU/Precision/Recall for each strategy
        per_frame.json        — per-frame metrics for both strategies
        comparison_plot.png   — Dice per frame, both strategies overlaid
"""

import os, sys, cv2, json, math, numpy as np, torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM2_REPO = os.environ.get('SAM2_REPO', os.path.expanduser('~/MedSAM2'))
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)
sys.path.insert(0, REPO_ROOT)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE     = "Ablation_3"
VIDEO_DIR    = os.path.join(REPO_ROOT, "data/videos",        SEQUENCE)
GT_DIR       = os.path.join(REPO_ROOT, "data/gt_masks_smooth", SEQUENCE)
FT_WEIGHTS   = os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_3.pt")
SAM2_CKPT    = os.environ.get('SAM2_CHECKPOINT',
               os.path.join(SAM2_REPO, 'checkpoints/sam2_hiera_large.pt'))
SAM2_CONFIG  = 'configs/sam2/sam2_hiera_l.yaml'
OUTPUT_DIR   = os.path.join(REPO_ROOT, "results/prompt_eval")
CONF         = 0.25
device       = torch.device('cpu')   # change to 'cuda' if available

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT STRATEGIES  (copied exactly from finetune.py / prompt_strategies.py)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_3pos_invtri(x1, y1, x2, y2):
    """Original geometric strategy — 3 fixed-angle points."""
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    r = min(bw/2, bh/2) * 0.5
    pos = [[cx + r*math.cos(math.radians(a)),
            cy - r*math.sin(math.radians(a))] for a in [150, 30, 270]]
    mx, my = bw*0.05, bh*0.05
    neg = [[x1+mx,y1+my],[x2-mx,y1+my],[x1+mx,y2-my],[x2-mx,y2-my]]
    return pos, neg

def strategy_3pos_midpoint_neg(x1, y1, x2, y2):
    """Improved geometric strategy — negatives at edge midpoints."""
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    r = min(bw/2, bh/2) * 0.5
    pos = [[cx + r*math.cos(math.radians(a)),
            cy - r*math.sin(math.radians(a))] for a in [150, 30, 270]]
    neg = [
        [cx, y1],   # top
        [cx, y2],   # bottom
        [x1, cy],   # left
        [x2, cy],   # right
    ]
    return pos, neg

def strategy_3pos_intensity(image_gray, x1, y1, x2, y2,
                             n_pos=3, blur_ksize=15,
                             suppress_radius_frac=0.25):
    """New intensity-guided strategy — 3 brightest local peaks."""
    img_h, img_w = image_gray.shape
    bx1 = max(0, int(x1));  by1 = max(0, int(y1))
    bx2 = min(img_w, int(x2)); by2 = min(img_h, int(y2))
    bw  = bx2 - bx1;  bh = by2 - by1

    if bw <= 0 or bh <= 0:
        return strategy_3pos_invtri(x1, y1, x2, y2)

    crop = image_gray[by1:by2, bx1:bx2].astype(np.float32)
    k    = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(crop, (k, k), 0)

    suppress_r = max(int(suppress_radius_frac * min(bw, bh)), 3)
    search = blurred.copy()
    pos = []

    for _ in range(n_pos):
        _, _, _, max_loc = cv2.minMaxLoc(search)
        lx, ly = max_loc
        pos.append([float(bx1 + lx), float(by1 + ly)])
        y_lo = max(0, ly - suppress_r);  y_hi = min(bh, ly + suppress_r + 1)
        x_lo = max(0, lx - suppress_r);  x_hi = min(bw, lx + suppress_r + 1)
        search[y_lo:y_hi, x_lo:x_hi] = -1.0

    mx = (bx2-bx1)*0.05;  my = (by2-by1)*0.05
    neg = [[bx1+mx,by1+my],[bx2-mx,by1+my],[bx1+mx,by2-my],[bx2-mx,by2-my]]
    return pos, neg


def dark_enhance_gamma12(img_rgb):
    """M_dark_gamma12 preprocessing — same as pipeline."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray-p2)*255/(p98-p2+1e-6), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    lut = np.array([((i/255.0)**1.2)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def get_bbox_from_mask(mask, padding=0.15):
    """Derive bounding box from GT mask — same as finetune.py."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bw, bh = x2-x1, y2-y1
    h, w = mask.shape
    return [max(0,int(x1-bw*padding)), max(0,int(y1-bh*padding)),
            min(w,int(x2+bw*padding)), min(h,int(y2+bh*padding))]


def compute_metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    return {
        'dice'     : float(2*inter / (p.sum()+g.sum()+1e-8)),
        'iou'      : float(inter / ((p|g).sum()+1e-8)),
        'precision': float(inter / (p.sum()+1e-8)),
        'recall'   : float(inter / (g.sum()+1e-8)),
    }


def light_postprocess(mask):
    """Keep largest connected component — same as finetune.py."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    clean = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clean)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print("Loading SAM2 + fine-tuned weights...")
    model = build_sam2(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CKPT,
        apply_postprocessing=True,
    ).to(device)

    # Load fine-tuned mask decoder weights
    ft_state = torch.load(FT_WEIGHTS, map_location='cpu')
    sd = model.state_dict()
    loaded = 0
    for k, v in ft_state.items():
        if k in sd:
            sd[k] = v
            loaded += 1
    model.load_state_dict(sd)
    print(f"  Fine-tuned weights loaded ({loaded} tensors from {FT_WEIGHTS})")

    predictor = SAM2ImagePredictor(model)
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE STRATEGY ON ALL FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(predictor, strategy_name, frames):
    """
    Run SAM2 inference on all frames using the given strategy.
    Returns list of per-frame metric dicts.
    """
    results = []
    total   = len(frames)

    for i, (img_path, gt_path) in enumerate(frames):
        # Load image and GT
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt      = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        # Get bbox from GT mask (same as finetune.py — fair comparison)
        bbox = get_bbox_from_mask(gt)
        if bbox is None:
            print(f"  [SKIP] frame {i} — empty GT mask")
            continue

        x1, y1, x2, y2 = bbox

        # Preprocessing
        proc_rgb  = dark_enhance_gamma12(img_rgb)
        gray_proc = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY)

        # Generate prompt points
        if strategy_name == "midpoint":
           pos, neg = strategy_3pos_midpoint_neg(x1, y1, x2, y2)
        else:
           pos, neg = strategy_3pos_invtri(x1, y1, x2, y2)

        points = np.array(pos + neg, dtype=np.float32)
        labels = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
        box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

        # SAM2 inference
        predictor.set_image(proc_rgb)
        with torch.no_grad():
            masks, _, _ = predictor.predict(
                point_coords  = points,
                point_labels  = labels,
                box           = box_np,
                multimask_output = False,
            )

        pred = light_postprocess(masks[0].astype(np.uint8))
        m    = compute_metrics(pred, gt)
        m['frame'] = i
        results.append(m)

        # Progress
        print(f"  [{strategy_name}] frame {i+1:3d}/{total} — "
              f"Dice={m['dice']:.3f}  IoU={m['iou']:.3f}", end='\r')

    print()  # newline after progress
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(geo_results, int_results, output_path):
    """Save a simple Dice-per-frame comparison plot using only opencv."""
    n      = max(len(geo_results), len(int_results))
    W, H   = max(900, n*6), 400
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 240

    # Axes
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 50
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    # Draw gridlines at 0.2 intervals
    for dice_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        y = pad_t + int(plot_h * (1 - dice_val))
        cv2.line(canvas, (pad_l, y), (W-pad_r, y), (200,200,200), 1)
        cv2.putText(canvas, f"{dice_val:.1f}", (5, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

    def dice_to_y(d):
        return pad_t + int(plot_h * (1 - d))

    def frame_to_x(i, total):
        return pad_l + int(plot_w * i / max(total-1, 1))

    # Draw lines
    for results, color, label in [
        (geo_results, (34,139,34),   "Geometric (original)"),
        (int_results, (255,140,0),   "Intensity-guided (new)"),
    ]:
        pts = [(frame_to_x(r['frame'], n), dice_to_y(r['dice'])) for r in results]
        for j in range(1, len(pts)):
            cv2.line(canvas, pts[j-1], pts[j], color, 2)

    # Legend
    cv2.circle(canvas, (pad_l+10, 15), 5, (34,139,34),  -1)
    cv2.putText(canvas, "Geometric (original)",   (pad_l+20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34,139,34),  1)
    cv2.circle(canvas, (pad_l+200, 15), 5, (255,140,0), -1)
    cv2.putText(canvas, "Intensity-guided (new)", (pad_l+210, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,140,0), 1)

    # Axis labels
    cv2.putText(canvas, "Frame",
                (W//2-20, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
    cv2.putText(canvas, "Dice",
                (5, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)

    cv2.imwrite(output_path, canvas)
    print(f"  Plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Prompt Strategy Evaluation — {SEQUENCE}")
    print(f"{'='*60}\n")

    # Collect frame/GT pairs
    frame_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.jpg')])
    frames = []
    for fname in frame_files:
        gt_path = os.path.join(GT_DIR, fname.replace('.jpg', '.png'))
        if os.path.exists(gt_path):
            frames.append((os.path.join(VIDEO_DIR, fname), gt_path))

    print(f"  Found {len(frames)} frame/GT pairs in {SEQUENCE}\n")

    # Load model once — reused for both strategies
    predictor = load_model()
    print()

    # ── Run geometric strategy ──
    print(f"  Running GEOMETRIC strategy ({len(frames)} frames)...")
    geo_results = run_strategy(predictor, "geometric", frames)

    # ── Run intensity strategy ──
    print(f"\n  Running INTENSITY strategy ({len(frames)} frames)...")
    int_results = run_strategy(predictor, "midpoint", frames)

    # ── Compute summaries ──
    def mean(lst, key):
        return float(np.mean([r[key] for r in lst]))

    summary = {
        "sequence": SEQUENCE,
        "n_frames": len(frames),
        "geometric": {
            "mean_dice"     : round(mean(geo_results, 'dice'),      4),
            "mean_iou"      : round(mean(geo_results, 'iou'),       4),
            "mean_precision": round(mean(geo_results, 'precision'), 4),
            "mean_recall"   : round(mean(geo_results, 'recall'),    4),
        },
        "midpoint": {
            "mean_dice"     : round(mean(int_results, 'dice'),      4),
            "mean_iou"      : round(mean(int_results, 'iou'),       4),
            "mean_precision": round(mean(int_results, 'precision'), 4),
            "mean_recall"   : round(mean(int_results, 'recall'),    4),
        },
    }

    # Delta
    delta_dice = summary['midpoint']['mean_dice'] - summary['geometric']['mean_dice']
    summary['delta_dice'] = round(delta_dice, 4)

    # ── Print results table ──
    print(f"\n{'='*60}")
    print(f"  RESULTS — {SEQUENCE}")
    print(f"{'='*60}")
    print(f"  {'Metric':<14} {'Geometric':>12} {'midpoint':>12} {'Delta':>10}")
    print(f"  {'-'*50}")
    for key in ['mean_dice', 'mean_iou', 'mean_precision', 'mean_recall']:
        label  = key.replace('mean_', '').capitalize()
        g_val  = summary['geometric'][key]
        i_val  = summary['midpoint'][key]
        delta  = i_val - g_val
        sign   = '▲' if delta >= 0 else '▼'
        print(f"  {label:<14} {g_val:>12.4f} {i_val:>12.4f} {sign}{abs(delta):>8.4f}")
    print(f"{'='*60}\n")

    if delta_dice > 0:
        print(f"  ✓ Midpoint strategy is BETTER by {delta_dice:.4f} Dice points")
    elif delta_dice < 0:
        print(f"  ✗ Geometric strategy is BETTER by {abs(delta_dice):.4f} Dice points")
    else:
        print(f"  = Strategies are equal on this sequence")

    # ── Save results ──
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")

    per_frame = {"geometric": geo_results, "midpoint": int_results}
    per_frame_path = os.path.join(OUTPUT_DIR, "per_frame.json")
    with open(per_frame_path, 'w') as f:
        json.dump(per_frame, f, indent=2)
    print(f"  Per-frame saved → {per_frame_path}")

    # ── Plot ──
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
    save_plot(geo_results, int_results, plot_path)

    print(f"\nDone. All results in: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
