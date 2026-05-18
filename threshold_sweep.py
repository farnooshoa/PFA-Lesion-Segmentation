"""
threshold_sweep.py
==================
Runs SAM2 once on Ablation_1, saves raw logits, then sweeps thresholds
from 0.1 to 0.9 to find the Dice-optimal threshold.

Motivated by the calibration analysis which showed:
  - Pixels with predicted prob 0.4-0.5 are actually lesion 78.7% of the time
  - Pixels with predicted prob 0.3-0.4 are actually lesion 54.3% of the time
  → The default threshold of 0.5 is too aggressive and cuts into real lesion

USAGE
-----
    python threshold_sweep.py

OUTPUT
------
    results/threshold_sweep/
        logits.npy           — raw SAM2 logits for all 104 frames (saved once)
        sweep_results.json   — Dice/IoU/Prec/Recall for each threshold
        sweep_plot.png       — Dice vs threshold curve
"""

import os, sys, cv2, json, math, numpy as np, torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM2_REPO = os.environ.get('SAM2_REPO', os.path.expanduser('~/MedSAM2'))
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2_video_predictor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE     = "Ablation_1"
VIDEO_DIR    = os.path.join(REPO_ROOT, "data/videos",          SEQUENCE)
GT_DIR       = os.path.join(REPO_ROOT, "data/gt_masks_smooth", SEQUENCE)
FT_WEIGHTS   = os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_1.pt")
YOLO_WEIGHTS = os.path.join(REPO_ROOT, "checkpoints/yolov8n_pfa.pt")
SAM2_CKPT    = os.environ.get('SAM2_CHECKPOINT',
               os.path.join(SAM2_REPO, 'checkpoints/sam2_hiera_large.pt'))
SAM2_CONFIG  = 'configs/sam2/sam2_hiera_l.yaml'
OUTPUT_DIR   = os.path.join(REPO_ROOT, "results/threshold_sweep")

# Thresholds to sweep (probability values, not logits)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (same as calibration_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def dark_enhance_gamma12(img_rgb):
    gray      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p2, p98   = np.percentile(gray, (2, 98))
    stretched = np.clip((gray-p2)*255/(p98-p2+1e-6), 0, 255).astype(np.uint8)
    out       = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    lut       = np.array([((i/255.0)**1.2)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def strategy_3pos_invtri(x1, y1, x2, y2):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    r = min(bw/2, bh/2) * 0.5
    pos = [[cx + r*math.cos(math.radians(a)),
            cy - r*math.sin(math.radians(a))] for a in [150, 30, 270]]
    mx, my = bw*0.05, bh*0.05
    neg = [[x1+mx,y1+my],[x2-mx,y1+my],[x1+mx,y2-my],[x2-mx,y2-my]]
    return pos, neg


def get_frame_files(video_dir):
    return sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])


def load_gt_mask(gt_dir, fname):
    gt_path = os.path.join(gt_dir, fname.replace('.jpg', '.png'))
    if not os.path.exists(gt_path):
        return None
    m = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    return (np.squeeze(m) > 127).astype(np.uint8)


def compute_metrics(pred, gt):
    gt   = np.squeeze(gt)
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    return {
        'dice'     : float(2*inter / (p.sum()+g.sum()+1e-8)),
        'iou'      : float(inter   / ((p|g).sum()+1e-8)),
        'precision': float(inter   / (p.sum()+1e-8)),
        'recall'   : float(inter   / (g.sum()+1e-8)),
    }


def light_postprocess(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    clean   = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clean)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Run SAM2 and save logits  (only if not already saved)
# ─────────────────────────────────────────────────────────────────────────────

def collect_logits(frame_files):
    """Run SAM2 propagation and save raw logits + GT masks to disk."""
    logits_path = os.path.join(OUTPUT_DIR, "logits.npy")
    gt_path     = os.path.join(OUTPUT_DIR, "gt_masks.npy")

    if os.path.exists(logits_path) and os.path.exists(gt_path):
        print("  Logits already saved — loading from disk (no SAM2 re-run needed)")
        all_logits = np.load(logits_path)
        all_gt     = np.load(gt_path)
        return all_logits, all_gt

    print("  Running SAM2 to collect logits (this takes ~27 min on CPU)...")

    # Preprocess frames
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp(prefix="pfa_thresh_")
    print(f"  Preprocessing → {temp_dir}")
    for fname in frame_files:
        img_bgr = cv2.imread(os.path.join(VIDEO_DIR, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        proc    = dark_enhance_gamma12(img_rgb)
        cv2.imwrite(os.path.join(temp_dir, fname),
                    cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))

    try:
        # Load predictor
        predictor = build_sam2_video_predictor(
            config_file=SAM2_CONFIG, ckpt_path=SAM2_CKPT,
            apply_postprocessing=True,
        )
        ft_state = torch.load(FT_WEIGHTS, map_location='cpu')
        vp_state = predictor.state_dict()
        for k, v in ft_state.items():
            if k in vp_state:
                vp_state[k] = v
        predictor.load_state_dict(vp_state)
        print("  Fine-tuned weights loaded")

        # YOLO bbox
        from ultralytics import YOLO
        yolo  = YOLO(YOLO_WEIGHTS)
        det   = yolo.predict(
            source=os.path.join(VIDEO_DIR, frame_files[0]),
            conf=0.25, device='cpu', verbose=False
        )
        boxes = det[0].boxes
        best  = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().tolist()

        # Init SAM2
        inference_state = predictor.init_state(
            video_path=temp_dir, async_loading_frames=False
        )
        pos, neg = strategy_3pos_invtri(x1, y1, x2, y2)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0, obj_id=1,
            box=np.array([x1,y1,x2,y2], dtype=np.float32),
            points=np.array(pos+neg, dtype=np.float32),
            labels=np.array([1]*3+[0]*4, dtype=np.int32),
        )

        # Collect logits
        raw_logits_dict = {}
        for frame_idx, _, video_res_masks in predictor.propagate_in_video(
            inference_state
        ):
            raw_logits_dict[frame_idx] = video_res_masks[0,0].cpu().numpy()
            print(f"  frame {frame_idx+1}/{len(frame_files)}", end='\r')

        predictor.reset_state(inference_state)
        print()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Stack into arrays  [n_frames, H, W]
    H, W = raw_logits_dict[0].shape
    n    = len(frame_files)
    all_logits = np.zeros((n, H, W), dtype=np.float32)
    all_gt     = np.zeros((n, H, W), dtype=np.uint8)

    for frame_idx in range(n):
        fname   = frame_files[frame_idx]
        gt_mask = load_gt_mask(GT_DIR, fname)
        if gt_mask is not None:
            all_gt[frame_idx] = np.squeeze(gt_mask)
        if frame_idx in raw_logits_dict:
            all_logits[frame_idx] = np.squeeze(raw_logits_dict[frame_idx])

    np.save(logits_path, all_logits)
    np.save(gt_path,     all_gt)
    print(f"  Logits saved → {logits_path}")
    print(f"  GT masks saved → {gt_path}")

    return all_logits, all_gt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Sweep thresholds  (instant, no SAM2 needed)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(all_logits, all_gt, thresholds):
    """
    For each threshold, compute mean Dice/IoU/Prec/Recall across all frames.
    Converts logits → probabilities via sigmoid, then thresholds.
    """
    print(f"\n  Sweeping {len(thresholds)} thresholds...")

    # Convert logits to probabilities once
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))   # sigmoid

    results = []
    n_frames = all_logits.shape[0]

    for thresh in thresholds:
        dices, ious, precs, recs = [], [], [], []

        for i in range(n_frames):
            pred_mask = (all_probs[i] > thresh).astype(np.uint8)
            pred_mask = light_postprocess(pred_mask)
            gt_mask   = all_gt[i]

            if gt_mask.sum() == 0:
                continue

            m = compute_metrics(pred_mask, gt_mask)
            dices.append(m['dice'])
            ious.append(m['iou'])
            precs.append(m['precision'])
            recs.append(m['recall'])

        results.append({
            'threshold': float(thresh),
            'mean_dice'     : round(float(np.mean(dices)), 4),
            'mean_iou'      : round(float(np.mean(ious)),  4),
            'mean_precision': round(float(np.mean(precs)), 4),
            'mean_recall'   : round(float(np.mean(recs)),  4),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Plot
# ─────────────────────────────────────────────────────────────────────────────

def save_sweep_plot(results, optimal_thresh, output_path):
    """Dice vs threshold curve with optimal point marked."""
    W, H   = 700, 450
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    pad    = 70

    plot_w = W - 2*pad
    plot_h = H - 2*pad

    # Grid
    for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = H - pad - int(plot_h * (v - 0.5) / 0.5)
        cv2.line(canvas, (pad, y), (W-pad, y), (220,220,220), 1)
        cv2.putText(canvas, f"{v:.1f}", (5, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)

    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        x = pad + int(plot_w * t)
        cv2.line(canvas, (x, pad), (x, H-pad), (220,220,220), 1)
        cv2.putText(canvas, f"{t:.1f}", (x-10, H-pad+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)

    def to_xy(thresh, dice):
        x = pad + int(plot_w * thresh)
        y = H - pad - int(plot_h * max(0, (dice - 0.5) / 0.5))
        return x, y

    # Draw Dice curve
    pts = [to_xy(r['threshold'], r['mean_dice']) for r in results]
    for j in range(1, len(pts)):
        cv2.line(canvas, pts[j-1], pts[j], (34,139,34), 2)
        cv2.circle(canvas, pts[j], 3, (34,139,34), -1)

    # Mark default threshold (0.5)
    x_def = pad + int(plot_w * 0.5)
    cv2.line(canvas, (x_def, pad), (x_def, H-pad), (180,100,50), 2)
    cv2.putText(canvas, "Default=0.5", (x_def+4, pad+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,100,50), 1)

    # Mark optimal threshold
    x_opt = pad + int(plot_w * optimal_thresh['threshold'])
    cv2.line(canvas, (x_opt, pad), (x_opt, H-pad), (0,0,200), 2)
    cv2.putText(canvas,
                f"Optimal={optimal_thresh['threshold']:.2f}",
                (x_opt+4, pad+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,200), 1)

    # Mark optimal point on curve
    ox, oy = to_xy(optimal_thresh['threshold'], optimal_thresh['mean_dice'])
    cv2.circle(canvas, (ox, oy), 8, (0,0,200), -1)
    cv2.putText(canvas, f"Dice={optimal_thresh['mean_dice']:.4f}",
                (ox+10, oy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,200), 1)

    # Default point on curve
    default_r = next(r for r in results if abs(r['threshold']-0.5) < 0.03)
    dx, dy = to_xy(default_r['threshold'], default_r['mean_dice'])
    cv2.circle(canvas, (dx, dy), 8, (180,100,50), -1)
    cv2.putText(canvas, f"Dice={default_r['mean_dice']:.4f}",
                (dx+10, dy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,100,50), 1)

    # Border and labels
    cv2.rectangle(canvas, (pad,pad), (W-pad,H-pad), (100,100,100), 2)
    cv2.putText(canvas, "Threshold Sweep — Dice vs Probability Threshold (Ablation_1)",
                (pad, pad-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1)
    cv2.putText(canvas, "Probability Threshold",
                (W//2-70, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1)

    cv2.imwrite(output_path, canvas)
    print(f"  Plot → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Threshold Sweep — {SEQUENCE}")
    print(f"  Motivated by calibration: ECE=0.0402, systematic underconfidence")
    print(f"{'='*60}\n")

    frame_files = get_frame_files(VIDEO_DIR)

    # Step 1: get logits
    all_logits, all_gt = collect_logits(frame_files)

    # Step 2: sweep thresholds
    results = sweep_thresholds(all_logits, all_gt, THRESHOLDS)

    # Find optimal
    optimal = max(results, key=lambda r: r['mean_dice'])

    # Find default (closest to 0.5)
    default = next(r for r in results if abs(r['threshold']-0.5) < 0.03)

    # Print table
    print(f"\n{'='*60}")
    print(f"  THRESHOLD SWEEP RESULTS — {SEQUENCE}")
    print(f"{'='*60}")
    print(f"  {'Threshold':>10} {'Dice':>8} {'IoU':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*50}")
    for r in results:
        marker = " ← OPTIMAL" if r['threshold'] == optimal['threshold'] \
            else " ← default" if abs(r['threshold']-0.5) < 0.03 else ""
        print(f"  {r['threshold']:>10.2f} {r['mean_dice']:>8.4f} "
              f"{r['mean_iou']:>8.4f} {r['mean_precision']:>10.4f} "
              f"{r['mean_recall']:>8.4f}{marker}")
    print(f"{'='*60}")

    delta = optimal['mean_dice'] - default['mean_dice']
    print(f"\n  Default threshold (0.5)  → Dice = {default['mean_dice']:.4f}")
    print(f"  Optimal threshold ({optimal['threshold']:.2f}) → Dice = {optimal['mean_dice']:.4f}")
    print(f"  Improvement: +{delta:.4f} Dice points")

    # Save
    sweep_path = os.path.join(OUTPUT_DIR, "sweep_results.json")
    with open(sweep_path, 'w') as f:
        json.dump({
            'sequence': SEQUENCE,
            'default_threshold': 0.5,
            'optimal_threshold': optimal['threshold'],
            'default_dice'     : default['mean_dice'],
            'optimal_dice'     : optimal['mean_dice'],
            'delta_dice'       : round(delta, 4),
            'results'          : results,
        }, f, indent=2)
    print(f"\n  Results → {sweep_path}")

    save_sweep_plot(
        results, optimal,
        os.path.join(OUTPUT_DIR, "sweep_plot.png")
    )

    print(f"\nDone. Results in: {OUTPUT_DIR}/\n")
    print("KEY FINDING:")
    print(f"  Calibration showed SAM2 is systematically underconfident.")
    print(f"  Threshold sweep confirms: lowering threshold from 0.5 to")
    print(f"  {optimal['threshold']:.2f} improves Dice by +{delta:.4f} on {SEQUENCE}.")


if __name__ == "__main__":
    main()
