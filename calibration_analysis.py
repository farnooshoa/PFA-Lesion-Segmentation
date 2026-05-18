"""
calibration_analysis.py
========================
Calibration analysis of SAM2's internal confidence on Ablation_1.

WHAT IS CALIBRATION?
--------------------
A model is well-calibrated if its predicted probability matches the
actual likelihood of being correct.

Example:
  SAM2 predicts probability = 0.8 for a pixel being lesion
  → that pixel should actually BE lesion 80% of the time

If SAM2 says 0.8 but the pixel is only lesion 50% of the time
→ SAM2 is OVERCONFIDENT (its high scores are not trustworthy)

If SAM2 says 0.8 but the pixel is lesion 95% of the time
→ SAM2 is UNDERCONFIDENT (it is being too cautious)

HOW WE MEASURE IT
-----------------
1. Run SAM2 inference on all 104 frames of Ablation_1
2. For every pixel in every frame:
   - Record SAM2's predicted probability (sigmoid of logit)
   - Record the GT label (1=lesion, 0=background)
3. Group pixels into 10 bins by predicted probability:
   [0.0-0.1], [0.1-0.2], ..., [0.9-1.0]
4. For each bin: compute fraction of pixels that are actually lesion
5. Plot predicted probability vs actual fraction (reliability diagram)
   - Perfect calibration = diagonal line
   - Above diagonal = underconfident
   - Below diagonal = overconfident
6. Compute Expected Calibration Error (ECE) — single number summary

WHY THIS MATTERS
----------------
- Clinicians need to trust uncertainty estimates near lesion boundaries
- Regulatory bodies (FDA, CE) increasingly require uncertainty quantification
- Nobody has published a calibration analysis of SAM2 on cardiac ultrasound
- Result tells you: can we trust SAM2's soft output for clinical use?

USAGE
-----
    python calibration_analysis.py

OUTPUT
------
    results/calibration/
        calibration_summary.json    — ECE, per-bin statistics
        reliability_diagram.png     — the calibration plot
        confidence_histogram.png    — distribution of SAM2 confidence values
        soft_mask_examples/         — example frames showing soft probability maps
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
OUTPUT_DIR   = os.path.join(REPO_ROOT, "results/calibration")
N_BINS       = 10     # number of calibration bins

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "soft_mask_examples"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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


def load_predictor():
    print("Loading SAM2 video predictor + fine-tuned weights...")
    predictor = build_sam2_video_predictor(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CKPT,
        apply_postprocessing=True,
    )
    ft_state = torch.load(FT_WEIGHTS, map_location='cpu')
    vp_state = predictor.state_dict()
    loaded = 0
    for k, v in ft_state.items():
        if k in vp_state:
            vp_state[k] = v
            loaded += 1
    predictor.load_state_dict(vp_state)
    print(f"  Fine-tuned weights loaded ({loaded} tensors)")
    return predictor


def get_yolo_bbox(frame_path):
    from ultralytics import YOLO
    yolo  = YOLO(YOLO_WEIGHTS)
    det   = yolo.predict(source=frame_path, conf=0.25, device='cpu', verbose=False)
    boxes = det[0].boxes
    if len(boxes) == 0:
        return None
    best = boxes.conf.argmax()
    return boxes.xyxy[best].cpu().numpy().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SOFT MASK VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def prob_to_heatmap(prob_map):
    """
    Convert a probability map [0,1] to a colour heatmap.

    Colour scale:
        0.0 → dark blue   (background, very confident)
        0.5 → yellow      (uncertain boundary)
        1.0 → bright red  (lesion, very confident)

    This uses the JET colormap: blue=low, green=mid, red=high.
    We remap so that:
        prob < 0.5 → shown as blue/green (background side)
        prob > 0.5 → shown as yellow/red (lesion side)
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    return heatmap


def save_soft_mask_example(img_rgb, prob_map, gt_mask, frame_idx, output_dir):
    """
    Save a 4-panel example image showing:
        1. Original frame
        2. GT mask overlay
        3. Hard binary mask (threshold=0.5)
        4. Soft probability heatmap
    """
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W      = img_bgr.shape[:2]

    # Panel 1: original
    panel1 = img_bgr.copy()
    cv2.putText(panel1, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Panel 2: GT overlay
    panel2 = img_bgr.copy()
    if gt_mask is not None:
        roi = gt_mask > 0
        panel2[roi] = (panel2[roi] * 0.5 + np.array([0,200,0]) * 0.5).astype(np.uint8)
    cv2.putText(panel2, "GT Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Panel 3: hard binary mask
    hard_mask = (prob_map > 0.5).astype(np.uint8)
    panel3    = img_bgr.copy()
    roi = hard_mask > 0
    panel3[roi] = (panel3[roi] * 0.5 + np.array([0,0,200]) * 0.5).astype(np.uint8)
    cv2.putText(panel3, "Hard Mask (threshold=0.5)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Panel 4: soft probability heatmap blended with original
    heatmap = prob_to_heatmap(prob_map)
    panel4  = cv2.addWeighted(img_bgr, 0.4, heatmap, 0.6, 0)
    cv2.putText(panel4, "Soft Probability Map", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Add colorbar label to panel 4
    cv2.putText(panel4, "Blue=background  Yellow=uncertain  Red=lesion",
                (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # Combine into one image
    top    = np.hstack([panel1, panel2])
    bottom = np.hstack([panel3, panel4])
    combined = np.vstack([top, bottom])

    out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(out_path, combined)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(all_probs, all_labels, n_bins=10):
    """
    Compute calibration statistics.

    Parameters
    ----------
    all_probs  : 1D numpy array of predicted probabilities [0,1]
    all_labels : 1D numpy array of GT labels {0,1}
    n_bins     : number of equal-width bins

    Returns
    -------
    bins : list of dicts with keys:
        bin_lower, bin_upper, mean_pred_prob, fraction_positive,
        n_samples, calibration_error
    ece  : Expected Calibration Error (weighted average of bin errors)
    """
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i+1]

        # Find pixels in this bin
        in_bin = (all_probs >= lower) & (all_probs < upper)
        # Include upper edge in last bin
        if i == n_bins - 1:
            in_bin = (all_probs >= lower) & (all_probs <= upper)

        n = in_bin.sum()
        if n == 0:
            bins.append({
                'bin_lower'        : round(float(lower), 2),
                'bin_upper'        : round(float(upper), 2),
                'mean_pred_prob'   : round(float((lower+upper)/2), 3),
                'fraction_positive': None,
                'n_samples'        : 0,
                'calibration_error': None,
            })
            continue

        mean_pred  = float(all_probs[in_bin].mean())
        frac_pos   = float(all_labels[in_bin].mean())
        cal_error  = abs(mean_pred - frac_pos)

        bins.append({
            'bin_lower'        : round(float(lower), 2),
            'bin_upper'        : round(float(upper), 2),
            'mean_pred_prob'   : round(mean_pred, 4),
            'fraction_positive': round(frac_pos, 4),
            'n_samples'        : int(n),
            'calibration_error': round(cal_error, 4),
        })

    # ECE = weighted average of calibration errors
    total = len(all_probs)
    ece   = sum(
        b['calibration_error'] * b['n_samples'] / total
        for b in bins
        if b['calibration_error'] is not None
    )

    return bins, float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS (opencv only — no matplotlib needed)
# ─────────────────────────────────────────────────────────────────────────────

def save_reliability_diagram(bins, ece, output_path):
    """
    Draw the reliability diagram (predicted prob vs actual fraction).
    Perfect calibration = diagonal line.
    """
    W, H   = 600, 600
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    pad    = 80
    plot_w = W - 2*pad
    plot_h = H - 2*pad

    # Background
    cv2.rectangle(canvas, (pad, pad), (W-pad, H-pad), (245,245,245), -1)

    # Perfect calibration diagonal (dashed)
    for i in range(10):
        x1 = pad + int(plot_w * i / 10)
        y1 = H - pad - int(plot_h * i / 10)
        x2 = pad + int(plot_w * (i+1) / 10)
        y2 = H - pad - int(plot_h * (i+1) / 10)
        cv2.line(canvas, (x1,y1), (x2,y2), (180,180,180), 1)

    cv2.putText(canvas, "Perfect calibration",
                (pad+5, H-pad - plot_h//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    # Grid lines
    for v in [0.2, 0.4, 0.6, 0.8]:
        x = pad + int(plot_w * v)
        y = H - pad - int(plot_h * v)
        cv2.line(canvas, (x, pad), (x, H-pad), (220,220,220), 1)
        cv2.line(canvas, (pad, y), (W-pad, y), (220,220,220), 1)

    # Axis labels
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = pad + int(plot_w * v)
        y = H - pad - int(plot_h * v)
        # x axis
        cv2.putText(canvas, f"{v:.1f}", (x-10, H-pad+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)
        # y axis
        cv2.putText(canvas, f"{v:.1f}", (pad-40, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)

    # Plot calibration bars (bars showing gap from diagonal)
    valid_bins = [b for b in bins if b['fraction_positive'] is not None]
    bar_w      = max(4, int(plot_w / len(bins)) - 4)

    for b in valid_bins:
        x_center = pad + int(plot_w * b['mean_pred_prob'])
        y_actual  = H - pad - int(plot_h * b['fraction_positive'])
        y_perfect = H - pad - int(plot_h * b['mean_pred_prob'])

        # Bar from perfect line to actual (gap = calibration error)
        color = (200, 100, 50) if b['fraction_positive'] < b['mean_pred_prob'] \
                else (50, 150, 200)

        top_y    = min(y_actual, y_perfect)
        bottom_y = max(y_actual, y_perfect)
        if bottom_y > top_y:
            cv2.rectangle(canvas,
                          (x_center - bar_w//2, top_y),
                          (x_center + bar_w//2, bottom_y),
                          color, -1)

        # Dot at actual fraction
        cv2.circle(canvas, (x_center, y_actual), 5, (0, 0, 180), -1)

    # Axes borders
    cv2.rectangle(canvas, (pad, pad), (W-pad, H-pad), (100,100,100), 2)

    # Title and ECE
    cv2.putText(canvas, "Reliability Diagram — SAM2 on Ablation_1",
                (pad, pad-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1)
    cv2.putText(canvas, f"ECE = {ece:.4f}",
                (pad, pad-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,0,180), 2)

    # Axis titles
    cv2.putText(canvas, "Predicted Probability",
                (W//2 - 80, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)

    # Legend
    cv2.rectangle(canvas, (W-pad-120, pad+10), (W-pad-10, pad+55), (240,240,240), -1)
    cv2.rectangle(canvas, (W-pad-15, pad+20), (W-pad-5, pad+30), (200,100,50), -1)
    cv2.putText(canvas, "Overconfident", (W-pad-110, pad+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1)
    cv2.rectangle(canvas, (W-pad-15, pad+38), (W-pad-5, pad+48), (50,150,200), -1)
    cv2.putText(canvas, "Underconfident", (W-pad-110, pad+46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1)

    cv2.imwrite(output_path, canvas)
    print(f"  Reliability diagram → {output_path}")


def save_confidence_histogram(all_probs, output_path, n_bins=50):
    """
    Distribution of SAM2 confidence values across all pixels.
    Shows how bimodal (confident) or spread out (uncertain) SAM2 is.
    """
    W, H   = 700, 400
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    pad    = 60

    hist, edges = np.histogram(all_probs, bins=n_bins, range=(0,1))
    max_count   = hist.max()
    plot_w      = W - 2*pad
    plot_h      = H - 2*pad
    bar_w       = max(1, plot_w // n_bins - 1)

    for i, count in enumerate(hist):
        x = pad + int(plot_w * i / n_bins)
        bar_h = int(plot_h * count / max_count)
        # Colour by confidence level
        prob_mid = (edges[i] + edges[i+1]) / 2
        if prob_mid < 0.3:
            color = (200, 100, 50)    # low conf — orange
        elif prob_mid > 0.7:
            color = (50, 150, 50)     # high conf — green
        else:
            color = (50, 150, 220)    # uncertain — blue
        cv2.rectangle(canvas,
                      (x, H-pad-bar_h),
                      (x+bar_w, H-pad),
                      color, -1)

    cv2.rectangle(canvas, (pad, pad), (W-pad, H-pad), (100,100,100), 2)

    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = pad + int(plot_w * v)
        cv2.putText(canvas, f"{v:.1f}", (x-10, H-pad+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)
        cv2.line(canvas, (x, pad), (x, H-pad), (220,220,220), 1)

    cv2.putText(canvas, "Distribution of SAM2 Predicted Probabilities — Ablation_1",
                (pad, pad-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1)
    cv2.putText(canvas, "Predicted Probability",
                (W//2-70, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1)

    cv2.imwrite(output_path, canvas)
    print(f"  Confidence histogram → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  SAM2 Calibration Analysis — {SEQUENCE}")
    print(f"{'='*60}\n")

    frame_files = get_frame_files(VIDEO_DIR)
    print(f"  Found {len(frame_files)} frames\n")

    # ── Preprocess frames ──
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp(prefix="pfa_calib_")
    print(f"  Preprocessing frames → {temp_dir}")
    for fname in frame_files:
        img_bgr = cv2.imread(os.path.join(VIDEO_DIR, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        proc    = dark_enhance_gamma12(img_rgb)
        cv2.imwrite(os.path.join(temp_dir, fname),
                    cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))

    try:
        # ── Load model ──
        predictor = load_predictor()

        # ── Get YOLO bbox ──
        print("\n  Getting YOLO bbox from first frame...")
        bbox = get_yolo_bbox(os.path.join(VIDEO_DIR, frame_files[0]))
        if bbox is None:
            print("  ERROR: YOLO detected no lesion.")
            return
        print(f"  YOLO bbox: {[round(v,1) for v in bbox]}")

        # ── Initialise SAM2 ──
        inference_state = predictor.init_state(
            video_path=temp_dir,
            async_loading_frames=False,
        )

        x1, y1, x2, y2 = bbox
        pos, neg = strategy_3pos_invtri(x1, y1, x2, y2)
        points   = np.array(pos+neg, dtype=np.float32)
        labels   = np.array([1]*3+[0]*4, dtype=np.int32)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0, obj_id=1,
            box=np.array([x1,y1,x2,y2], dtype=np.float32),
            points=points, labels=labels,
        )

        # ── Collect logits from all frames ──
        print("\n  Running SAM2 propagation to collect logits...")
        all_logits = {}

        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            inference_state
        ):
            # video_res_masks shape: (n_objects, 1, H, W) — raw logits
            logits = video_res_masks[0, 0].cpu().numpy()   # (H, W)
            all_logits[frame_idx] = logits
            print(f"  frame {frame_idx+1:3d}/{len(frame_files)}", end='\r')

        print(f"\n  Collected logits for {len(all_logits)} frames")
        predictor.reset_state(inference_state)

        # ── Convert logits to probabilities and collect pixel data ──
        print("\n  Computing calibration statistics...")
        all_probs_list  = []
        all_labels_list = []

        # Also track per-frame Dice for reference
        per_frame_results = []

        # Save soft mask examples for 5 evenly spaced frames
        example_frames = [0, 25, 50, 75, 103]

        for frame_idx, logits in sorted(all_logits.items()):
            fname   = frame_files[frame_idx]
            gt_mask = load_gt_mask(GT_DIR, fname)
            if gt_mask is None:
                continue

            # Sigmoid to get probability map
            prob_map = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
            prob_map = np.squeeze(prob_map)
            gt_flat  = np.squeeze(gt_mask)

            # Collect all pixels
            all_probs_list.append(prob_map.flatten())
            all_labels_list.append(gt_flat.flatten())

            # Per-frame Dice (hard threshold at 0.5)
            hard_mask = (prob_map > 0.5).astype(np.uint8)
            p, g      = hard_mask.astype(bool), gt_flat.astype(bool)
            inter     = (p & g).sum()
            dice      = float(2*inter / (p.sum()+g.sum()+1e-8))
            per_frame_results.append({
                'frame'         : frame_idx,
                'dice'          : round(dice, 4),
                'mean_prob_lesion'    : round(float(prob_map[gt_flat==1].mean()), 4)
                                        if gt_flat.sum() > 0 else None,
                'mean_prob_background': round(float(prob_map[gt_flat==0].mean()), 4),
            })

            # Save soft mask examples
            if frame_idx in example_frames:
                img_bgr = cv2.imread(os.path.join(VIDEO_DIR, fname))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                out_path = save_soft_mask_example(
                    img_rgb, prob_map, gt_flat,
                    frame_idx,
                    os.path.join(OUTPUT_DIR, "soft_mask_examples")
                )
                print(f"  Soft mask example → {out_path}")

        # Concatenate all pixel data
        all_probs  = np.concatenate(all_probs_list)
        all_labels = np.concatenate(all_labels_list)

        print(f"\n  Total pixels analysed: {len(all_probs):,}")
        print(f"  Lesion pixels: {all_labels.sum():,} "
              f"({100*all_labels.mean():.1f}%)")
        print(f"  Background pixels: {(1-all_labels).sum():,} "
              f"({100*(1-all_labels).mean():.1f}%)")

        # ── Compute calibration ──
        bins, ece = compute_calibration(all_probs, all_labels, N_BINS)

        # ── Print calibration table ──
        print(f"\n{'='*60}")
        print(f"  CALIBRATION RESULTS — {SEQUENCE}")
        print(f"  Expected Calibration Error (ECE) = {ece:.4f}")
        print(f"{'='*60}")
        print(f"  {'Bin':<14} {'Pred Prob':>10} {'Actual Frac':>12} "
              f"{'Error':>8} {'N pixels':>10}")
        print(f"  {'-'*56}")
        for b in bins:
            if b['fraction_positive'] is None:
                print(f"  [{b['bin_lower']:.1f}-{b['bin_upper']:.1f}]"
                      f"{'(empty)':>40}")
                continue
            direction = "↑under" if b['fraction_positive'] > b['mean_pred_prob'] \
                        else "↓over"
            print(
                f"  [{b['bin_lower']:.1f}-{b['bin_upper']:.1f}]"
                f"  {b['mean_pred_prob']:>10.4f}"
                f"  {b['fraction_positive']:>12.4f}"
                f"  {b['calibration_error']:>8.4f}"
                f"  {b['n_samples']:>10,}"
                f"  {direction}"
            )
        print(f"{'='*60}")

        # Interpretation
        print(f"\n  ECE = {ece:.4f}")
        if ece < 0.05:
            print("  → Well calibrated (ECE < 0.05)")
            print("    SAM2's confidence scores are trustworthy for clinical use.")
        elif ece < 0.10:
            print("  → Moderately calibrated (0.05 ≤ ECE < 0.10)")
            print("    Confidence scores are useful but should be interpreted with caution.")
        else:
            print("  → Poorly calibrated (ECE ≥ 0.10)")
            print("    Confidence scores do not reliably reflect true accuracy.")

        # Mean probability in lesion vs background regions
        mean_prob_lesion = float(all_probs[all_labels==1].mean())
        mean_prob_bg     = float(all_probs[all_labels==0].mean())
        print(f"\n  Mean predicted probability:")
        print(f"    Inside  lesion (GT=1): {mean_prob_lesion:.4f}")
        print(f"    Outside lesion (GT=0): {mean_prob_bg:.4f}")
        print(f"    Separation:            {mean_prob_lesion - mean_prob_bg:.4f}")

        # ── Save results ──
        summary = {
            "sequence"         : SEQUENCE,
            "n_frames"         : len(frame_files),
            "total_pixels"     : int(len(all_probs)),
            "lesion_pixels"    : int(all_labels.sum()),
            "background_pixels": int((1-all_labels).sum()),
            "ece"              : round(ece, 6),
            "mean_prob_lesion" : round(mean_prob_lesion, 4),
            "mean_prob_background": round(mean_prob_bg, 4),
            "prob_separation"  : round(mean_prob_lesion - mean_prob_bg, 4),
            "mean_dice"        : round(float(np.mean([r['dice']
                                  for r in per_frame_results])), 4),
            "calibration_bins" : bins,
        }

        summary_path = os.path.join(OUTPUT_DIR, "calibration_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary → {summary_path}")

        per_frame_path = os.path.join(OUTPUT_DIR, "per_frame_results.json")
        with open(per_frame_path, 'w') as f:
            json.dump(per_frame_results, f, indent=2)
        print(f"  Per-frame → {per_frame_path}")

        # ── Plots ──
        save_reliability_diagram(
            bins, ece,
            os.path.join(OUTPUT_DIR, "reliability_diagram.png")
        )
        save_confidence_histogram(
            all_probs,
            os.path.join(OUTPUT_DIR, "confidence_histogram.png")
        )

        print(f"\nDone. All results in: {OUTPUT_DIR}/\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
