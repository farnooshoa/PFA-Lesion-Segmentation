"""
confidence_gated_propagation.py
================================
Confidence-gated memory propagation for SAM2.

THE PROBLEM (drift)
-------------------
SAM2 updates its memory at every frame unconditionally. If the model makes
a bad segmentation on frame N (e.g. due to shadow or motion blur), that
bad mask enters the memory and corrupts frames N+1, N+2, ... causing
the error to propagate — this is called drift.

THE SOLUTION (confidence gating)
---------------------------------
SAM2 internally computes an `object_score_logits` for every frame — a
confidence score that says "how sure am I that the object is here?"
After sigmoid, values close to 1.0 = confident, close to 0.0 = uncertain.

We exploit Python generator semantics:
  1. propagate_in_video() yields one frame at a time and PAUSES
  2. While paused, we check the confidence of the just-processed frame
  3. If confidence < threshold: we DELETE that frame from SAM2's memory
     before the generator resumes — so it never contaminates future frames
  4. The mask prediction for that frame is still used for output

This requires ZERO changes to SAM2 source code.

HOW IT WORKS IN PRACTICE
-------------------------
Standard:
  Frame 1 (conf=0.99) → memory ✓
  Frame 2 (conf=0.97) → memory ✓
  Frame 3 (conf=0.41) → memory ✓  ← BAD frame contaminates memory
  Frame 4 (conf=0.55) → already drifted

Gated (threshold=0.85):
  Frame 1 (conf=0.99) → memory ✓
  Frame 2 (conf=0.97) → memory ✓
  Frame 3 (conf=0.41) → memory BLOCKED  ← bad frame removed from memory
  Frame 4              → uses Frame 2's memory instead → stays on track

USAGE
-----
    python confidence_gated_propagation.py

OUTPUT
------
    results/confidence_gating/
        summary.json          — mean Dice for standard vs gated propagation
        per_frame.json        — per-frame Dice + confidence scores
        confidence_plot.png   — confidence scores per frame with threshold line
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

SEQUENCE        = "Ablation_3"
VIDEO_DIR       = os.path.join(REPO_ROOT, "data/videos",          SEQUENCE)
GT_DIR          = os.path.join(REPO_ROOT, "data/gt_masks_smooth", SEQUENCE)
FT_WEIGHTS      = os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_3.pt")
YOLO_WEIGHTS    = os.path.join(REPO_ROOT, "checkpoints/yolov8n_pfa.pt")
SAM2_CKPT       = os.environ.get('SAM2_CHECKPOINT',
                  os.path.join(SAM2_REPO, 'checkpoints/sam2_hiera_large.pt'))
SAM2_CONFIG     = 'configs/sam2/sam2_hiera_l.yaml'
OUTPUT_DIR      = os.path.join(REPO_ROOT, "results/confidence_gating")

# Confidence threshold — frames with sigmoid(object_score_logits) below
# this value will NOT update the memory.
# Start at 0.85 — tune lower if too many frames are blocked.
CONFIDENCE_THRESHOLD = 0.85

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cpu')   # change to 'cuda' if available


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dark_enhance_gamma12(img_rgb):
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray-p2)*255/(p98-p2+1e-6), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    lut = np.array([((i/255.0)**1.2)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def strategy_3pos_invtri(x1, y1, x2, y2):
    """Original 3-point inverted triangle — copied from finetune.py."""
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    r = min(bw/2, bh/2) * 0.5
    pos = [[cx + r*math.cos(math.radians(a)),
            cy - r*math.sin(math.radians(a))] for a in [150, 30, 270]]
    mx, my = bw*0.05, bh*0.05
    neg = [[x1+mx,y1+my],[x2-mx,y1+my],[x1+mx,y2-my],[x2-mx,y2-my]]
    return pos, neg


def light_postprocess(mask):
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


def compute_metrics(pred, gt):
    p, g  = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    return {
        'dice'     : float(2*inter / (p.sum()+g.sum()+1e-8)),
        'iou'      : float(inter / ((p|g).sum()+1e-8)),
        'precision': float(inter / (p.sum()+1e-8)),
        'recall'   : float(inter / (g.sum()+1e-8)),
    }


def get_frame_files(video_dir):
    return sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])


def load_predictor():
    """Load SAM2 video predictor with fine-tuned weights."""
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
    """Get YOLO bounding box for the first frame."""
    from ultralytics import YOLO
    yolo   = YOLO(YOLO_WEIGHTS)
    det    = yolo.predict(source=frame_path, conf=0.25, device='cpu', verbose=False)
    boxes  = det[0].boxes
    if len(boxes) == 0:
        return None
    best   = boxes.conf.argmax()
    return boxes.xyxy[best].cpu().numpy().tolist()


def get_confidence(inference_state, frame_idx):
    """
    Extract SAM2's confidence score for a given frame.

    object_score_logits is a raw logit — we apply sigmoid to get
    a probability in [0, 1].  Values close to 1.0 = confident the
    object is present; close to 0.0 = uncertain.
    """
    output_dict = inference_state["output_dict"]

    # Check non-conditional outputs first (propagated frames)
    if frame_idx in output_dict["non_cond_frame_outputs"]:
        frame_out = output_dict["non_cond_frame_outputs"][frame_idx]
    elif frame_idx in output_dict["cond_frame_outputs"]:
        # Frame 0 (the frame with the initial prompt) is always confident
        return 1.0
    else:
        return None

    logit = frame_out["object_score_logits"]
    if logit is None:
        return 1.0

    # logit may be a tensor of shape (1,) or scalar
    if isinstance(logit, torch.Tensor):
        conf = torch.sigmoid(logit.float()).mean().item()
    else:
        conf = float(torch.sigmoid(torch.tensor(float(logit))))

    return conf


def remove_from_memory(inference_state, frame_idx):
    """
    Remove a frame from SAM2's non-conditional memory.

    This prevents a low-confidence frame from affecting future frames.
    The mask prediction for this frame is still valid — we just don't
    let it update the memory bank.
    """
    output_dict = inference_state["output_dict"]
    if frame_idx in output_dict["non_cond_frame_outputs"]:
        del output_dict["non_cond_frame_outputs"][frame_idx]

        # Also remove from consolidated index so SAM2 doesn't try to
        # re-use this frame as a reference
        consolidated = inference_state["consolidated_frame_inds"]
        nc_inds = consolidated["non_cond_frame_outputs"]
        if hasattr(nc_inds, 'discard'):
            nc_inds.discard(frame_idx)
        elif frame_idx in nc_inds:
            nc_inds.remove(frame_idx)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: run one propagation pass
# ─────────────────────────────────────────────────────────────────────────────

def run_propagation(predictor, video_dir, frame_files, bbox,
                    confidence_threshold=None, proc_dir=None):
    """
    Run SAM2 video propagation on a sequence.

    confidence_threshold : float or None
        If None  → standard propagation (all frames update memory)
        If float → gated propagation (low-confidence frames blocked)

    Returns list of dicts: {frame_idx, dice, iou, precision, recall, confidence}
    """
    gated = confidence_threshold is not None
    mode  = f"gated (threshold={confidence_threshold})" if gated else "standard"

    # Use preprocessed frames for SAM2 (same as finetune.py)
    # If proc_dir not available, preprocess on the fly
    use_proc = proc_dir and os.path.isdir(proc_dir)
    inference_video_dir = proc_dir if use_proc else video_dir

    # If proc_dir not available, create temp preprocessed frames
    temp_proc_dir = None
    if not use_proc:
        import tempfile, shutil
        temp_proc_dir = tempfile.mkdtemp(prefix="pfa_proc_")
        inference_video_dir = temp_proc_dir
        print(f"  Preprocessing frames → {temp_proc_dir}")
        for fname in frame_files:
            img_bgr = cv2.imread(os.path.join(video_dir, fname))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            proc    = dark_enhance_gamma12(img_rgb)
            proc_bgr = cv2.cvtColor(proc, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(temp_proc_dir, fname), proc_bgr)

    try:
        # ── Initialise inference state ──
        inference_state = predictor.init_state(
            video_path=inference_video_dir,
            async_loading_frames=False,
        )

        # ── Add first-frame prompt ──
        x1, y1, x2, y2 = bbox
        pos, neg = strategy_3pos_invtri(x1, y1, x2, y2)
        points   = np.array(pos + neg, dtype=np.float32)
        labels   = np.array([1]*3 + [0]*4, dtype=np.int32)
        box_np   = np.array([x1, y1, x2, y2], dtype=np.float32)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box_np,
            points=points,
            labels=labels,
        )

        # ── Propagate with optional confidence gating ──
        results          = []
        n_blocked        = 0
        n_total          = 0
        confidences      = {}

        print(f"\n  Running {mode} propagation ({len(frame_files)} frames)...")

        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            inference_state
        ):
            # ── Get confidence BEFORE potentially removing from memory ──
            conf = get_confidence(inference_state, frame_idx)
            confidences[frame_idx] = conf

            # ── Confidence gating ──
            blocked = False
            if gated and frame_idx > 0 and conf is not None:
                if conf < confidence_threshold:
                    remove_from_memory(inference_state, frame_idx)
                    blocked = True
                    n_blocked += 1

            n_total += 1

            # ── Extract mask ──
            mask_logits = video_res_masks[0, 0].cpu().numpy()  # (H, W)
            pred_mask   = light_postprocess((mask_logits > 0).astype(np.uint8))

            # ── Load GT mask ──
            fname   = frame_files[frame_idx]
            gt_path = os.path.join(GT_DIR, fname.replace('.jpg', '.png'))
            if not os.path.exists(gt_path):
                gt_path = os.path.join(GT_DIR, f"{frame_idx:05d}.png")

            if os.path.exists(gt_path):
                gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                m  = compute_metrics(pred_mask, gt)
            else:
                m = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

            m['frame']      = frame_idx
            m['confidence'] = round(conf, 4) if conf is not None else None
            m['blocked']    = blocked
            results.append(m)

            status = "BLOCKED" if blocked else "ok"
            print(
                f"  frame {frame_idx+1:3d}/{len(frame_files)} — "
                f"Dice={m['dice']:.3f}  conf={conf:.3f if conf else 0:.3f}  [{status}]",
                end='\r'
            )

        print()  # newline
        if gated:
            print(f"  Blocked {n_blocked}/{n_total} frames "
                  f"({100*n_blocked/max(n_total,1):.1f}%) from memory update")

        predictor.reset_state(inference_state)

    finally:
        # Clean up temp dir if we created one
        if temp_proc_dir:
            import shutil
            shutil.rmtree(temp_proc_dir, ignore_errors=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(std_results, gated_results, output_path):
    """Save confidence scores + Dice comparison plot."""
    n      = len(std_results)
    W, H   = max(900, n*7), 500
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 240

    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 60
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    # Gridlines
    for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
        y = pad_t + int(plot_h * (1 - v))
        cv2.line(canvas, (pad_l, y), (W-pad_r, y), (200,200,200), 1)
        cv2.putText(canvas, f"{v:.1f}", (5, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)

    def to_y(v):
        return pad_t + int(plot_h * (1 - max(0, min(1, v))))

    def to_x(i):
        return pad_l + int(plot_w * i / max(n-1, 1))

    # Confidence scores (background, light blue area)
    conf_pts = []
    for r in gated_results:
        c = r.get('confidence') or 0
        conf_pts.append((to_x(r['frame']), to_y(c)))
    for j in range(1, len(conf_pts)):
        cv2.line(canvas, conf_pts[j-1], conf_pts[j], (173, 216, 230), 1)

    # Threshold line
    thresh_y = to_y(CONFIDENCE_THRESHOLD)
    cv2.line(canvas, (pad_l, thresh_y), (W-pad_r, thresh_y), (255, 100, 0), 2)
    cv2.putText(canvas, f"threshold={CONFIDENCE_THRESHOLD}",
                (pad_l+5, thresh_y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,100,0), 1)

    # Mark blocked frames
    for r in gated_results:
        if r.get('blocked'):
            x = to_x(r['frame'])
            cv2.line(canvas, (x, pad_t), (x, H-pad_b), (255, 200, 200), 1)

    # Standard Dice line (green)
    std_pts = [(to_x(r['frame']), to_y(r['dice'])) for r in std_results]
    for j in range(1, len(std_pts)):
        cv2.line(canvas, std_pts[j-1], std_pts[j], (34, 139, 34), 2)

    # Gated Dice line (blue)
    gated_pts = [(to_x(r['frame']), to_y(r['dice'])) for r in gated_results]
    for j in range(1, len(gated_pts)):
        cv2.line(canvas, gated_pts[j-1], gated_pts[j], (200, 100, 0), 2)

    # Legend
    items = [
        ((34,139,34),    "Standard Dice"),
        ((200,100,0),    "Gated Dice"),
        ((173,216,230),  "Confidence score"),
        ((255,100,0),    "Threshold"),
        ((255,200,200),  "Blocked frame"),
    ]
    x = pad_l
    for color, label in items:
        cv2.circle(canvas, (x+8, H-20), 6, color, -1)
        cv2.putText(canvas, label, (x+18, H-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,60,60), 1)
        x += 160

    cv2.imwrite(output_path, canvas)
    print(f"  Plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Confidence-Gated Memory Propagation — {SEQUENCE}")
    print(f"  Threshold: sigmoid(object_score_logits) >= {CONFIDENCE_THRESHOLD}")
    print(f"{'='*60}\n")

    frame_files = get_frame_files(VIDEO_DIR)
    print(f"  Found {len(frame_files)} frames in {SEQUENCE}")

    # ── Get YOLO bbox from first frame ──
    print("\n  Getting YOLO bounding box from first frame...")
    first_frame_path = os.path.join(VIDEO_DIR, frame_files[0])
    bbox = get_yolo_bbox(first_frame_path)
    if bbox is None:
        print("  ERROR: YOLO detected no lesion in first frame.")
        return
    print(f"  YOLO bbox: {[round(v,1) for v in bbox]}")

    # ── Load model once — reused for both runs ──
    predictor = load_predictor()

    # ── Run 1: Standard propagation ──
    print(f"\n{'─'*60}")
    std_results = run_propagation(
        predictor, VIDEO_DIR, frame_files, bbox,
        confidence_threshold=None,
    )

    # ── Run 2: Confidence-gated propagation ──
    print(f"\n{'─'*60}")
    gated_results = run_propagation(
        predictor, VIDEO_DIR, frame_files, bbox,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # ── Summary ──
    def mean(lst, key):
        return float(np.mean([r[key] for r in lst]))

    summary = {
        "sequence"  : SEQUENCE,
        "threshold" : CONFIDENCE_THRESHOLD,
        "n_frames"  : len(frame_files),
        "n_blocked" : sum(1 for r in gated_results if r.get('blocked')),
        "standard"  : {
            "mean_dice"     : round(mean(std_results,   'dice'),      4),
            "mean_iou"      : round(mean(std_results,   'iou'),       4),
            "mean_precision": round(mean(std_results,   'precision'), 4),
            "mean_recall"   : round(mean(std_results,   'recall'),    4),
        },
        "gated": {
            "mean_dice"     : round(mean(gated_results, 'dice'),      4),
            "mean_iou"      : round(mean(gated_results, 'iou'),       4),
            "mean_precision": round(mean(gated_results, 'precision'), 4),
            "mean_recall"   : round(mean(gated_results, 'recall'),    4),
        },
    }
    summary['delta_dice'] = round(
        summary['gated']['mean_dice'] - summary['standard']['mean_dice'], 4
    )

    # ── Print table ──
    print(f"\n{'='*60}")
    print(f"  RESULTS — {SEQUENCE}  (threshold={CONFIDENCE_THRESHOLD})")
    print(f"  Frames blocked from memory: "
          f"{summary['n_blocked']}/{summary['n_frames']}")
    print(f"{'='*60}")
    print(f"  {'Metric':<14} {'Standard':>12} {'Gated':>12} {'Delta':>10}")
    print(f"  {'-'*50}")
    for key in ['mean_dice', 'mean_iou', 'mean_precision', 'mean_recall']:
        label = key.replace('mean_', '').capitalize()
        s_val = summary['standard'][key]
        g_val = summary['gated'][key]
        delta = g_val - s_val
        sign  = '▲' if delta >= 0 else '▼'
        print(f"  {label:<14} {s_val:>12.4f} {g_val:>12.4f} "
              f"{sign}{abs(delta):>8.4f}")
    print(f"{'='*60}")

    delta = summary['delta_dice']
    if delta > 0:
        print(f"\n  ✓ Gated propagation is BETTER by {delta:.4f} Dice points")
    elif delta < 0:
        print(f"\n  ✗ Standard propagation is BETTER by {abs(delta):.4f} Dice points")
    else:
        print(f"\n  = No difference")

    # ── Save ──
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {summary_path}")

    per_frame_path = os.path.join(OUTPUT_DIR, "per_frame.json")
    with open(per_frame_path, 'w') as f:
        json.dump({"standard": std_results, "gated": gated_results}, f, indent=2)
    print(f"  Per-frame → {per_frame_path}")

    plot_path = os.path.join(OUTPUT_DIR, "confidence_plot.png")
    save_plot(std_results, gated_results, plot_path)

    print(f"\nDone. Results in: {OUTPUT_DIR}/\n")
    print("TIP: If too many/few frames were blocked, adjust CONFIDENCE_THRESHOLD")
    print("     at the top of this script (currently", CONFIDENCE_THRESHOLD, ")")
    print("     Higher = more strict = fewer frames update memory")
    print("     Lower  = more relaxed = more frames update memory\n")


if __name__ == "__main__":
    main()
