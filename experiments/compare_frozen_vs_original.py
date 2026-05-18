"""
compare_frozen_vs_original.py
==============================
Runs full pipeline (YOLO + SAM2) with frozen backbone YOLO weights,
then generates side-by-side comparison images for each sequence.

WHAT IT DOES
------------
1. For each fold, loads the new frozen backbone YOLO weights
2. Runs full pipeline on the test sequence (YOLO → prompts → SAM2)
3. Saves new predicted masks
4. Finds the best frame per sequence (highest GT mask area = clearest lesion)
5. Generates one comparison image per sequence showing:
   - Original frame
   - GT mask (green)
   - Original prediction (red)
   - Frozen backbone prediction (blue)

REQUIREMENTS
------------
Before running this script, make sure these exist:
    runs/yolo_exp/fold1_Ablation_1_frozen/weights/best.pt
    runs/yolo_exp/fold2_Ablation_2_frozen/weights/best.pt
    runs/yolo_exp/fold3_Ablation_3_frozen/weights/best.pt
    runs/yolo_exp/fold4_Ablation_4_frozen/weights/best.pt

USAGE
-----
    python experiments/compare_frozen_vs_original.py

OUTPUT
------
    results/frozen_comparison/
        Ablation_1_comparison.png
        Ablation_2_comparison.png
        Ablation_3_comparison.png
        Ablation_4_comparison.png
        new_masks/
            Ablation_1/  00000.png  00001.png  ...
            Ablation_2/  ...
            Ablation_3/  ...
            Ablation_4/  ...
"""

import os, sys, cv2, math, numpy as np, torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_REPO = os.environ.get('SAM2_REPO', os.path.expanduser('~/MedSAM2'))
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2_video_predictor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCES  = ["Ablation_1", "Ablation_2", "Ablation_3", "Ablation_4"]
VIDEO_BASE = os.path.join(REPO_ROOT, "data/videos")
GT_BASE    = os.path.join(REPO_ROOT, "data/gt_masks_smooth")
OLD_MASKS  = os.path.join(REPO_ROOT, "results/masks")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results/frozen_comparison")
NEW_MASKS  = os.path.join(OUTPUT_DIR, "new_masks")

SAM2_CKPT  = os.environ.get('SAM2_CHECKPOINT',
             os.path.join(SAM2_REPO, 'checkpoints/sam2_hiera_large.pt'))
SAM2_CONFIG = 'configs/sam2/sam2_hiera_l.yaml'

# Frozen backbone YOLO weights — one per fold
# fold_idx matches the val sequence index in SEQUENCES
FROZEN_YOLO_WEIGHTS = {
    "Ablation_1": os.path.join(REPO_ROOT, "runs/yolo_exp/fold1_Ablation_1_frozen/weights/best.pt"),
    "Ablation_2": os.path.join(REPO_ROOT, "runs/yolo_exp/fold2_Ablation_2_frozen/weights/best.pt"),
    "Ablation_3": os.path.join(REPO_ROOT, "runs/yolo_exp/fold3_Ablation_3_frozen/weights/best.pt"),
    "Ablation_4": os.path.join(REPO_ROOT, "runs/yolo_exp/fold4_Ablation_4_frozen/weights/best.pt"),
}

# SAM2 fine-tuned weights — one per fold
FT_WEIGHTS = {
    "Ablation_1": os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_1.pt"),
    "Ablation_2": os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_2.pt"),
    "Ablation_3": os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_3.pt"),
    "Ablation_4": os.path.join(REPO_ROOT, "checkpoints/ft_4fold/ft_Ablation_4.pt"),
}

# Colours
COLOR_GT      = (0,   200,   0)    # green
COLOR_OLD     = (0,    50, 200)    # red (BGR)
COLOR_NEW     = (200,  50,   0)    # blue (BGR)
ALPHA         = 0.45
CONF          = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NEW_MASKS,  exist_ok=True)


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


def get_frame_files(video_dir):
    return sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])


def load_gt_mask(gt_dir, fname):
    gt_path = os.path.join(gt_dir, fname.replace('.jpg', '.png'))
    if not os.path.exists(gt_path):
        return None
    m = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    return (np.squeeze(m) > 127).astype(np.uint8)


def find_best_frame(gt_dir, frame_files):
    """Find frame with largest GT mask area — clearest lesion."""
    best_idx  = 0
    best_area = 0
    for i, fname in enumerate(frame_files):
        gt = load_gt_mask(gt_dir, fname)
        if gt is not None and gt.sum() > best_area:
            best_area = gt.sum()
            best_idx  = i
    return best_idx


def overlay(img_bgr, mask, color, alpha=ALPHA):
    out = img_bgr.copy()
    if mask is not None and mask.any():
        roi = np.squeeze(mask) > 0
        out[roi] = (
            (1-alpha) * out[roi] + alpha * np.array(color)
        ).astype(np.uint8)
    return out


def add_label(img, text, color=(255,255,255)):
    out = img.copy()
    cv2.rectangle(out, (0,0), (img.shape[1], 36), (0,0,0), -1)
    cv2.putText(out, text, (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAM2
# ─────────────────────────────────────────────────────────────────────────────

def load_sam2(ft_weights_path):
    predictor = build_sam2_video_predictor(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CKPT,
        apply_postprocessing=True,
    )
    ft_state = torch.load(ft_weights_path, map_location='cpu')
    vp_state = predictor.state_dict()
    for k, v in ft_state.items():
        if k in vp_state:
            vp_state[k] = v
    predictor.load_state_dict(vp_state)
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# RUN FULL PIPELINE FOR ONE SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(seq, predictor):
    """
    Run YOLO + SAM2 pipeline for one sequence using frozen backbone YOLO.
    Returns dict of {frame_idx: mask (H,W uint8)}
    """
    from ultralytics import YOLO

    video_dir   = os.path.join(VIDEO_BASE, seq)
    frame_files = get_frame_files(video_dir)

    # ── Check frozen YOLO weights exist ──
    yolo_path = FROZEN_YOLO_WEIGHTS[seq]
    if not os.path.exists(yolo_path):
        print(f"  [ERROR] Frozen YOLO weights not found: {yolo_path}")
        print(f"  Make sure training finished for fold {SEQUENCES.index(seq)+1}")
        return None

    # ── YOLO detection on first frame ──
    yolo = YOLO(yolo_path)
    det  = yolo.predict(
        source=os.path.join(video_dir, frame_files[0]),
        conf=CONF, device='cpu', verbose=False
    )
    boxes = det[0].boxes
    if len(boxes) == 0:
        print(f"  [ERROR] YOLO detected nothing in first frame of {seq}")
        return None

    best        = boxes.conf.argmax()
    x1,y1,x2,y2 = boxes.xyxy[best].cpu().numpy().tolist()
    print(f"  YOLO bbox: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    # ── Preprocess frames ──
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp(prefix=f"pfa_{seq}_")
    for fname in frame_files:
        img_bgr = cv2.imread(os.path.join(video_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        proc    = dark_enhance_gamma12(img_rgb)
        cv2.imwrite(os.path.join(temp_dir, fname),
                    cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))

    try:
        # ── SAM2 propagation ──
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

        all_logits = {}
        for frame_idx, _, video_res_masks in predictor.propagate_in_video(
            inference_state
        ):
            all_logits[frame_idx] = video_res_masks[0,0].cpu().numpy()
            print(f"  frame {frame_idx+1}/{len(frame_files)}", end='\r')

        predictor.reset_state(inference_state)
        print()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Find optimal threshold (use 0.30 from calibration analysis) ──
    THRESHOLD = 0.30
    masks = {}
    for frame_idx, logits in all_logits.items():
        prob = 1.0 / (1.0 + np.exp(-np.squeeze(logits)))
        mask = light_postprocess((prob > THRESHOLD).astype(np.uint8))
        masks[frame_idx] = mask

    return masks, frame_files


# ─────────────────────────────────────────────────────────────────────────────
# SAVE NEW MASKS
# ─────────────────────────────────────────────────────────────────────────────

def save_new_masks(seq, masks, frame_files):
    out_dir = os.path.join(NEW_MASKS, seq)
    os.makedirs(out_dir, exist_ok=True)
    for frame_idx, mask in masks.items():
        fname    = frame_files[frame_idx]
        out_path = os.path.join(out_dir, fname.replace('.jpg', '.png'))
        cv2.imwrite(out_path, (mask * 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE COMPARISON IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison(seq, masks, frame_files, best_frame_idx):
    """
    Generate 4-panel comparison image for the best frame of a sequence.
    Panels: Original | GT | Original prediction | Frozen backbone prediction
    """
    fname       = frame_files[best_frame_idx]
    video_dir   = os.path.join(VIDEO_BASE, seq)
    gt_dir      = os.path.join(GT_BASE,    seq)
    old_dir     = os.path.join(OLD_MASKS,  seq)

    # Load images
    img_bgr  = cv2.imread(os.path.join(video_dir, fname))
    gt_mask  = load_gt_mask(gt_dir, fname)
    old_path = os.path.join(old_dir, fname.replace('.jpg', '.png'))
    old_mask = (cv2.imread(old_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) \
               if os.path.exists(old_path) else None
    new_mask = masks.get(best_frame_idx)

    # Compute Dice scores
    def dice(pred, gt):
        if pred is None or gt is None:
            return 0.0
        p, g  = pred.astype(bool), np.squeeze(gt).astype(bool)
        inter = (p & g).sum()
        return float(2*inter / (p.sum()+g.sum()+1e-8))

    old_dice = dice(old_mask, gt_mask)
    new_dice = dice(new_mask, gt_mask)

    # Build panels
    panel1 = add_label(img_bgr.copy(), "Original")
    panel2 = add_label(overlay(img_bgr, gt_mask,  COLOR_GT),  "GT Mask")
    panel3 = add_label(overlay(img_bgr, old_mask, COLOR_OLD),
                       f"Original pred  Dice={old_dice:.3f}")
    panel4 = add_label(overlay(img_bgr, new_mask, COLOR_NEW),
                       f"Frozen backbone  Dice={new_dice:.3f}")

    # Add sequence title bar
    combined = np.hstack([panel1, panel2, panel3, panel4])
    title_bar = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar,
                f"{seq}  —  Frame {best_frame_idx}  "
                f"(Original Dice={old_dice:.3f}  |  Frozen Dice={new_dice:.3f}  |  "
                f"Delta={new_dice-old_dice:+.3f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    final = np.vstack([title_bar, combined])

    # Legend bar
    legend = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
    items  = [
        (COLOR_GT,  "GT mask"),
        (COLOR_OLD, "Original prediction"),
        (COLOR_NEW, "Frozen backbone prediction"),
    ]
    x = 10
    for color, label in items:
        cv2.rectangle(legend, (x,8), (x+20,25), color, -1)
        cv2.putText(legend, label, (x+25, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        x += 220

    final = np.vstack([final, legend])

    out_path = os.path.join(OUTPUT_DIR, f"{seq}_comparison.png")
    cv2.imwrite(out_path, final)
    print(f"  Saved → {out_path}")
    print(f"  Original Dice={old_dice:.3f}  Frozen Dice={new_dice:.3f}  "
          f"Delta={new_dice-old_dice:+.3f}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Frozen Backbone vs Original — Full Pipeline Comparison")
    print(f"{'='*60}\n")

    # Check all frozen YOLO weights exist before starting
    missing = []
    for seq, path in FROZEN_YOLO_WEIGHTS.items():
        if not os.path.exists(path):
            missing.append(f"  {seq}: {path}")
    if missing:
        print("ERROR: Missing frozen YOLO weights:")
        for m in missing:
            print(m)
        print("\nRun training first:")
        print("  python training/train_yolo_frozen_exp.py --mode frozen --cv "
              "--images_dir data/yolo/images --labels_dir data/yolo/labels")
        return

    all_results = []

    for seq in SEQUENCES:
        print(f"\n{'─'*60}")
        print(f"  Processing {seq}...")

        # Load SAM2 with fold-specific weights
        print(f"  Loading SAM2 ({FT_WEIGHTS[seq]})...")
        predictor = load_sam2(FT_WEIGHTS[seq])

        # Run pipeline
        result = run_pipeline(seq, predictor)
        if result is None:
            continue
        masks, frame_files = result

        # Save new masks
        save_new_masks(seq, masks, frame_files)

        # Find best frame
        gt_dir         = os.path.join(GT_BASE, seq)
        best_frame_idx = find_best_frame(gt_dir, frame_files)
        print(f"  Best frame: {best_frame_idx} ({frame_files[best_frame_idx]})")

        # Generate comparison image
        out_path = generate_comparison(seq, masks, frame_files, best_frame_idx)
        all_results.append(out_path)

        # Clean up SAM2 from memory
        del predictor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} comparison images saved")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
