"""
prepare_yolo_dataset.py
=======================
Converts GT masks → YOLO bounding-box labels, then builds the dataset
folder structure ready for training.

WHAT IT DOES
------------
1. Reads every GT mask from  data/gt_masks_smooth/Ablation_{1..4}/
2. Finds the white (lesion) region and computes its bounding box
3. Writes a YOLO .txt label file  (format: 0 cx cy w h, all 0-1 normalised)
4. Copies (or symlinks) the matching .jpg frame into data/yolo/images/
5. Writes a dataset.yaml for each CV fold + one all_data.yaml

RESULT
------
data/yolo/
  images/
    Ablation_1/   frame_0001.jpg  frame_0002.jpg  ...
    Ablation_2/   ...
    Ablation_3/   ...
    Ablation_4/   ...
  labels/
    Ablation_1/   frame_0001.txt  frame_0002.txt  ...
    Ablation_2/   ...
    Ablation_3/   ...
    Ablation_4/   ...
  folds/
    fold_1_val_Ablation_1.yaml
    fold_2_val_Ablation_2.yaml
    fold_3_val_Ablation_3.yaml
    fold_4_val_Ablation_4.yaml
  all_data.yaml

USAGE
-----
  # From the root of your repo:
  python prepare_yolo_dataset.py

  # Or with explicit paths:
  python prepare_yolo_dataset.py \\
      --masks_dir  data/gt_masks_smooth \\
      --frames_dir data/videos \\
      --output_dir data/yolo

  # Preview only — don't write any files:
  python prepare_yolo_dataset.py --dry_run
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCES   = ["Ablation_1", "Ablation_2", "Ablation_3", "Ablation_4"]
MASK_THRESH = 127          # pixel value above which a pixel counts as "lesion"
PADDING     = 0.05         # fractional padding added to each side of the bbox
                           # e.g. 0.05 = 5% of image width/height on each side
                           # helps YOLO see a little context around the lesion


# ─────────────────────────────────────────────────────────────────────────────
# CORE: mask → YOLO label
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_yolo_bbox(mask_path: Path, padding: float = PADDING):
    """
    Read a binary GT mask and return YOLO bbox (cx, cy, w, h) normalised 0-1.

    Returns None if the mask contains no lesion pixels.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    img_h, img_w = mask.shape

    # Find all lesion pixels
    ys, xs = np.where(mask > MASK_THRESH)

    if len(xs) == 0:
        return None   # no lesion in this frame

    # Tight bounding box in pixel coords
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Add padding (clamped to image boundaries)
    pad_x = int(padding * img_w)
    pad_y = int(padding * img_h)
    x_min = max(0,     x_min - pad_x)
    x_max = min(img_w, x_max + pad_x)
    y_min = max(0,     y_min - pad_y)
    y_max = min(img_h, y_max + pad_y)

    # Convert to YOLO format: centre + size, normalised
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h

    return cx, cy, w, h


def write_yolo_label(label_path: Path, bbox):
    """Write a single YOLO label file.  bbox = (cx, cy, w, h) normalised."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, w, h = bbox
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: find matching frame for a mask
# ─────────────────────────────────────────────────────────────────────────────

def find_frame(frames_seq_dir: Path, mask_stem: str) -> Path | None:
    """
    Find the .jpg frame that corresponds to a mask filename.

    Masks were generated from frames, so stems should match exactly.
    Falls back to a numeric search if stems don't match.
    """
    # Exact stem match first
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = frames_seq_dir / (mask_stem + ext)
        if candidate.exists():
            return candidate

    # Sometimes masks are named differently — try matching by frame number
    # e.g. mask "frame_0042_mask.png"  →  frame "frame_0042.jpg"
    # Strip common suffixes
    for suffix in ("_mask", "_gt", "_seg", "_smooth"):
        stem_clean = mask_stem.replace(suffix, "")
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = frames_seq_dir / (stem_clean + ext)
            if candidate.exists():
                return candidate

    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET YAML
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset_yaml(path: Path, train_seqs: list[str],
                       val_seqs: list[str], images_root: Path):
    """Write one Ultralytics dataset.yaml."""
    def fmt(seqs):
        paths = [str((images_root / s).resolve()) for s in seqs]
        if len(paths) == 1:
            return paths[0]
        lines = "\n".join(f"  - {p}" for p in paths)
        return f"\n{lines}"

    content = f"""# Auto-generated by prepare_yolo_dataset.py
path: .
train: {fmt(train_seqs)}
val:   {fmt(val_seqs)}

nc: 1
names:
  - lesion
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def convert_all(masks_dir: Path, frames_dir: Path,
                output_dir: Path, dry_run: bool):
    """
    Main loop: for every mask in every sequence, generate a label
    and copy the matching frame.
    """
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"

    total_ok      = 0
    total_empty   = 0   # masks with no lesion pixels
    total_missing = 0   # masks whose frame couldn't be found
    skipped_seqs  = []

    for seq in SEQUENCES:
        mask_seq_dir  = masks_dir  / seq
        frame_seq_dir = frames_dir / seq

        if not mask_seq_dir.exists():
            print(f"  [SKIP] Mask folder not found: {mask_seq_dir}")
            skipped_seqs.append(seq)
            continue
        if not frame_seq_dir.exists():
            print(f"  [SKIP] Frame folder not found: {frame_seq_dir}")
            skipped_seqs.append(seq)
            continue

        mask_files = sorted(mask_seq_dir.glob("*.png")) + \
                     sorted(mask_seq_dir.glob("*.jpg"))

        if not mask_files:
            print(f"  [WARN] No mask files found in {mask_seq_dir}")
            continue

        print(f"\n  [{seq}]  {len(mask_files)} masks found")

        seq_ok = seq_empty = seq_missing = 0

        for mask_path in mask_files:
            # ── Find matching frame ──
            frame_path = find_frame(frame_seq_dir, mask_path.stem)
            if frame_path is None:
                print(f"    [WARN] No matching frame for: {mask_path.name}")
                seq_missing += 1
                continue

            # ── Compute bbox ──
            try:
                bbox = mask_to_yolo_bbox(mask_path)
            except Exception as e:
                print(f"    [ERROR] {mask_path.name}: {e}")
                seq_missing += 1
                continue

            if bbox is None:
                # Empty mask — write an empty label (background frame)
                if not dry_run:
                    label_path = labels_out / seq / (frame_path.stem + ".txt")
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("")   # empty = no object
                seq_empty += 1
                continue

            cx, cy, w, h = bbox

            if not dry_run:
                # ── Write label ──
                label_path = labels_out / seq / (frame_path.stem + ".txt")
                write_yolo_label(label_path, bbox)

                # ── Copy frame ──
                dest_frame = images_out / seq / frame_path.name
                dest_frame.parent.mkdir(parents=True, exist_ok=True)
                if not dest_frame.exists():
                    shutil.copy2(frame_path, dest_frame)

            seq_ok += 1

        print(f"    ✓ {seq_ok} labels written  |  "
              f"{seq_empty} empty masks  |  {seq_missing} missing frames")
        total_ok      += seq_ok
        total_empty   += seq_empty
        total_missing += seq_missing

    return total_ok, total_empty, total_missing, skipped_seqs


def write_all_yamls(output_dir: Path):
    """Write the 4 fold yamls + 1 all_data yaml."""
    images_root = output_dir / "images"
    folds_dir   = output_dir / "folds"

    # 4-fold leave-one-out
    for fold_idx, val_seq in enumerate(SEQUENCES, start=1):
        train_seqs = [s for s in SEQUENCES if s != val_seq]
        yaml_path  = folds_dir / f"fold_{fold_idx}_val_{val_seq}.yaml"
        write_dataset_yaml(yaml_path, train_seqs, [val_seq], images_root)
        print(f"  Fold {fold_idx}: train={train_seqs}  val=[{val_seq}]  → {yaml_path}")

    # All data (useful for final training, not evaluation)
    all_yaml = output_dir / "all_data.yaml"
    write_dataset_yaml(all_yaml, SEQUENCES, SEQUENCES, images_root)
    print(f"  All data: {all_yaml}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert GT masks to YOLO labels and build dataset structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--masks_dir",  default="data/gt_masks_smooth",
                        help="Root folder of GT masks (default: data/gt_masks_smooth)")
    parser.add_argument("--frames_dir", default="data/videos",
                        help="Root folder of video frames (default: data/videos)")
    parser.add_argument("--output_dir", default="data/yolo",
                        help="Where to write images/, labels/, yamls (default: data/yolo)")
    parser.add_argument("--padding",    type=float, default=PADDING,
                        help=f"Fractional bbox padding (default: {PADDING})")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Preview only — don't write any files")
    args = parser.parse_args()

    masks_dir  = Path(args.masks_dir)
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "="*60)
    print("  PFA YOLO Dataset Preparation")
    print("="*60)
    print(f"  Masks dir  : {masks_dir.resolve()}")
    print(f"  Frames dir : {frames_dir.resolve()}")
    print(f"  Output dir : {output_dir.resolve()}")
    print(f"  Padding    : {args.padding*100:.0f}% per side")
    print(f"  Dry run    : {args.dry_run}")
    print("="*60 + "\n")

    if not masks_dir.exists():
        print(f"ERROR: masks_dir not found: {masks_dir.resolve()}")
        print("  Make sure you run this from the root of your repo.")
        return

    if not frames_dir.exists():
        print(f"ERROR: frames_dir not found: {frames_dir.resolve()}")
        return

    # Step 1: convert masks → labels
    print("STEP 1: Converting masks to YOLO labels...")
    total_ok, total_empty, total_missing, skipped = convert_all(
        masks_dir, frames_dir, output_dir, dry_run=args.dry_run
    )

    print(f"\n  ── Summary ──")
    print(f"  Labels written : {total_ok}")
    print(f"  Empty masks    : {total_empty}  (frames with no visible lesion)")
    print(f"  Missing frames : {total_missing}")
    if skipped:
        print(f"  Skipped seqs   : {skipped}")

    # Step 2: write dataset yamls
    if not args.dry_run:
        print("\nSTEP 2: Writing dataset YAML files...")
        write_all_yamls(output_dir)

    print("\nDone!")
    if not args.dry_run:
        print(f"\nNext step — run the frozen backbone experiment:")
        print(f"  python training/train_yolo_frozen_exp.py \\")
        print(f"      --mode compare --cv \\")
        print(f"      --images_dir {output_dir}/images \\")
        print(f"      --labels_dir {output_dir}/labels")
    print()


if __name__ == "__main__":
    main()
