"""
train_yolo_frozen_exp.py
========================
Frozen-backbone vs full fine-tune experiment for YOLOv8n on PFA lesion data.

WHAT THIS DOES
--------------
Compares two training strategies:
  1. Full fine-tune  : all layers train  (freeze=None)
  2. Frozen backbone : layers 0-9 frozen, only head trains  (freeze=10)

YOLOv8n layer split:
  Backbone  [0-9]  : Conv stem → C2f blocks → SPPF   (1.27 M params, 42%)
  Head     [10-22] : FPN neck (Upsample/Concat/C2f) + Detect  (1.74 M, 58%)

USAGE EXAMPLES
--------------
# 1. Single experiment — frozen backbone, your existing dataset.yaml:
python training/train_yolo_frozen_exp.py \\
    --mode frozen \\
    --dataset data/yolo/dataset.yaml

# 2. Single experiment — full fine-tune:
python training/train_yolo_frozen_exp.py \\
    --mode full \\
    --dataset data/yolo/dataset.yaml

# 3. Compare both in one run (sequential):
python training/train_yolo_frozen_exp.py \\
    --mode compare \\
    --dataset data/yolo/dataset.yaml

# 4. 4-fold leave-one-sequence-out CV (builds per-fold dataset yamls automatically):
python training/train_yolo_frozen_exp.py \\
    --mode frozen \\
    --cv \\
    --images_dir data/yolo/images \\
    --labels_dir data/yolo/labels

# 5. CV comparison (runs all 8 experiments: 4 folds × 2 modes):
python training/train_yolo_frozen_exp.py \\
    --mode compare \\
    --cv \\
    --images_dir data/yolo/images \\
    --labels_dir data/yolo/labels

DATASET FORMAT EXPECTED
-----------------------
For --dataset (single yaml):
  Provide a standard Ultralytics dataset.yaml with train/val paths.
  See create_dataset_yaml() below or data/yolo/dataset_template.yaml.

For --cv (auto-builds yamls):
  images_dir should contain one sub-folder per sequence:
    data/yolo/images/Ablation_1/  *.jpg
    data/yolo/images/Ablation_2/  *.jpg
    data/yolo/images/Ablation_3/  *.jpg
    data/yolo/images/Ablation_4/  *.jpg
  labels_dir mirrors this with YOLO .txt files:
    data/yolo/labels/Ablation_1/  *.txt
    ...

RESULTS
-------
All runs saved under:  runs/yolo_exp/
Summary JSON:          runs/yolo_exp/experiment_summary.json
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here if you want different defaults
# ──────────────────────────────────────────────────────────────────────────────

# Starting checkpoint.  Use 'yolov8n.pt' to start from COCO pretrained (fairer
# comparison with the original training).  Use the PFA checkpoint to continue
# from the existing PFA-trained weights.
DEFAULT_WEIGHTS = "checkpoints/yolov8n_pfa.pt"   # change to 'yolov8n.pt' for COCO start

TRAIN_CFG = dict(
    epochs        = 50,
    patience      = 15,
    batch         = 8,          # reduced from 16 — CPU/RAM friendly; increase if you have GPU
    imgsz         = 640,
    optimizer     = "auto",
    lr0           = 0.001,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3.0,
    # ── augmentation (matches original training) ──
    degrees       = 180,
    translate     = 0.3,
    scale         = 0.7,
    flipud        = 0.5,
    fliplr        = 0.5,
    mosaic        = 1.0,
    mixup         = 0.3,
    hsv_h         = 0.015,
    hsv_s         = 0.3,
    hsv_v         = 0.4,
    # ── reproducibility ──
    seed          = 42,
    deterministic = True,
    # ── output ──
    plots         = True,
    verbose       = True,
    device        = 0,      # change to 0 if you have a GPU
)

SEQUENCES = ["Ablation_1", "Ablation_2", "Ablation_3", "Ablation_4"]

# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Frozen-backbone vs full fine-tune experiment for YOLOv8n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["frozen", "full", "compare"],
        default="compare",
        help="frozen: freeze backbone only. full: train all layers. compare: run both.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Path to a dataset.yaml. Used when --cv is NOT set.",
    )
    p.add_argument(
        "--cv",
        action="store_true",
        help="Run 4-fold leave-one-sequence-out cross-validation.",
    )
    p.add_argument(
        "--images_dir",
        default="data/yolo/images",
        help="Root folder containing one sub-dir per sequence (used with --cv).",
    )
    p.add_argument(
        "--labels_dir",
        default="data/yolo/labels",
        help="Root folder containing one sub-dir per sequence (used with --cv).",
    )
    p.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Starting weights. Default: {DEFAULT_WEIGHTS}",
    )
    p.add_argument(
        "--output_dir",
        default="runs/yolo_exp",
        help="Root directory for all experiment outputs.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_CFG["epochs"],
        help="Number of training epochs.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would run, but don't actually train.",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Dataset YAML helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_dataset_yaml(
    train_dirs: list[str],
    val_dirs: list[str],
    output_path: str | Path,
    nc: int = 1,
    names: list[str] | None = None,
) -> Path:
    """
    Write a minimal Ultralytics dataset.yaml.

    train_dirs / val_dirs should be lists of absolute (or repo-relative) paths
    to image folders.  If there is only one entry, it is written as a plain
    string; if there are multiple, they are written as a YAML list.
    """
    if names is None:
        names = ["lesion"]

    def _fmt(dirs):
        if len(dirs) == 1:
            return dirs[0]
        # YAML list
        lines = "\n".join(f"  - {d}" for d in dirs)
        return f"\n{lines}"

    content = f"""# Auto-generated by train_yolo_frozen_exp.py — {datetime.now().isoformat(timespec='seconds')}
path: .
train: {_fmt(train_dirs)}
val:   {_fmt(val_dirs)}

nc: {nc}
names: {names}
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    return out


def build_cv_yamls(images_dir: str, labels_dir: str, output_dir: str) -> list[dict]:
    """
    Build 4 dataset.yaml files for leave-one-sequence-out CV.

    Returns a list of dicts:
      [{'fold': 1, 'val_seq': 'Ablation_1', 'yaml': Path(...), 'train_seqs': [...]}, ...]
    """
    images_root = Path(images_dir).resolve()
    output_root = Path(output_dir) / "cv_yamls"
    output_root.mkdir(parents=True, exist_ok=True)

    # Verify all sequence folders exist
    for seq in SEQUENCES:
        img_seq = images_root / seq
        if not img_seq.exists():
            raise FileNotFoundError(
                f"Expected image folder not found: {img_seq}\n"
                f"Make sure your images are arranged as:\n"
                f"  {images_root}/Ablation_1/  *.jpg\n"
                f"  {images_root}/Ablation_2/  *.jpg\n  ..."
            )

    folds = []
    for fold_idx, val_seq in enumerate(SEQUENCES, start=1):
        train_seqs = [s for s in SEQUENCES if s != val_seq]
        train_dirs = [str(images_root / s) for s in train_seqs]
        val_dirs   = [str(images_root / val_seq)]

        yaml_path = output_root / f"fold_{fold_idx}_val_{val_seq}.yaml"
        create_dataset_yaml(train_dirs, val_dirs, yaml_path)

        folds.append({
            "fold": fold_idx,
            "val_seq": val_seq,
            "train_seqs": train_seqs,
            "yaml": yaml_path,
        })
        print(f"  [CV] Fold {fold_idx}: train={train_seqs}, val={val_seq} → {yaml_path}")

    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Core training function
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    *,
    mode: str,                  # "frozen" or "full"
    dataset_yaml: str | Path,
    weights: str,
    output_dir: str | Path,
    run_name: str,
    epochs: int,
    dry_run: bool = False,
) -> dict:
    """
    Run one YOLO training experiment and return its metrics.

    mode="frozen" : sets freeze=10  → backbone layers 0-9 are frozen
    mode="full"   : sets freeze=None → all layers train (baseline)
    """
    from ultralytics import YOLO

    freeze_layers = 10 if mode == "frozen" else None

    # Build the full kwargs for model.train()
    train_kwargs = {
        **TRAIN_CFG,
        "data"    : str(dataset_yaml),
        "epochs"  : epochs,
        "project" : str(output_dir),
        "name"    : run_name,
        "freeze"  : freeze_layers,
        "exist_ok": True,
    }

    # Summary line
    frozen_str = f"freeze=10 (backbone locked, head trains)" if mode == "frozen" else "freeze=None (all layers train)"
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {run_name}")
    print(f"  Mode      : {mode.upper()} — {frozen_str}")
    print(f"  Weights   : {weights}")
    print(f"  Dataset   : {dataset_yaml}")
    print(f"  Epochs    : {epochs}  |  Device: {train_kwargs['device']}")
    print(f"{'='*70}\n")

    if dry_run:
        print("  [DRY RUN] Skipping actual training.")
        return {"run_name": run_name, "mode": mode, "dry_run": True}

    # Load model fresh for each run (avoids state leakage between experiments)
    model = YOLO(weights)

    # ── IMPORTANT: inspect what gets frozen ──
    if freeze_layers is not None:
        frozen_params = sum(
            p.numel()
            for i in range(freeze_layers)
            for p in model.model.model[i].parameters()
        )
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Freezing layers 0-{freeze_layers-1}:")
        print(f"    Frozen : {frozen_params:,} params ({100*frozen_params/total_params:.1f}%)")
        print(f"    Trainable: {total_params-frozen_params:,} params ({100*(total_params-frozen_params)/total_params:.1f}%)\n")

    # ── Train ──
    results = model.train(**train_kwargs)

    # ── Extract and return key metrics ──
    metrics = _extract_metrics(results, run_name, mode, str(dataset_yaml))
    return metrics


def _extract_metrics(results, run_name: str, mode: str, dataset_yaml: str) -> dict:
    """Pull the metrics we care about from the Ultralytics Results object."""
    # results.results_dict holds the final epoch metrics
    rd = results.results_dict if hasattr(results, "results_dict") else {}

    metrics = {
        "run_name"       : run_name,
        "mode"           : mode,
        "dataset_yaml"   : dataset_yaml,
        "mAP50"          : round(rd.get("metrics/mAP50(B)",    0.0), 5),
        "mAP50_95"       : round(rd.get("metrics/mAP50-95(B)", 0.0), 5),
        "precision"      : round(rd.get("metrics/precision(B)", 0.0), 5),
        "recall"         : round(rd.get("metrics/recall(B)",    0.0), 5),
        "val_box_loss"   : round(rd.get("val/box_loss",         0.0), 5),
        "val_cls_loss"   : round(rd.get("val/cls_loss",         0.0), 5),
        "val_dfl_loss"   : round(rd.get("val/dfl_loss",         0.0), 5),
        "best_epoch"     : int(results.epoch) if hasattr(results, "epoch") else -1,
        "save_dir"       : str(results.save_dir) if hasattr(results, "save_dir") else "",
        "timestamp"      : datetime.now().isoformat(timespec="seconds"),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Comparison / reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics: list[dict]) -> None:
    """Pretty-print a side-by-side comparison of all runs."""
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    header = f"{'Run':<40} {'Mode':<8} {'mAP50':>7} {'mAP50-95':>9} {'Prec':>7} {'Rec':>7}"
    print(header)
    print("-" * 70)

    for m in all_metrics:
        if m.get("dry_run"):
            print(f"  {m['run_name']:<38} [DRY RUN]")
            continue
        print(
            f"  {m['run_name']:<38} {m['mode']:<8} "
            f"{m['mAP50']:>7.4f} {m['mAP50_95']:>9.4f} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f}"
        )

    # If we have CV results, compute per-mode averages
    cv_frozen = [m for m in all_metrics if m.get("mode") == "frozen" and not m.get("dry_run")]
    cv_full   = [m for m in all_metrics if m.get("mode") == "full"   and not m.get("dry_run")]

    def _avg(lst, key):
        return round(sum(m[key] for m in lst) / len(lst), 4) if lst else 0.0

    if len(cv_frozen) > 1 or len(cv_full) > 1:
        print("-" * 70)
        if cv_frozen:
            print(
                f"  {'FROZEN (mean)':<38} {'frozen':<8} "
                f"{_avg(cv_frozen,'mAP50'):>7.4f} {_avg(cv_frozen,'mAP50_95'):>9.4f} "
                f"{_avg(cv_frozen,'precision'):>7.4f} {_avg(cv_frozen,'recall'):>7.4f}"
            )
        if cv_full:
            print(
                f"  {'FULL FT (mean)':<38} {'full':<8} "
                f"{_avg(cv_full,'mAP50'):>7.4f} {_avg(cv_full,'mAP50_95'):>9.4f} "
                f"{_avg(cv_full,'precision'):>7.4f} {_avg(cv_full,'recall'):>7.4f}"
            )

        if cv_frozen and cv_full:
            delta_map50    = _avg(cv_frozen, "mAP50")    - _avg(cv_full, "mAP50")
            delta_map5095  = _avg(cv_frozen, "mAP50_95") - _avg(cv_full, "mAP50_95")
            sign50    = "▲" if delta_map50   >= 0 else "▼"
            sign5095  = "▲" if delta_map5095 >= 0 else "▼"
            print(f"\n  Δ mAP50   (frozen − full): {sign50} {abs(delta_map50):.4f}")
            print(f"  Δ mAP50-95(frozen − full): {sign5095} {abs(delta_map5095):.4f}")

    print(f"{'='*70}\n")


def save_summary(all_metrics: list[dict], output_dir: str | Path) -> Path:
    """Save all_metrics to a JSON file and return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Summary saved → {summary_path}")
    return summary_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    TRAIN_CFG["epochs"] = args.epochs   # allow CLI override

    output_dir = Path(args.output_dir)
    all_metrics = []

    # ── Determine which modes to run ──
    modes = ["frozen", "full"] if args.mode == "compare" else [args.mode]

    # ── Build experiment list ──
    experiments = []   # list of (run_name, mode, dataset_yaml)

    if args.cv:
        # 4-fold leave-one-sequence-out
        print("\n[CV] Building per-fold dataset YAML files...")
        if args.dry_run:
            # In dry-run, just simulate the fold structure without touching disk
            folds = [
                {"fold": i+1, "val_seq": seq, "train_seqs": [s for s in SEQUENCES if s != seq],
                 "yaml": Path(output_dir) / "cv_yamls" / f"fold_{i+1}_val_{seq}.yaml"}
                for i, seq in enumerate(SEQUENCES)
            ]
            for f in folds:
                print(f"  [CV] Fold {f['fold']}: train={f['train_seqs']}, val={f['val_seq']} → {f['yaml']}")
        else:
            folds = build_cv_yamls(args.images_dir, args.labels_dir, str(output_dir))

        for fold in folds:
            for mode in modes:
                run_name = f"fold{fold['fold']}_{fold['val_seq']}_{mode}"
                experiments.append((run_name, mode, fold["yaml"]))

    else:
        # Single dataset.yaml
        if args.dataset is None:
            parser.error("--dataset is required when --cv is not set.")

        dataset_yaml = Path(args.dataset)
        if not dataset_yaml.exists():
            sys.exit(f"ERROR: dataset.yaml not found: {dataset_yaml}\n"
                     f"  See the docstring at the top of this script for the expected format.")

        for mode in modes:
            run_name = f"yolo_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiments.append((run_name, mode, dataset_yaml))

    # ── Print plan ──
    print(f"\n{'='*70}")
    print(f"  PLAN: {len(experiments)} experiment(s) to run")
    print(f"  Weights : {args.weights}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*70}")
    for i, (name, mode, yaml) in enumerate(experiments, 1):
        freeze_info = "backbone frozen (layers 0-9)" if mode == "frozen" else "all layers train"
        print(f"  [{i}/{len(experiments)}] {name}  ({freeze_info})")
    print()

    if args.dry_run:
        print("  [DRY RUN] Exiting without training.\n")
        return

    # ── Run experiments ──
    for i, (run_name, mode, dataset_yaml) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting: {run_name}")
        try:
            metrics = run_experiment(
                mode        = mode,
                dataset_yaml= dataset_yaml,
                weights     = args.weights,
                output_dir  = output_dir,
                run_name    = run_name,
                epochs      = TRAIN_CFG["epochs"],
                dry_run     = args.dry_run,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n  ERROR in {run_name}: {e}")
            all_metrics.append({
                "run_name": run_name,
                "mode"    : mode,
                "error"   : str(e),
            })

        # Save after each run so progress isn't lost if something fails later
        save_summary(all_metrics, output_dir)

    # ── Final comparison table ──
    print_comparison_table(all_metrics)
    save_summary(all_metrics, output_dir)
    print(f"All done.  Results in: {output_dir}/\n")


if __name__ == "__main__":
    main()
