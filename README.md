# PFA Cardiac Ultrasound Lesion Segmentation

Fully-automatic segmentation of Pulsed-Field-Ablation (PFA) lesions in cardiac
ultrasound video, using **YOLOv8 + fine-tuned SAM2-Hiera-Large** with video
propagation.

**Final result: Dice = 0.878 (4-Fold CV, no data leakage, 431 frames / 4 sequences).**

---

## 1. Repository layout

```
.
├── demo.py                      # Streamlit demo (4 pages)
├── build_compare_panels.py      # Render Original | Manual GT | Best Pred panels
├── training/
│   └── finetune.py              # Fine-tune SAM2-L mask decoder (4-fold CV)
├── checkpoints/
│   ├── yolov8n_pfa.pt           # YOLOv8n bbox detector trained on PFA
│   ├── ft_large_demo.pt         # Mask-decoder weights trained on ALL 4 sequences (demo)
│   └── ft_4fold/                # Per-fold mask-decoder weights (for CV evaluation)
│       ├── ft_Ablation_1.pt
│       ├── ft_Ablation_2.pt
│       ├── ft_Ablation_3.pt
│       └── ft_Ablation_4.pt
├── data/
│   ├── videos/Ablation_{1..4}/  # Raw ultrasound frames (.jpg)
│   └── gt_masks_smooth/         # Manually-prompted SAM2-L GT masks (.png)
├── results/
│   ├── all_metrics_final.json   # Final 4-fold CV metrics
│   ├── per_fold_metrics.json    # Per-fold train-set thresholds and Dice
│   └── masks/Ablation_{1..4}/   # Best predicted masks (binary)
├── examples/
│   └── compare_panels.zip       # 431 side-by-side comparison panels (Original|GT|Pred)
├── optimization_report.md       # 19 experiments — the optimization journey
└── requirements.txt
```

The repo uses **Git LFS** for `.pt`, `.zip`, and the raw `.jpg` frames. Make
sure `git lfs install` is run on your machine before cloning.

```bash
git lfs install
git clone <repo-url>
cd <repo>
```

---

## 2. External dependency: SAM2

The SAM2-Hiera-Large foundation checkpoint (857 MB) is **not** bundled.
Download the official Facebook checkpoint and the SAM2 source tree.

```bash
# Clone MedSAM2 (provides the `sam2` python package + hydra configs)
git clone https://github.com/bowang-lab/MedSAM2 ~/MedSAM2

# Download the SAM2-L base checkpoint
mkdir -p ~/MedSAM2/checkpoints
wget -O ~/MedSAM2/checkpoints/sam2_hiera_large.pt \
  https://huggingface.co/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt
```

If you place MedSAM2 elsewhere, export `SAM2_REPO=/path/to/MedSAM2` (and
optionally `SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt`) before running any
script.

---

## 3. Quick-start: run the Streamlit demo

```bash
pip install -r requirements.txt
streamlit run demo.py --server.port 8501
# Open http://localhost:8501
```

The demo has four pages:
- **Real-time Segmentation** – upload an image (or pick a frame), run end-to-end inference
- **Case Study** – browse the 431 frames with prediction + GT overlays
- **Methods Comparison** – qualitative comparison
- **Metrics Dashboard** – per-sequence Dice / IoU / Precision / Recall

---

## 4. The pipeline (best method)

```
raw frame (1536×545)
    │
    ▼
M_dark_gamma12 preprocessing  ── percentile stretch (2,98) + γ=1.2
    │
    ▼
YOLOv8n bbox detector (lesion ROI)
    │
    ▼
prompt generation
    ├─ 3 positive points, inverted-triangle inside the bbox
    └─ 4 negative points, near the bbox corners
    │
    ▼
SAM2-Hiera-Large + fine-tuned mask decoder  ──  video propagation
    │
    ▼
threshold (searched on training fold only — no leakage)
    │
    ▼
post-processing  ──  largest connected component + fill holes
    │
    ▼
binary mask
```

Fine-tuning details (`training/finetune.py`):
- Frozen image encoder (212 M params), trainable mask decoder (4.2 M, ~2 %)
- 40 epochs, LR = 2e-5, cosine schedule, early stopping (patience 8)
- Loss = Dice + Focal + BCE
- 4-fold leave-one-sequence-out cross-validation
- Threshold for each fold is searched on the *training* split only

---

## 5. Reproducing the metrics

```bash
# Re-train (one fold per Ablation; needs ~12 GB GPU)
python training/finetune.py
```

The 4-fold weights produced under `checkpoints/ft_4fold_train/` should match
the ones bundled in `checkpoints/ft_4fold/` (small variance due to dataloader
shuffling).

Final CV metrics (`results/all_metrics_final.json`):

| Sequence    | Dice  | IoU   | Precision | Recall | Frames |
|-------------|-------|-------|-----------|--------|--------|
| Ablation_1  | 0.897 | 0.816 | 0.898     | 0.908  | 104    |
| Ablation_2  | 0.865 | 0.766 | 0.962     | 0.793  | 128    |
| Ablation_3  | 0.862 | 0.765 | 0.794     | 0.964  | 104    |
| Ablation_4  | 0.888 | 0.802 | 0.879     | 0.909  | 95     |
| **Mean**    | **0.878** | **0.786** | **0.888** | **0.888** | **431** |

---

## 6. Visualizations

`examples/compare_panels.zip` contains 431 side-by-side composites
(Original | Manual GT | Best Prediction). To regenerate them yourself:

```bash
python build_compare_panels.py
# writes ./compare_panels/Ablation_{1..4}/*.png
```

---

## 7. Optimization journey

See `optimization_report.md` for the full record of 19 experiments (different
models, preprocessing variants, prompt strategies, post-processing,
fine-tuning configs) and the rationale for each choice.

---

## 8. Handover notes

- The SAM2 weights live in `checkpoints/`. Demo uses `ft_large_demo.pt`
  (trained on all 4 sequences); CV evaluation uses `ft_4fold/`.
- Per-fold thresholds are stored in `results/per_fold_metrics.json`. If you
  want to re-derive them, search Dice on the training split only — searching
  on the test split is data leakage (we measured a 0.014 drop after fixing).
- Ground truth was generated by manually prompting SAM2-L on every frame
  (~13 boundary points) followed by morphological smoothing. It is *not*
  pixel-perfect; treat Dice ≈ 0.9 as the empirical ceiling.
- Adding more positive points (9, 13) or upgrading YOLO (s/m) did **not**
  improve over 3-pos inverted-triangle. See exp15 / exp17 in the report.

## 9. Extended Experiments 
The following experiments were conducted during the project handover phase.
Full details in `optimization_report.md` (Experiments 20–24).

| Experiment | Result |
|---|---|
| Frozen backbone YOLO training | mAP50-95: 0.443 → 0.601 (+35%) |
| Intensity-guided prompt points | Below baseline — negative result |
| Edge-midpoint negative points | Below baseline — negative result |
| SAM2 calibration analysis | ECE = 0.040 — well calibrated |
| Threshold sweep | Dice 0.854 → 0.885 (+0.031) |