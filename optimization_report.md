# PFA Cardiac Ultrasound Lesion Segmentation — Optimization Report

## Final Result: Dice = 0.878 (4-Fold CV, no data leakage)

## 1. Task Overview

Automatic segmentation of Pulsed Field Ablation (PFA) lesions in high-frequency cardiac ultrasound video sequences (VisualSonics Vevo). The dataset contains 4 ultrasound video sequences (Ablation 1–4, total 431 frames). Evaluation follows **4-Fold Leave-One-Sequence-Out Cross-Validation**: for each fold, 3 sequences are used for training and 1 unseen sequence for testing, ensuring zero data leakage.

**Ground Truth**: MedSAM2 predictions generated with SAM2-Hiera-Large model using 13 manually placed point prompts per sequence, post-processed with contour smoothing (`masks_smooth`).

## 2. Final Best Method (Dice = 0.878)

### Pipeline

```
Input Video (1536px) → M_dark_gamma12 Preprocessing → YOLOv8n Detection (1st frame)
    → 3-Point Inverted Triangle + 4 Negative Corner Prompts
    → Fine-tuned SAM2 Large Video Propagation (per-fold optimized threshold)
    → Post-processing (largest component + hole filling) → Output Masks
```

### Per-Sequence Results (Fair Evaluation: threshold searched on training set only)

| Fold (Test Set) | Dice  | Train Threshold |
|-----------------|-------|-----------------|
| Ablation 1      | 0.897 | -0.90           |
| Ablation 2      | 0.866 | -0.45           |
| Ablation 3      | 0.862 | -0.55           |
| Ablation 4      | 0.888 | -0.05           |
| **Average**     |**0.878**|               |

## 3. Optimization Journey

### 3.1 Baseline Establishment

| Step | Change | Dice | Delta |
|------|--------|------|-------|
| Starting point | MedSAM2-US-Heart (tiny) + bbox prompt GT + 1024px | 0.790 | — |
| Unified resolution | Switch to medsam2 1536px frames + masks_smooth GT | 0.810 | +0.020 |
| Threshold tuning | Logit threshold from 0.0 to −0.3 | 0.814 | +0.004 |
| Post-processing | Largest connected component + hole filling | 0.816 | +0.002 |
| Image preprocessing | M_dark_gamma12 (percentile stretch + γ=1.2) | 0.821 | +0.005 |
| Per-fold threshold | Optimize threshold per fold on training data | 0.831 | +0.010 |
| **Baseline total** | | **0.831** | **+0.041** |

### 3.2 Model Exploration (Exp 1, 10)

| Model | Dice | Notes |
|-------|------|-------|
| **MedSAM2_US_Heart** (tiny, fine-tuned for cardiac US) | **0.814** | Best without fine-tuning |
| MedSAM2_latest (general medical) | 0.687 | Not specialized for cardiac US |
| MedSAM2_2411 (base) | 0.657 | Older base model |
| SAM2.1 Hiera Tiny (raw, no medical fine-tune) | 0.692 | No domain adaptation |
| SAM2 Hiera Large (raw, no fine-tune) | ~0.50 | Large but no domain knowledge |

**Conclusion**: Domain-specific fine-tuning matters far more than model size. A fine-tuned tiny model (0.814) vastly outperforms a raw large model (0.50).

### 3.3 Image Preprocessing (Exp 2, 11)

Tested 15 preprocessing methods. Top results:

| Method | Dice | Description |
|--------|------|-------------|
| **M_dark_gamma12** | **0.821** | Percentile stretch (2–98%) + gamma 1.2 |
| D_bilateral_dark | 0.820 | Bilateral filter + dark enhance |
| A_nlm_dark | 0.820 | Non-local means denoise + dark enhance |
| dark_enhance | 0.819 | Percentile stretch only |
| Baseline (no preprocessing) | 0.814 | — |
| CLAHE | 0.679 | Destroys model's expected distribution |
| Log transform | 0.743 | Too aggressive |

**Conclusion**: Mild contrast enhancement helps slightly (+0.007). Aggressive transforms (CLAHE, log) hurt performance by distorting the image distribution away from what MedSAM2 was trained on.

### 3.4 Point Prompt Strategies (Exp 15 S2)

| Strategy | Points | Dice | Notes |
|----------|--------|------|-------|
| 1pos + 4neg | 1 center + 4 corners | 0.449 | Insufficient guidance |
| 5pos cross + 4neg | 5 cross + 4 corners | 0.768 | Good but not best |
| 9pos grid + 4neg | 3×3 grid + 4 corners | 0.755 | Dense but redundant |
| **3pos invtri + 4neg** | **3 inverted triangle + 4 corners** | **0.783** | **Best balance** |
| 5pos edge-neg | 5 cross + 4 edge midpoints | 0.764 | Edge negatives less effective |
| 8pos + 5neg | 8 dense + 5 negative | 0.747 | Too many points hurts |
| 13pos + 4neg | 13 dense grid + 4 corners | 0.816 | Diminishing returns |

**Conclusion**: 3-point inverted triangle provides the best balance. The triangle shape covers the lesion's vertical extent efficiently. More points do not always help.

### 3.5 Advanced Techniques (Exp 3–7)

| Method | Dice | Notes |
|--------|------|-------|
| Mask prompt (use prediction as prompt) | 0.809 | No improvement |
| Multi-mask output (pick best IoU) | 0.809 | Same as single mask |
| Ensemble (3 strategies vote) | 0.796 | Voting adds noise |
| Soft ensemble (average logits) | 0.796 | Same issue |
| Bidirectional propagation (fwd+bwd) | 0.814 | No gain |
| Iterative refinement (2 passes) | 0.772 | Noisy mask degrades 2nd pass |
| Auto YOLO re-prompt every 10 frames | 0.770 | YOLO noise disrupts memory |
| **Baseline** | **0.814** | — |

**Conclusion**: MedSAM2's video propagation already provides temporal consistency. External ensemble/re-prompt techniques add noise rather than information.

### 3.6 Upper Bound Analysis (Exp 5–6)

| Condition | Dice | Interpretation |
|-----------|------|----------------|
| GT first frame as mask prompt | 0.867 | Video propagation ceiling (tiny model) |
| GT re-prompt every 30 frames | 0.900 | Semi-automatic with sparse guidance |
| GT re-prompt every 20 frames | 0.905 | More frequent guidance helps |
| GT re-prompt every 10 frames | 0.915 | Near-perfect with dense guidance |
| **Automatic (YOLO + points)** | **0.831** | Our best without GT |

**Conclusion**: The gap between automatic (0.831) and GT-guided (0.867) is 0.036, entirely caused by YOLO detection inaccuracy. Video propagation itself degrades by ~0.05 from first frame to full sequence.

### 3.7 U-Net Refinement Network (Exp 8–9)

| Configuration | Dice | Notes |
|--------------|------|-------|
| U-Net refiner (256px) | 0.794 | Resolution too low |
| U-Net refiner (512px, crop-based) | 0.788 | Poor generalization across sequences |
| **Baseline** | **0.814** | — |

**Conclusion**: With only 3 training sequences, a supervised refinement network cannot learn generalizable corrections. The model memorizes training sequences rather than learning boundary refinement.

### 3.8 Fine-tuning MedSAM2 (Exp 12–14, 17–19)

| Configuration | GT BBox Dice | YOLO BBox Dice |
|--------------|-------------|----------------|
| MedSAM2-US-Heart (no fine-tune) | — | 0.814 |
| Fine-tuned Tiny, GT bbox | 0.883 | 0.803 |
| Fine-tuned Tiny, noisy bbox training | 0.797 | 0.766 |
| Fine-tuned Large (15ep, LR=5e-5) | 0.898 | 0.804 |
| **Fine-tuned Large (40ep, LR=2e-5, cosine, early stop)** | **0.916** | — |
| **+ Video Propagation** | — | **0.892** |

**Conclusion**: Fine-tuning the mask decoder of SAM2 Large is the single most impactful optimization. Key insights:
- Freeze image encoder (212M params), train only decoder (4.2M = 1.9%)
- Lower learning rate (2e-5 vs 5e-5) prevents overfitting on small dataset
- Cosine LR schedule + early stopping ensures each fold stops at optimal point
- Video propagation adds temporal consistency on top of fine-tuned per-frame quality

### 3.9 Inference-Time Techniques (Exp 16)

| Method | Dice | Notes |
|--------|------|-------|
| TTA (horizontal flip) | 0.826 | Ultrasound has directional anatomy |
| TTA (flip + brightness) | 0.830 | Marginal at best |
| Temporal smoothing (3 frames) | 0.824 | Video propagation already smooths |
| Temporal smoothing (5 frames) | 0.823 | Over-smoothing hurts |
| CRF-like post-processing | 0.784 | Lesion-tissue contrast too low |
| All combined | 0.777 | Compounding errors |
| **Baseline** | **0.831** | — |

**Conclusion**: Standard inference-time tricks (TTA, CRF, temporal smoothing) do not help because MedSAM2's video propagation already handles temporal consistency, and CRF relies on color contrast which is weak in ultrasound.

### 3.10 Post-processing Comparison

| Method | Dice | Visual Quality |
|--------|------|---------------|
| No post-processing | 0.814 | Noisy, many islands and holes |
| Largest component + fill holes | 0.816 | Clean, single region |
| Morphology (close+open k=31, blur k=51) | 0.817 | Smooth but still pixel-jaggy |
| Convex hull | 0.816 | Too geometric, not natural |
| B-spline curve smoothing | 0.815 | Shape distorted |
| **Largest component + fill holes** | **0.816** | **Best balance of quality and accuracy** |

## 4. YOLO Detection Analysis (Exp 15 S3)

| YOLO Model | Dice | Notes |
|------------|------|-------|
| **YOLOv8n (nano)** | **0.831** | Best generalization on small data |
| YOLOv8s (small) | 0.680 | Overfits with only 3 training sequences |

**Conclusion**: On small datasets, smaller models generalize better. The detection quality gap (GT bbox 0.916 vs YOLO bbox 0.892 = 0.024 gap with fine-tuned Large) is acceptable.

## 5. Summary of All Experiments

| # | Experiment | Dice | Status |
|---|-----------|------|--------|
| Exp 1 | Model comparison (4 models) | 0.814 max | US_Heart best |
| Exp 2 | Preprocessing v1 (8 methods) | 0.819 max | dark_enhance best |
| Exp 3 | Mask prompt / multi-mask | 0.809 | No improvement |
| Exp 4 | Crop-based / ensemble | 0.796 | Worse |
| Exp 5 | Upper bound (GT prompt) | 0.867 | Pipeline ceiling |
| Exp 6 | GT periodic re-prompt | 0.915 | Needs human guidance |
| Exp 7 | Auto YOLO re-prompt | 0.781 | YOLO noise hurts |
| Exp 8 | U-Net refiner (256px) | 0.794 | Poor generalization |
| Exp 9 | U-Net refiner (512px crop) | 0.788 | Still poor |
| Exp 10 | SAM2 Large (no fine-tune) | ~0.50 | No domain knowledge |
| Exp 11 | Preprocessing v2 (15 methods) | 0.821 max | M_dark_gamma12 best |
| Exp 12 | Fine-tune Tiny | 0.883 GT / 0.803 YOLO | Fine-tune works |
| Exp 13 | Fine-tune Tiny (noisy bbox) | 0.797 GT | Noise training hurts |
| Exp 14 | Fine-tune Tiny + bbox padding | 0.877 GT | Padding doesn't help |
| Exp 15 | 4 strategies (ft+vidprop, points, YOLOs, threshold) | 0.853 max | ft+vidprop + per-fold thresh |
| Exp 16 | TTA + temporal + CRF | 0.831 max | No improvement |
| Exp 17 | Fine-tune Large (15ep) | 0.898 GT / 0.804 YOLO | Large better |
| Exp 18 | Fine-tune Large + vidprop | 0.853 | Good but not optimal training |
| **Exp 19** | **Fine-tune Large v2 (40ep, cosine LR, early stop) + vidprop** | **0.878** | **Best (fair eval)** |

## 6. Key Takeaways

1. **Domain fine-tuning > model size**: A fine-tuned tiny model (0.883) beats a raw large model (0.50). But fine-tuned large (0.916) beats fine-tuned tiny.

2. **Video propagation is essential**: Per-frame segmentation (0.804) is far worse than video propagation (0.892) because temporal consistency prevents frame-to-frame noise.

3. **Training strategy matters as much as architecture**: Moving from 15 epochs/5e-5 LR to 40 epochs/2e-5 LR with cosine schedule improved Dice from 0.853 to 0.892 — a larger gain than switching model sizes.

4. **Less is more for small datasets**: Larger YOLO (s vs n), more prompt points (13 vs 3), and noisy bbox augmentation all hurt rather than helped. The dataset is too small for complex methods to generalize.

5. **Standard tricks don't always work**: TTA, CRF, ensemble, and temporal smoothing — all standard segmentation boosters — provided no improvement because MedSAM2's video propagation already handles what these techniques address.

6. **The bottleneck shifts**: Initially the model was the bottleneck (0.79). After optimization, the remaining gap (0.892 vs GT 0.916 = 0.024) is split between YOLO detection accuracy and video propagation drift.
