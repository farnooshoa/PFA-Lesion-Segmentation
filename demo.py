"""
PFA Cardiac Ultrasound Lesion Segmentation Demo
Interactive Streamlit app showcasing YOLO+MedSAM2 pipeline results.
Updated: 4-Fold CV with new mask-prompt GT.
"""
import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the local MedSAM2 / SAM2 source tree (provides the `sam2` package).
# Override via env var SAM2_REPO if you have it installed elsewhere.
SAM2_REPO = os.environ.get('SAM2_REPO', os.path.expanduser('~/MedSAM2'))
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

# Path to the SAM2 Hiera Large foundation checkpoint (857MB, NOT bundled).
# Download from https://huggingface.co/facebook/sam2-hiera-large or place it
# yourself; override via env var SAM2_CHECKPOINT.
SAM2_CHECKPOINT = os.environ.get(
    'SAM2_CHECKPOINT',
    os.path.join(SAM2_REPO, 'checkpoints/sam2_hiera_large.pt'),
)

# ---- Repo-local paths ----
YOLO_WEIGHTS = os.path.join(REPO_ROOT, 'checkpoints/yolov8n_pfa.pt')
MEDSAM2_CONFIG = 'configs/sam2/sam2_hiera_l.yaml'
FT_WEIGHTS = os.path.join(REPO_ROOT, 'checkpoints/ft_large_demo.pt')
VIDEO_BASE = os.path.join(REPO_ROOT, 'data/videos')
GT_BASE = os.path.join(REPO_ROOT, 'data/gt_masks_smooth')
PRED_BASE = os.path.join(REPO_ROOT, 'results')
METRICS_FILE = os.path.join(PRED_BASE, 'all_metrics_final.json')
MEDSAM2_CHECKPOINT = SAM2_CHECKPOINT

ABLATIONS = ['Ablation_1', 'Ablation_2', 'Ablation_3', 'Ablation_4']

# 4-fold methods (video propagation)
# dir name -> metrics key mapping (results_final has masks directly, no subdirs)
METHODS = {
    'YOLO+MedSAM2 (best)': '',
}
METRICS_KEYS = {
    'YOLO+MedSAM2 (best)': 'ft_large_v2_fair',
}

COLORS = {
    'GT':         (0, 255, 0),    # green
    'Prediction': (255, 80, 80),  # red
}

CONF_THRESHOLD = 0.25


# ---- Model loading (cached) ----
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS)


@st.cache_resource
def load_medsam2():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(
        config_file=MEDSAM2_CONFIG,
        ckpt_path=MEDSAM2_CHECKPOINT,
        apply_postprocessing=True,
    )
    # Load fine-tuned weights
    if os.path.exists(FT_WEIGHTS):
        import torch as _torch
        ft_state = _torch.load(FT_WEIGHTS, map_location='cpu')
        sd = model.state_dict()
        for k, v in ft_state.items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd)
    predictor = SAM2ImagePredictor(model)
    return predictor


def make_grid_points_from_box(x1, y1, x2, y2):
    """9 positive (3x3 grid) + 4 negative (bbox corners)."""
    bw, bh = x2 - x1, y2 - y1
    pos = []
    for fy in [0.25, 0.5, 0.75]:
        for fx in [0.25, 0.5, 0.75]:
            pos.append([x1 + bw * fx, y1 + bh * fy])
    mx, my = bw * 0.05, bh * 0.05
    neg = [
        [x1 + mx, y1 + my],
        [x2 - mx, y1 + my],
        [x1 + mx, y2 - my],
        [x2 - mx, y2 - my],
    ]
    points = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * 9 + [0] * 4, dtype=np.int32)
    return points, labels, pos, neg

def strategy_3pos_invtri(x1, y1, x2, y2):
    import math
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    r = min(bw/2, bh/2) * 0.5
    pos = [[cx + r*math.cos(math.radians(a)),
            cy - r*math.sin(math.radians(a))] for a in [150, 30, 270]]
    mx, my = bw*0.05, bh*0.05
    neg = [[x1+mx,y1+my],[x2-mx,y1+my],[x1+mx,y2-my],[x2-mx,y2-my]]
    return pos, neg

def dark_enhance_gamma12(img_rgb):
    """M_dark_gamma12 preprocessing"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray - p2) * 255 / (p98 - p2 + 1e-6), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    lookup = np.array([((i/255.0)**1.2)*255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(out, lookup)


def run_inference(image_rgb):
    """Run YOLO + MedSAM2 pipeline on a single image."""
    yolo = load_yolo()
    predictor = load_medsam2()

    # YOLO on original, MedSAM2 on preprocessed
    det = yolo.predict(source=image_rgb, conf=CONF_THRESHOLD, device='cpu', verbose=False)
    boxes = det[0].boxes
    if len(boxes) == 0:
        return None, None, None, None, None

    best_idx = boxes.conf.argmax()
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().tolist()
    conf = boxes.conf[best_idx].cpu().item()

    pos_pts, neg_pts = strategy_3pos_invtri(x1, y1, x2, y2)
    points = np.array(pos_pts + neg_pts, dtype=np.float32)
    labels = np.array([1]*3 + [0]*4, dtype=np.int32)
    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

    image_proc = dark_enhance_gamma12(image_rgb)
    predictor.set_image(image_proc)
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=labels,
            box=box_np, multimask_output=False,
        )
    mask = masks[0].astype(np.uint8)

    clip_mask = np.zeros_like(mask)
    bx1, by1 = int(max(0, x1)), int(max(0, y1))
    bx2, by2 = int(min(mask.shape[1], x2)), int(min(mask.shape[0], y2))
    clip_mask[by1:by2, bx1:bx2] = mask[by1:by2, bx1:bx2]

    # Post-processing: largest component + fill holes + morphology smooth
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clip_mask, 8)
    if n_labels > 1:
        largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        clip_mask = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(clip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clip_mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k)
    blurred = cv2.GaussianBlur(filled.astype(np.float32) * 255, (51, 51), 0)
    filled = (blurred > 127).astype(np.uint8)

    return filled, (x1, y1, x2, y2), pos_pts, neg_pts, conf


@st.cache_data
def load_metrics():
    with open(METRICS_FILE) as f:
        return json.load(f)


@st.cache_data
def get_frame_list(ablation):
    d = os.path.join(VIDEO_BASE, ablation)
    return sorted([f for f in os.listdir(d) if f.endswith('.jpg')])


def load_frame(ablation, fname):
    path = os.path.join(VIDEO_BASE, ablation, fname)
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(method_dir, ablation, idx):
    if method_dir == '':
        mask_path = os.path.join(PRED_BASE, 'masks', ablation, f'{idx:05d}.png')
    else:
        mask_path = os.path.join(PRED_BASE, method_dir, 'masks', ablation, f'{idx:05d}.png')
    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8)
    return None


def load_gt_mask(ablation, idx):
    mask_path = os.path.join(GT_BASE, ablation, f'{idx:05d}.png')
    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8)
    return None


def overlay(image, mask, color, alpha=0.4):
    out = image.copy()
    if mask is not None and mask.any():
        roi = mask > 0
        out[roi] = ((1 - alpha) * out[roi] + alpha * np.array(color)).astype(np.uint8)
    return out


def compute_frame_metrics(pred, gt):
    if pred is None or gt is None:
        return {}
    p, g = pred.astype(bool), gt.astype(bool)
    intersection = (p & g).sum()
    union = (p | g).sum()
    dice = 2 * intersection / (p.sum() + g.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    prec = intersection / (p.sum() + 1e-8)
    rec = intersection / (g.sum() + 1e-8)
    return {'Dice': dice, 'IoU': iou, 'Precision': prec, 'Recall': rec}


# ---- Page Config ----
st.set_page_config(
    page_title="PFA Lesion Segmentation Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("PFA Cardiac Ultrasound Lesion Segmentation")
st.markdown("**YOLO + MedSAM2 Pipeline** — Fully Automatic PFA Lesion Detection (4-Fold Cross-Validation)")

# ---- Sidebar ----
st.sidebar.header("Settings")
page = st.sidebar.radio("Page", ["Real-time Segmentation", "Case Study", "Methods Comparison", "Metrics Dashboard"])

if page == "Real-time Segmentation":
    st.subheader("Upload Image for Real-time Segmentation")
    st.markdown("Upload a cardiac ultrasound image and the **YOLO+MedSAM2 (3-point inverted triangle)** pipeline will automatically detect and segment PFA lesions.")

    upload_mode = st.sidebar.radio("Input source", ["Upload Image", "Select from Dataset"])

    image_rgb = None

    if upload_mode == "Upload Image":
        uploaded = st.file_uploader("Upload a cardiac ultrasound image", type=["png", "jpg", "jpeg", "bmp", "tif"])
        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        abl = st.sidebar.selectbox("Sequence", ABLATIONS, key="live_abl")
        flist = get_frame_list(abl)
        fidx = st.sidebar.number_input("Picture", 0, len(flist) - 1, 0, key="live_fidx")
        image_rgb = load_frame(abl, flist[fidx])

    if image_rgb is not None:
        run_btn = st.button("Run Segmentation", type="primary")

        if run_btn:
            with st.spinner("Running YOLO detection + MedSAM2 segmentation..."):
                mask, bbox, pos_pts, neg_pts, conf = run_inference(image_rgb)

            if mask is None:
                st.error("No lesion detected by YOLO. Try a different image.")
            else:
                st.success(f"Detection confidence: {conf:.3f}")

                alpha = st.slider("Overlay opacity", 0.1, 0.8, 0.45, 0.05, key="live_alpha")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original + YOLO Detection**")
                    det_vis = image_rgb.copy()
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(det_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    for p in pos_pts:
                        cv2.circle(det_vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
                    for p in neg_pts:
                        cv2.circle(det_vis, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)
                    st.image(det_vis, use_column_width=True)
                    st.caption("Green box: YOLO bbox | Green dots: positive prompts | Red dots: negative prompts")

                with col2:
                    st.markdown("**Segmentation Result**")
                    seg_vis = overlay(image_rgb, mask, COLORS['Prediction'], alpha)
                    st.image(seg_vis, use_column_width=True)
                    area_px = int(mask.sum())
                    total_px = mask.shape[0] * mask.shape[1]
                    st.caption(f"Lesion area: {area_px:,} px ({100 * area_px / total_px:.2f}% of image)")

                st.markdown("---")
                st.markdown("**Combined View**")
                combined = image_rgb.copy()
                combined = overlay(combined, mask, COLORS['Prediction'], alpha)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(combined, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                st.image(combined, use_column_width=True)

                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                import io
                buf = io.BytesIO()
                mask_pil.save(buf, format='PNG')
                st.download_button(
                    label="Download Mask (PNG)",
                    data=buf.getvalue(),
                    file_name="segmentation_mask.png",
                    mime="image/png",
                )
    else:
        st.info("Please upload an image or select one from the dataset to begin.")


elif page == "Case Study":
    st.sidebar.markdown("---")
    ablation = st.sidebar.selectbox("Sequence", ABLATIONS,
                                     format_func=lambda x: f"{x} (Test fold)")
    method_name = st.sidebar.selectbox("Method", list(METHODS.keys()))
    method_dir = METHODS[method_name]

    frames = get_frame_list(ablation)
    n_frames = len(frames)

    frame_idx = st.sidebar.number_input("Picture", 0, n_frames - 1, 0, key="frame_slider")
    show_gt = st.sidebar.checkbox("Show Manually Assisted Segmentation", value=True)
    show_pred = st.sidebar.checkbox("Show Auto Segmentation", value=True)
    alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.8, 0.4, 0.05)

    # Load data
    img = load_frame(ablation, frames[frame_idx])
    gt = load_gt_mask(ablation, frame_idx)
    pred = load_mask(method_dir, ablation, frame_idx)

    # Build visualizations
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original**")
        st.image(img, use_column_width=True)

    with col2:
        vis_gt = overlay(img, gt, COLORS['GT'], alpha) if show_gt and gt is not None else img
        st.markdown("**Manually Assisted Segmentation** :green_circle:")
        st.image(vis_gt, use_column_width=True)

    with col3:
        vis_pred = overlay(img, pred, COLORS['Prediction'], alpha) if show_pred and pred is not None else img
        st.markdown(f"**{method_name}** :red_circle:")
        st.image(vis_pred, use_column_width=True)

    # Combined overlay
    st.markdown("---")
    st.markdown("**Combined Overlay** (Green = Manually Assisted, Red = Auto Segmentation, Yellow = Overlap)")
    combined = img.copy()
    if show_gt and gt is not None:
        combined = overlay(combined, gt, COLORS['GT'], alpha * 0.7)
    if show_pred and pred is not None:
        combined = overlay(combined, pred, COLORS['Prediction'], alpha * 0.7)
    st.image(combined, use_column_width=True)

    # Per-frame metrics
    metrics = compute_frame_metrics(pred, gt)
    if metrics:
        st.markdown("---")
        mcols = st.columns(4)
        for i, (k, v) in enumerate(metrics.items()):
            mcols[i].metric(k, f"{v:.4f}")

    st.caption(f"Picture {frame_idx + 1} / {n_frames}  |  {ablation}  |  {method_name}")


elif page == "Methods Comparison":
    st.sidebar.markdown("---")
    ablation = st.sidebar.selectbox("Sequence", ABLATIONS,
                                     format_func=lambda x: f"{x} (Test fold)")
    frames = get_frame_list(ablation)
    n_frames = len(frames)
    frame_idx = st.sidebar.number_input("Picture", 0, n_frames - 1, 0, key="cmp_slider")
    alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.8, 0.4, 0.05, key="cmp_alpha")

    selected = st.sidebar.multiselect(
        "Methods to compare",
        list(METHODS.keys()),
        default=list(METHODS.keys())[:3]
    )

    img = load_frame(ablation, frames[frame_idx])
    gt = load_gt_mask(ablation, frame_idx)

    n_methods = len(selected)
    if n_methods == 0:
        st.warning("Select at least one method.")
    else:
        cols = st.columns(min(n_methods + 1, 4))
        with cols[0]:
            st.markdown("**Manually Assisted Segmentation**")
            vis_gt = overlay(img, gt, COLORS['GT'], alpha) if gt is not None else img
            st.image(vis_gt, use_column_width=True)
            if gt is not None:
                st.caption(f"Area: {gt.sum()} px")

        for i, mname in enumerate(selected):
            col_idx = (i + 1) % min(n_methods + 1, 4)
            if col_idx == 0:
                cols = st.columns(min(n_methods + 1 - i, 4))
                col_idx = 0
            with cols[col_idx]:
                pred = load_mask(METHODS[mname], ablation, frame_idx)
                vis = overlay(img, pred, COLORS['Prediction'], alpha) if pred is not None else img
                m = compute_frame_metrics(pred, gt)
                dice_str = f"Dice={m['Dice']:.3f}" if m else "N/A"
                st.markdown(f"**{mname}**")
                st.image(vis, use_column_width=True)
                st.caption(dice_str)

    st.caption(f"Picture {frame_idx + 1} / {n_frames}  |  {ablation}")


elif page == "Metrics Dashboard":
    import pandas as pd
    metrics = load_metrics()

    st.subheader("4-Fold Cross-Validation Results")
    st.markdown("Each sequence serves as the test set once, with the other 3 for training. No data leakage.")

    # Build table
    rows = []
    for name in METHODS:
        mkey = METRICS_KEYS.get(name, '')
        if mkey in metrics:
            m = metrics[mkey]
            rows.append({
                'Method': name,
                'Dice': m['mean_dice'],
                'Dice Std': m['std_dice'],
                'IoU': m['mean_iou'],
                'Precision': m['mean_precision'],
                'Recall': m['mean_recall'],
            })
    if rows:
        df = pd.DataFrame(rows).sort_values('Dice', ascending=False)
        st.dataframe(
            df.style.format({
                'Dice': '{:.4f}', 'Dice Std': '{:.4f}',
                'IoU': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}'
            }).highlight_max(subset=['Dice', 'IoU', 'Precision', 'Recall'], color='#90EE90'),
            hide_index=True
        )

    # Bar chart
    st.subheader("Dice Score Comparison")
    chart_data = pd.DataFrame({
        'Method': [r['Method'] for r in rows],
        'Dice': [r['Dice'] for r in rows],
    }).sort_values('Dice', ascending=True)
    st.bar_chart(chart_data.set_index('Method'))

    # Per-sequence breakdown
    st.subheader("Per-Sequence Dice Scores (each as test set)")
    seq_rows = []
    for name in METHODS:
        mkey = METRICS_KEYS.get(name, '')
        if mkey in metrics and 'per_sequence' in metrics[mkey]:
            ps = metrics[mkey]['per_sequence']
            row = {'Method': name}
            for abl in ABLATIONS:
                row[abl] = ps.get(abl, {}).get('dice', 0)
            seq_rows.append(row)

    if seq_rows:
        df_seq = pd.DataFrame(seq_rows)
        st.dataframe(
            df_seq.style.format({a: '{:.4f}' for a in ABLATIONS})
                .highlight_max(subset=ABLATIONS, color='#90EE90'),
            hide_index=True
        )

        st.subheader("Per-Sequence Performance")
        chart_seq = df_seq.set_index('Method')[ABLATIONS]
        st.bar_chart(chart_seq)

    # Pipeline description
    st.markdown("---")
    st.subheader("Pipeline Overview")
    st.markdown("""
    ```
    Input: Cardiac ultrasound video (N frames)
            │
            ▼
    ┌──────────────────────────────┐
    │  YOLOv8-nano Detection       │  First frame only
    │  (4-Fold CV, no data leak)   │
    └───────────┬──────────────────┘
                │
                ▼
    ┌──────────────────────────────┐
    │  Prompt Construction         │
    │  Multiple point strategies   │
    └───────────┬──────────────────┘
                │
                ▼
    ┌──────────────────────────────┐
    │  MedSAM2 Segmentation        │
    │  + Video Propagation         │
    └───────────┬──────────────────┘
                │
                ▼
    Output: Per-frame binary masks
    ```

    **Ground Truth**: MedSAM2 with manual first-frame mask prompt + video propagation

    **Evaluation**: 4-Fold Leave-One-Sequence-Out Cross-Validation
    """)
    st.info("**Best method**: Fine-tuned SAM2 Large + Video Propagation — Dice = 0.878, 4-Fold CV (no data leakage)")
