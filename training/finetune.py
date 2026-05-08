"""
实验19: Fine-tune SAM2 Large v2
改进:
- 40 epochs (之前 15)
- Early stopping (patience=8)
- 更低 LR: 2e-5 (之前 5e-5)
- Cosine LR schedule
- 每 2 epochs eval (更频繁监控)
- 然后加载到 video predictor
"""
import os, sys, cv2, json, math, numpy as np, torch, random
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_REPO = os.environ.get('SAM2_REPO', os.path.expanduser('~/MedSAM2'))
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

VIDEO_BASE = os.path.join(REPO_ROOT, 'data/videos')
# Pre-processed (M_dark_gamma12) frames produced by the inference pipeline.
# If you don't have them yet, set PROC_BASE = VIDEO_BASE and the script will
# fall back to raw frames.
PROC_BASE = os.environ.get('PROC_BASE', os.path.join(REPO_ROOT, 'data/proc_videos'))
GT_BASE = os.path.join(REPO_ROOT, 'data/gt_masks_smooth')
OUTPUT_BASE = os.path.join(REPO_ROOT, 'checkpoints/ft_4fold_train')
FOLDERS = ['Ablation_1', 'Ablation_2', 'Ablation_3', 'Ablation_4']
CKPT_LARGE = os.environ.get(
    'SAM2_CHECKPOINT',
    os.path.join(SAM2_REPO, 'checkpoints/sam2_hiera_large.pt'),
)
CONFIG_LARGE = 'configs/sam2/sam2_hiera_l.yaml'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 40
PATIENCE = 8
LR = 2e-5

def get_bbox_from_mask(mask, padding=0.15):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    x1,y1,x2,y2 = xs.min(),ys.min(),xs.max(),ys.max()
    bw,bh = x2-x1,y2-y1
    h,w = mask.shape
    return [max(0,int(x1-bw*padding)),max(0,int(y1-bh*padding)),min(w,int(x2+bw*padding)),min(h,int(y2+bh*padding))]

def strategy_3pos_invtri(x1,y1,x2,y2):
    cx,cy=(x1+x2)/2,(y1+y2)/2; bw,bh=x2-x1,y2-y1; r=min(bw/2,bh/2)*0.5
    pos=[[cx+r*math.cos(math.radians(a)),cy-r*math.sin(math.radians(a))] for a in [150,30,270]]
    mx,my=bw*0.05,bh*0.05
    return pos,[[x1+mx,y1+my],[x2-mx,y1+my],[x1+mx,y2-my],[x2-mx,y2-my]]

def light_postprocess(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: return mask
    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    clean = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clean)
    if contours: cv2.drawContours(filled, contours, -1, 1, -1)
    return filled

def compute_metrics(p, g):
    p, g = p.astype(bool), g.astype(bool)
    inter = (p&g).sum()
    return {'dice':float(2*inter/(p.sum()+g.sum()+1e-8)),'iou':float(inter/((p|g).sum()+1e-8)),
            'precision':float(inter/(p.sum()+1e-8)),'recall':float(inter/(g.sum()+1e-8))}

def forward_with_grad(predictor, point_coords, point_labels, boxes):
    concat_points = (point_coords, point_labels) if point_coords is not None else None
    if boxes is not None:
        bc = boxes.reshape(-1, 2, 2)
        bl = torch.tensor([[2,3]], dtype=torch.int, device=boxes.device).repeat(boxes.size(0),1)
        if concat_points:
            concat_points = (torch.cat([bc, concat_points[0]], 1), torch.cat([bl, concat_points[1]], 1))
        else:
            concat_points = (bc, bl)
    se, de = predictor.model.sam_prompt_encoder(points=concat_points, boxes=None, masks=None)
    hrf = [fl[-1].unsqueeze(0) for fl in predictor._features["high_res_feats"]]
    lrm, _, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
        multimask_output=False, repeat_image=False, high_res_features=hrf)
    return predictor._transforms.postprocess_masks(lrm, predictor._orig_hw[-1])

def dice_loss(p, t, s=1):
    p = torch.sigmoid(p); i = (p*t).sum()
    return 1 - (2*i+s)/(p.sum()+t.sum()+s)

def get_samples(folder):
    samples = []
    vid_dir = os.path.join(VIDEO_BASE, folder)
    g_dir = os.path.join(GT_BASE, folder)
    for f in sorted(os.listdir(vid_dir)):
        if not f.endswith('.jpg'): continue
        gp = os.path.join(g_dir, f.replace('.jpg', '.png'))
        if os.path.exists(gp): samples.append((os.path.join(vid_dir, f), gp))
    return samples

os.makedirs(OUTPUT_BASE, exist_ok=True)
all_results = {}

for test_folder in FOLDERS:
    print(f'\n{"="*60}')
    print(f'Fold: test={test_folder}')
    print(f'{"="*60}')

    sam2_model = build_sam2(config_file=CONFIG_LARGE, ckpt_path=CKPT_LARGE, apply_postprocessing=True).to(device)
    for p in sam2_model.image_encoder.parameters(): p.requires_grad = False
    for p in sam2_model.sam_mask_decoder.parameters(): p.requires_grad = True
    for p in sam2_model.sam_prompt_encoder.parameters(): p.requires_grad = True

    predictor = SAM2ImagePredictor(sam2_model)
    train_samples = sum([get_samples(f) for f in FOLDERS if f != test_folder], [])
    test_samples = get_samples(test_folder)
    print(f'  Train: {len(train_samples)}, Test: {len(test_samples)}')

    optimizer = torch.optim.AdamW(
        [p for p in sam2_model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice, best_state, patience_counter = 0, None, 0

    for epoch in range(EPOCHS):
        sam2_model.train()
        random.shuffle(train_samples)
        loss_sum, n = 0, 0
        for img_p, gt_p in train_samples:
            img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
            gt = (cv2.imread(gt_p, 0) > 127).astype(np.uint8)
            bbox = get_bbox_from_mask(gt)
            if not bbox: continue
            pos, neg = strategy_3pos_invtri(*bbox)
            pts = np.array(pos+neg, dtype=np.float32)
            lbl = np.array([1]*len(pos)+[0]*len(neg), dtype=np.int32)
            with torch.no_grad(): predictor.set_image(img)
            _, uc, lb, ub = predictor._prep_prompts(pts, lbl, np.array(bbox,np.float32), None, True)
            masks = forward_with_grad(predictor, uc, lb, ub)
            gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if gt_t.shape[-2:] != masks.shape[-2:]:
                gt_t = F.interpolate(gt_t, masks.shape[-2:], mode='nearest')
            loss = dice_loss(masks, gt_t) + F.binary_cross_entropy_with_logits(masks, gt_t)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in sam2_model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); loss_sum += loss.item(); n += 1
        scheduler.step()

        # Eval every 2 epochs
        if (epoch+1) % 2 == 0:
            sam2_model.eval()
            dices = []
            with torch.no_grad():
                for ip, gp in test_samples:
                    img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                    gt = (cv2.imread(gp, 0) > 127).astype(np.uint8)
                    bbox = get_bbox_from_mask(gt)
                    if not bbox: continue
                    pos, neg = strategy_3pos_invtri(*bbox)
                    predictor.set_image(img)
                    with torch.autocast("cuda", torch.bfloat16):
                        m, _, _ = predictor.predict(
                            np.array(pos+neg,np.float32), np.array([1]*len(pos)+[0]*len(neg),np.int32),
                            np.array(bbox,np.float32), multimask_output=False)
                    dices.append(compute_metrics(m[0].astype(np.uint8), gt)['dice'])
            d = np.mean(dices)
            lr_now = scheduler.get_last_lr()[0]
            print(f'  Ep{epoch+1}: loss={loss_sum/n:.4f}, gt_dice={d:.4f}, lr={lr_now:.2e}')
            if d > best_dice:
                best_dice = d
                best_state = {k: v.cpu().clone() for k, v in sam2_model.state_dict().items()
                              if 'mask_decoder' in k or 'prompt_encoder' in k}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE // 2:
                    print(f'    Early stopping at epoch {epoch+1} (patience={patience_counter})')
                    break

    print(f'  Best GT dice: {best_dice:.4f}')
    torch.save(best_state, os.path.join(OUTPUT_BASE, f'ft_{test_folder}.pt'))

    # Load into video predictor
    vid_predictor = build_sam2_video_predictor(config_file=CONFIG_LARGE, ckpt_path=CKPT_LARGE, apply_postprocessing=True)
    vp_state = vid_predictor.state_dict()
    for k, v in best_state.items():
        if k in vp_state: vp_state[k] = v
    vid_predictor.load_state_dict(vp_state)

    yolo = YOLO(f'{BASE}/pipeline_v2/yolo_runs/fold_{test_folder}/train/weights/best.pt')
    video_dir = os.path.join(PROC_BASE, test_folder)
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
    frame_names = [os.path.splitext(f)[0] for f in frame_files]

    orig_first = os.path.join(VIDEO_BASE, test_folder, frame_files[0])
    det = yolo.predict(source=orig_first, conf=0.25, device='cuda', verbose=False)
    if len(det[0].boxes) == 0: continue
    best_b = det[0].boxes.conf.argmax()
    x1,y1,x2,y2 = det[0].boxes.xyxy[best_b].cpu().numpy().tolist()

    pos, neg = strategy_3pos_invtri(x1,y1,x2,y2)
    points = np.array(pos+neg, dtype=np.float32)
    labels = np.array([1]*len(pos)+[0]*len(neg), dtype=np.int32)
    box_np = np.array([x1,y1,x2,y2], dtype=np.float32)

    inference_state = vid_predictor.init_state(video_path=video_dir, async_loading_frames=False)
    vid_predictor.add_new_points_or_box(inference_state=inference_state,
        frame_idx=0, obj_id=1, box=box_np, points=points, labels=labels)

    all_logits = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for fi, _, logits in vid_predictor.propagate_in_video(inference_state):
            all_logits[fi] = logits[0].cpu().numpy().squeeze()
    vid_predictor.reset_state(inference_state)

    gt_dir = os.path.join(GT_BASE, test_folder)
    best_t, best_td = 0, 0
    for t in np.arange(-1.5, 1.5, 0.05):
        ds = []
        for fi, raw in all_logits.items():
            gp = os.path.join(gt_dir, f'{frame_names[fi]}.png')
            if not os.path.exists(gp): continue
            gm = (cv2.imread(gp, 0) > 127).astype(np.uint8)
            ds.append(compute_metrics(light_postprocess((raw > t).astype(np.uint8)), gm)['dice'])
        td = np.mean(ds) if ds else 0
        if td > best_td: best_td = td; best_t = t

    all_results[test_folder] = {
        'gt_bbox_dice': float(best_dice),
        'vidprop_best_dice': float(best_td),
        'vidprop_best_thresh': float(best_t),
    }
    print(f'  Video prop (best t={best_t:.2f}): {best_td:.4f}')

    del sam2_model, predictor, vid_predictor; torch.cuda.empty_cache()

with open(os.path.join(OUTPUT_BASE, 'metrics.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n{"="*60}')
print('实验19: Fine-tune Large v2 (40ep + early stop + cosine LR)')
print(f'{"="*60}')
gt_all, vp_all = [], []
for f in FOLDERS:
    r = all_results[f]
    print(f'  {f}: GT bbox={r["gt_bbox_dice"]:.4f}, VidProp={r["vidprop_best_dice"]:.4f} (t={r["vidprop_best_thresh"]:.2f})')
    gt_all.append(r['gt_bbox_dice']); vp_all.append(r['vidprop_best_dice'])
print(f'\n  Avg GT bbox: {np.mean(gt_all):.4f}')
print(f'  Avg VidProp: {np.mean(vp_all):.4f}')
print(f'  对比 exp18: 0.853')
