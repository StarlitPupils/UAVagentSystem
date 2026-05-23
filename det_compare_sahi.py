"""SAHI 检测增强验证 (修复版)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, json
from config.settings import config
from core.vision_system import VisionSystem
from core.detection.sahi import SAHIInference
from core.visdrone_loader import VisDroneLoader

VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 30

def box_cxcywh_to_xyxy(box):
    """[cx,cy,w,h] -> [x1,y1,x2,y2]"""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def compute_iou(box1, box2):
    """计算 IoU，自动适配格式"""
    # 统一转为 [x1,y1,x2,y2]
    if len(box1) == 4 and len(box2) == 4:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-8)
    return 0

def match_detections(dets, gts_xyxy, iou_thr=0.5):
    matched_gt = set()
    for d in dets:
        d_xyxy = box_cxcywh_to_xyxy(d['bbox'])
        for gi, gt_xyxy in enumerate(gts_xyxy):
            if gi in matched_gt: continue
            if compute_iou(d_xyxy, gt_xyxy) >= iou_thr:
                matched_gt.add(gi)
                break
    return len(matched_gt)

# 加载数据
loader = VisDroneLoader(VISDRONE_ROOT, VISDRONE_SEQ)
gt_data = {}
with open(os.path.join(VISDRONE_ROOT, "annotations", f"{VISDRONE_SEQ}.txt")) as f:
    for line in f:
        p = line.strip().split(',')
        if len(p) < 8: continue
        frame = int(p[0]) - 1
        if frame >= MAX_FRAMES: continue
        cls = int(p[7]) if len(p) > 7 else 0
        if cls in [0, 11]: continue
        # GT: [left, top, width, height] -> [x1, y1, x2, y2]
        left, top, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        gt_data.setdefault(frame, []).append([left, top, left+w, top+h])

# 初始化
config.DETECTION_CONFIDENCE = 0.28
vs = VisionSystem(device="cuda", use_ensemble=True)
sahi = SAHIInference(vs, slice_height=640, slice_width=640, overlap_ratio=0.15, min_area_ratio=0.00001)

base_tp, base_fp, base_fn = 0, 0, 0
sahi_tp, sahi_fp, sahi_fn = 0, 0, 0

print(f"{'Frame':<6} {'GT':>4} {'基础检测':>8} {'基础TP':>8} {'SAHI检测':>8} {'SAHI_TP':>8}")
print("-" * 50)

for fi in range(MAX_FRAMES):
    frame = loader.get_next_frame()
    if frame is None: break
    gt_xyxy = gt_data.get(fi, [])
    n_gt = len(gt_xyxy)

    # 基础检测
    dets_base = vs.detect_only(frame)
    tp_base = match_detections(dets_base, gt_xyxy)
    fp_base = len(dets_base) - tp_base
    fn_base = n_gt - tp_base
    base_tp += tp_base; base_fp += fp_base; base_fn += fn_base

    # SAHI 检测
    dets_sahi = sahi.detect(frame)
    tp_sahi = match_detections(dets_sahi, gt_xyxy)
    fp_sahi = len(dets_sahi) - tp_sahi
    fn_sahi = n_gt - tp_sahi
    sahi_tp += tp_sahi; sahi_fp += fp_sahi; sahi_fn += fn_sahi

    if fi % 5 == 0:
        print(f"{fi:<6} {n_gt:>4} {len(dets_base):>8} {tp_base:>8} {len(dets_sahi):>8} {tp_sahi:>8}")

total_gt = base_tp + base_fn
print(f"\n{'='*55}")
print(f"检测层面对比 (30帧, GT总数={total_gt})")
print(f"{'='*55}")
recall_b = base_tp / (base_tp + base_fn + 1e-8)
prec_b = base_tp / (base_tp + base_fp + 1e-8)
recall_s = sahi_tp / (sahi_tp + sahi_fn + 1e-8)
prec_s = sahi_tp / (sahi_tp + sahi_fp + 1e-8)
print(f"{'':>8} {'基础融合':>10} {'SAHI融合':>10} {'变化':>10}")
print(f"TP       {base_tp:>10} {sahi_tp:>10} {sahi_tp-base_tp:>+10}")
print(f"FP       {base_fp:>10} {sahi_fp:>10} {sahi_fp-base_fp:>+10}")
print(f"FN       {base_fn:>10} {sahi_fn:>10} {sahi_fn-base_fn:>+10}")
print(f"Recall   {recall_b:>10.3f} {recall_s:>10.3f} {recall_s-recall_b:>+10.3f}")
print(f"Prec     {prec_b:>10.3f} {prec_s:>10.3f} {prec_s-prec_b:>+10.3f}")