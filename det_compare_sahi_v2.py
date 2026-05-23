# E:/UAVagent/det_compare_sahi_v2.py
"""SAHI v2 检测对比"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from config.settings import config
from core.vision_system import VisionSystem
from core.detection.sahi_v2 import SAHIInferenceV2
from core.visdrone_loader import VisDroneLoader

VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 30

def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

def compute_iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1, area2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (area1+area2-inter+1e-8)

def match_dets(dets, gts_xyxy, iou_thr=0.5):
    matched = set()
    for d in dets:
        d_xyxy = box_cxcywh_to_xyxy(d['bbox'])
        for gi, gt in enumerate(gts_xyxy):
            if gi in matched: continue
            if compute_iou(d_xyxy, gt) >= iou_thr:
                matched.add(gi)
                break
    return len(matched)

# 加载 GT
loader = VisDroneLoader(VISDRONE_ROOT, VISDRONE_SEQ)
gt_data = {}
with open(os.path.join(VISDRONE_ROOT, "annotations", f"{VISDRONE_SEQ}.txt")) as f:
    for line in f:
        p = line.strip().split(',')
        if len(p) < 8: continue
        frame = int(p[0]) - 1
        if frame >= MAX_FRAMES: continue
        cls = int(p[7]) if len(p) > 7 else 0
        if cls in [0,11]: continue
        l,t,w,h = float(p[2]),float(p[3]),float(p[4]),float(p[5])
        gt_data.setdefault(frame,[]).append([l,t,l+w,t+h])

# 初始化
config.DETECTION_CONFIDENCE = 0.28
vs = VisionSystem(device="cuda", use_ensemble=True)
sahi_v2 = SAHIInferenceV2(vs.ensemble, slice_size=640, overlap=0.25)

base_tp = base_fp = base_fn = 0
sahi_tp = sahi_fp = sahi_fn = 0

for fi in range(MAX_FRAMES):
    frame = loader.get_next_frame()
    if frame is None: break
    gt = gt_data.get(fi, [])
    n_gt = len(gt)
    
    # 基础融合
    dets_base = vs.detect_only(frame)
    tp_b = match_dets(dets_base, gt)
    base_tp += tp_b; base_fp += len(dets_base)-tp_b; base_fn += n_gt-tp_b
    
    # SAHI v2
    dets_sahi = sahi_v2.detect(frame)
    tp_s = match_dets(dets_sahi, gt)
    sahi_tp += tp_s; sahi_fp += len(dets_sahi)-tp_s; sahi_fn += n_gt-tp_s
    
    if fi % 5 == 0:
        print(f"F{fi:3d}: GT={n_gt:2d} Base={len(dets_base):3d}(TP={tp_b}) SAHIv2={len(dets_sahi):3d}(TP={tp_s})")

print(f"\n{'='*50}")
print(f"检测对比 (30帧, GT={base_tp+base_fn})")
print(f"{'':>8} {'基础融合':>10} {'SAHI v2':>10} {'变化':>10}")
recall_b = base_tp/(base_tp+base_fn+1e-8)
prec_b = base_tp/(base_tp+base_fp+1e-8)
recall_s = sahi_tp/(sahi_tp+sahi_fn+1e-8)
prec_s = sahi_tp/(sahi_tp+sahi_fp+1e-8)
print(f"TP       {base_tp:>10} {sahi_tp:>10} {sahi_tp-base_tp:>+10}")
print(f"FP       {base_fp:>10} {sahi_fp:>10} {sahi_fp-base_fp:>+10}")
print(f"FN       {base_fn:>10} {sahi_fn:>10} {sahi_fn-base_fn:>+10}")
print(f"Recall   {recall_b:>10.3f} {recall_s:>10.3f} {recall_s-recall_b:>+10.3f}")
print(f"Prec     {prec_b:>10.3f} {prec_s:>10.3f} {prec_s-prec_b:>+10.3f}")