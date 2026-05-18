# E:/UAVagent1.1/benchmark_real_mot_v2.py
"""真实数据集 MOT 基准测试 v2 - 外部共识过滤"""
import sys, os, cv2, json, numpy as np, time, glob
sys.path.insert(0, "E:/UAVagent1.1")
from config.settings import config
from core.vision_system import VisionSystem
import motmetrics as mm

# ---- 共识过滤器（内联，不依赖 vision_system 修改）----
def consensus_filter(detections, min_models=2, min_conf_single=0.25):
    """只保留多模型确认 或 高置信度单检"""
    kept = []
    dropped = 0
    for d in detections:
        n = d.get('num_models', 1)
        c = d.get('confidence', 0)
        if n >= min_models:
            kept.append(d)
        elif c >= min_conf_single:
            kept.append(d)
        else:
            dropped += 1
    if dropped:
        print(f"  [Filter] -{dropped} low-conf single-model detections")
    return kept

# ---- 配置 ----
VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 100
APPLY_CONSENSUS_FILTER = True  # ← 开关

# ---- 数据集加载函数（同前）----
def load_ground_truth(root, seq, max_frames):
    gt = {}
    for ann_name in [f"{seq}.txt", os.path.join(seq, "gt.txt")]:
        ann_path = os.path.join(root, "annotations", ann_name)
        if os.path.isfile(ann_path):
            with open(ann_path) as f:
                for line in f:
                    p = line.strip().split(',')
                    if len(p) < 8: continue
                    frame = int(p[0]) - 1
                    if frame >= max_frames: continue
                    cls = int(p[7]) if len(p) > 7 else 0
                    if cls in [0, 11]: continue
                    gt.setdefault(frame, []).append({
                        'id': int(p[1]), 
                        'bbox': [float(p[2]), float(p[3]), float(p[4]), float(p[5])],
                        'class': cls
                    })
            break
    return gt

def load_frames(root, seq, max_frames):
    seq_path = os.path.join(root, "sequences", seq)
    frames = []
    if os.path.isdir(seq_path):
        imgs = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(('.jpg','.png'))])
        for fname in imgs[:max_frames]:
            img = cv2.imread(os.path.join(seq_path, fname))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames

def evaluate_mot(gt_data, pred_data, n_frames):
    acc = mm.MOTAccumulator(auto_id=True)
    for fi in range(n_frames):
        g = gt_data.get(fi, [])
        p = pred_data.get(fi, [])
        if not g and not p: continue
        if g and p:
            ious = np.zeros((len(g), len(p)))
            for i, gg in enumerate(g):
                gx, gy, gw, gh = gg['bbox']
                for j, pp in enumerate(p):
                    px, py, pw, ph = pp['bbox']
                    ix1, iy1 = max(gx, px), max(gy, py)
                    ix2, iy2 = min(gx+gw, px+pw), min(gy+gh, py+ph)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    union = gw*gh + pw*ph - inter
                    ious[i, j] = inter / (union + 1e-8)
        else:
            ious = np.zeros((len(g), len(p)))
        acc.update([gg['id'] for gg in g], [pp['id'] for pp in p], ious)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='eval')

def main():
    print("=" * 70)
    print(f"UAVagent 1.1 真实 MOT 基准 v2 (共识过滤={'ON' if APPLY_CONSENSUS_FILTER else 'OFF'})")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    frames = load_frames(VISDRONE_ROOT, VISDRONE_SEQ, MAX_FRAMES)
    gt = load_ground_truth(VISDRONE_ROOT, VISDRONE_SEQ, MAX_FRAMES)
    print(f"  帧={len(frames)}  GT目标={sum(len(v) for v in gt.values())}")

    print("\n[2/5] 初始化系统...")
    config.DETECTION_CONFIDENCE = 0.15
    vs_single = VisionSystem(device="cpu", use_ensemble=False)
    vs_ensemble = VisionSystem(device="cpu", use_ensemble=True)
    print(f"  单模型: {vs_single.get_stats()['model_names'][0]}")
    print(f"  融合: {vs_ensemble.get_stats()['num_models']} models")

    print(f"\n[3/5] 运行跟踪 ({len(frames)} 帧)...")
    sp, ep = {}, {}
    s_total, e_total, e_filtered = 0, 0, 0

    for fi, frame in enumerate(frames):
        # 单模型
        ds = vs_single.process_frame(frame)
        sp[fi] = [{'id': d.get('id', -1), 'bbox': d['bbox'], 
                   'confidence': d.get('confidence', 0)} for d in ds]
        s_total += len(ds)

        # 多模型融合（原始）
        de_raw = vs_ensemble.process_frame(frame)
        e_total += len(de_raw)

        # 共识过滤
        if APPLY_CONSENSUS_FILTER:
            de = consensus_filter(de_raw, min_models=2, min_conf_single=0.25)
        else:
            de = de_raw
        e_filtered += len(de)
        ep[fi] = [{'id': d.get('id', -1), 'bbox': d['bbox'],
                   'confidence': d.get('confidence', 0)} for d in de]

        if fi < 5 or fi % 20 == 0:
            print(f"  Frame {fi:3d}: GT={len(gt.get(fi,[])):2d}  "
                  f"单模型={len(ds):2d}  融合原始={len(de_raw):2d}  融合过滤={len(de):2d}")

    print(f"\n[4/5] MOT 评估...")
    sum_s = evaluate_mot(gt, sp, len(frames))
    sum_e = evaluate_mot(gt, ep, len(frames))

    if sum_s.empty or sum_e.empty:
        print("评估失败")
        return

    row_s = sum_s.iloc[0]
    row_e = sum_e.iloc[0]

    print(f"\n{'指标':<16} {'单模型':>10} {'融合+过滤':>10} {'变化':>10} {'结论':>8}")
    print("-" * 56)
    for name, key in [('MOTA','mota'),('IDF1','idf1'),('Recall','recall'),
                       ('Precision','precision'),('IDP','idp'),('IDR','idr'),
                       ('ID Sw','num_switches'),('FP','num_false_positives'),
                       ('FN','num_misses')]:
        v1 = float(row_s.get(key, 0))
        v2 = float(row_e.get(key, 0))
        diff = v2 - v1
        if key in ['num_switches','num_false_positives','num_misses']:
            better = '✅' if diff <= 0 else '⚠️'
            print(f"{name:<16} {int(v1):>10} {int(v2):>10} {int(diff):>+10d} {better:>8}")
        else:
            better = '✅' if diff >= 0 else '⚠️'
            print(f"{name:<16} {v1:>10.3f} {v2:>10.3f} {diff:>+10.3f} {better:>8}")

    mota_d = (float(row_e.get('mota',0)) - float(row_s.get('mota',0))) * 100
    idf1_d = (float(row_e.get('idf1',0)) - float(row_s.get('idf1',0))) * 100

    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    print(f"  总检测: 单模型={s_total}  融合原始={e_total}  融合过滤={e_filtered}")
    print(f"  过滤移除: {e_total - e_filtered} 个")
    print(f"  MOTA 变化: {mota_d:+.1f} pp")
    print(f"  IDF1 变化: {idf1_d:+.1f} pp")
    
    if mota_d > 0.5:
        print(f"\n  ✅ 共识过滤后融合 MOTA 超越单模型 {mota_d:.1f} pp")
    elif mota_d > 0:
        print(f"\n  ➖ 融合略优于单模型 (+{mota_d:.1f} pp)")
    else:
        print(f"\n  ⚠️ 需调整过滤参数")

    # 保存
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": f"visdrone_{VISDRONE_SEQ}",
        "num_frames": len(frames),
        "total_gt": sum(len(v) for v in gt.values()),
        "single_detections": s_total,
        "ensemble_raw": e_total,
        "ensemble_filtered": e_filtered,
        "filter_removed": e_total - e_filtered,
        "single_metrics": {'mota': float(row_s.get('mota',0)), 'idf1': float(row_s.get('idf1',0))},
        "ensemble_metrics": {'mota': float(row_e.get('mota',0)), 'idf1': float(row_e.get('idf1',0))},
        "mota_improvement_ppt": round(mota_d, 1),
        "idf1_improvement_ppt": round(idf1_d, 1),
    }
    path = os.path.join(config.OUTPUT_DIR, "real_mot_benchmark_v2.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n📁 {path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

