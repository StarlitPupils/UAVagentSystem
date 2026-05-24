# E:/UAVagent/benchmark_v14_clean.py (UAVagent 1.4)
"""1.4 干净基准 — 使用 v14 共识过滤 + 1.3 成熟参数 + IoU=0.50"""
import sys, os, cv2, json, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import config
from core.vision_system import VisionSystem
from core.visdrone_loader import VisDroneLoader
import motmetrics as mm

VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 100

# 1.4 关键参数
config.ADAPTIVE_THRESHOLD = False
config.DETECTION_CONFIDENCE = 0.28
config.ENSEMBLE_IOU_THR = 0.50      # 1.3 文档记载 100帧最优值
config.PREPROCESSING_ENABLED = False

def load_gt(root, seq, mf):
    gt = {}
    for an in [f"{seq}.txt", os.path.join(seq, "gt.txt")]:
        ap = os.path.join(root, "annotations", an)
        if os.path.isfile(ap):
            with open(ap) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) < 8: continue
                    frame = int(p[0]) - 1
                    if frame >= mf: continue
                    cls = int(p[7]) if len(p) > 7 else 0
                    if cls in [0, 11]: continue
                    gt.setdefault(frame, []).append({
                        "id": int(p[1]),
                        "bbox": [float(p[2]), float(p[3]), float(p[4]), float(p[5])],
                        "class": cls,
                    })
            break
    return gt

def evaluate_mot(gt_data, pred_data, n_frames):
    acc = mm.MOTAccumulator(auto_id=True)
    for fi in range(n_frames):
        g = gt_data.get(fi, [])
        p = pred_data.get(fi, [])
        if not g and not p: continue
        gt_ids = [gg["id"] for gg in g]
        pred_ids = [pp["id"] for pp in p]
        if g and p:
            ious = np.zeros((len(g), len(p)))
            for i, gg in enumerate(g):
                gx, gy, gw, gh = gg["bbox"]
                for j, pp in enumerate(p):
                    px, py, pw, ph = pp["bbox"]
                    if pw <= 0 or ph <= 0: continue
                    ix1, iy1 = max(gx, px), max(gy, py)
                    ix2, iy2 = min(gx + gw, px + pw), min(gy + gh, py + ph)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = gw * gh + pw * ph - inter
                    ious[i, j] = inter / (union + 1e-8)
        else:
            ious = np.zeros((len(g), len(p)))
        acc.update(gt_ids, pred_ids, ious)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="eval")

def main():
    print("=" * 70)
    print("UAVagent 1.4 干净基准 (v14 共识过滤 + 1.3 参数)")
    print(f"conf={config.DETECTION_CONFIDENCE}, IoU={config.ENSEMBLE_IOU_THR}, "
          f"filter=v14 (primary=0.55, single=0.70)")
    print("=" * 70)

    config.setup_session()

    print(f"\n[1/4] 加载 VisDrone ({MAX_FRAMES}帧)...")
    loader = VisDroneLoader(VISDRONE_ROOT, VISDRONE_SEQ)
    gt_data = load_gt(VISDRONE_ROOT, VISDRONE_SEQ, MAX_FRAMES)
    total_gt = sum(len(v) for v in gt_data.values())
    print(f"  GT: {total_gt}")

    print(f"\n[2/4] 初始化...")
    vs_single = VisionSystem(device="cuda", use_ensemble=False)
    vs_single.adaptive_threshold = False
    
    vs_ensemble = VisionSystem(device="cuda", use_ensemble=True)
    vs_ensemble.adaptive_threshold = False
    
    print(f"  单模型: {vs_single.get_stats()['model_names'][0]}")
    print(f"  融合: {vs_ensemble.get_stats()['num_models']} models")

    print(f"\n[3/4] 运行跟踪 ({MAX_FRAMES}帧)...")
    sp, ep = {}, {}
    s_total, e_total = 0, 0

    for fi in range(MAX_FRAMES):
        frame = loader.get_next_frame()
        if frame is None: break

        ds = vs_single.process_frame(frame)
        sp[fi] = [{"id": d.get("id", -1), "bbox": d.get("bbox", [0,0,0,0]),
                    "confidence": d.get("confidence", 0)} for d in ds]
        s_total += len(ds)

        de = vs_ensemble.process_frame(frame)
        ep[fi] = [{"id": d.get("id", -1), "bbox": d.get("bbox", [0,0,0,0]),
                    "confidence": d.get("confidence", 0)} for d in de]
        e_total += len(de)

        if fi % 20 == 0:
            print(f"  Frame {fi:3d}: GT={len(gt_data.get(fi,[])):2d}  "
                  f"Single={len(ds):2d} Ensemble={len(de):2d}")

    print(f"\n  单模型: {s_total} dets | 融合: {e_total} dets")

    print(f"\n[4/4] MOT 评估...")
    sum_s = evaluate_mot(gt_data, sp, MAX_FRAMES)
    sum_e = evaluate_mot(gt_data, ep, MAX_FRAMES)

    if not sum_s.empty and not sum_e.empty:
        row_s = sum_s.iloc[0]
        row_e = sum_e.iloc[0]

        print(f"\n{'Metric':<16} {'v14 Single':>10} {'v14 Ensemble':>12} {'Change':>10}")
        print("-" * 50)
        for name, key in [
            ("MOTA", "mota"), ("IDF1", "idf1"), ("Recall", "recall"),
            ("Precision", "precision"), ("ID Sw", "num_switches"),
            ("FP", "num_false_positives"), ("FN", "num_misses"),
        ]:
            v1 = float(row_s.get(key, 0))
            v2 = float(row_e.get(key, 0))
            diff = v2 - v1
            if key in ["num_switches", "num_false_positives", "num_misses"]:
                print(f"{name:<16} {int(v1):>10} {int(v2):>12} {int(diff):>+10}")
            else:
                print(f"{name:<16} {v1:>10.3f} {v2:>12.3f} {diff:>+10.3f}")

        mota_change = (float(row_e.get("mota", 0)) - float(row_s.get("mota", 0))) * 100
        print(f"\n  MOTA change: {mota_change:+.1f} pp")

    # 保存结果
    result = {
        "version": "1.4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"conf": 0.28, "iou": 0.50, "adaptive": False,
                   "filter": "v14 (primary=0.55, 2model=0.35, single=0.70)"},
        "single_detections": s_total,
        "ensemble_detections": e_total,
        "single_metrics": {
            "mota": float(row_s.get("mota", 0)), "idf1": float(row_s.get("idf1", 0)),
            "fp": int(row_s.get("num_false_positives", 0)),
            "fn": int(row_s.get("num_misses", 0)),
        },
        "ensemble_metrics": {
            "mota": float(row_e.get("mota", 0)), "idf1": float(row_e.get("idf1", 0)),
            "fp": int(row_e.get("num_false_positives", 0)),
            "fn": int(row_e.get("num_misses", 0)),
        },
    }
    result_path = os.path.join(config.OUTPUT_DIR, "benchmark_v14_clean.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result: {result_path}")

if __name__ == "__main__":
    main()