# E:/UAVagent/benchmark_v13_full.py (UAVagent 1.3)
"""UAVagent 1.3 完整基准测试 - 微调模型 + OSNet ReID"""
import sys, os, cv2, json, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import config
from core.vision_system import VisionSystem
from core.visdrone_loader import VisDroneLoader
from evaluation.temporal_visualizer import TemporalVisualizer
import motmetrics as mm

VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 100


def load_gt(root, seq, mf):
    gt = {}
    for an in [f"{seq}.txt", os.path.join(seq, "gt.txt")]:
        ap = os.path.join(root, "annotations", an)
        if os.path.isfile(ap):
            with open(ap) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) < 8:
                        continue
                    frame = int(p[0]) - 1
                    if frame >= mf:
                        continue
                    cls = int(p[7]) if len(p) > 7 else 0
                    if cls in [0, 11]:
                        continue
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
        if not g and not p:
            continue
        if g and p:
            ious = np.zeros((len(g), len(p)))
            for i, gg in enumerate(g):
                gx, gy, gw, gh = gg["bbox"]
                for j, pp in enumerate(p):
                    px, py, pw, ph = pp["bbox"]
                    ix1, iy1 = max(gx, px), max(gy, py)
                    ix2, iy2 = min(gx + gw, px + pw), min(gy + gh, py + ph)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = gw * gh + pw * ph - inter
                    ious[i, j] = inter / (union + 1e-8)
        else:
            ious = np.zeros((len(g), len(p)))
        acc.update([gg["id"] for gg in g], [pp["id"] for pp in p], ious)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="eval")


def main():
    print("=" * 70)
    print("UAVagent 1.3 完整基准测试 (微调模型 + OSNet ReID)")
    print("=" * 70)

    config.setup_session()
    config.DETECTION_CONFIDENCE = 0.28
    config.ENSEMBLE_IOU_THR = 0.50

    # 检查 ReID 后端
    from core.tracking.reid_features import reid_extractor
    reid_info = reid_extractor.get_backend_info()
    print(f"\n[ReID] 后端: {reid_info['backend']} (dim={reid_info['feature_dim']})")

    # 加载数据
    print(f"\n[1/4] 加载 VisDrone ({MAX_FRAMES}帧)...")
    loader = VisDroneLoader(VISDRONE_ROOT, VISDRONE_SEQ)
    gt_data = load_gt(VISDRONE_ROOT, VISDRONE_SEQ, MAX_FRAMES)
    print(f"  GT 目标: {sum(len(v) for v in gt_data.values())}")

    # 初始化视觉系统
    print(f"\n[2/4] 初始化...")
    vs_single = VisionSystem(device="cuda", use_ensemble=False)
    vs_ensemble = VisionSystem(device="cuda", use_ensemble=True)

    stats_s = vs_single.get_stats()
    stats_e = vs_ensemble.get_stats()
    print(f"  单模型: {stats_s['model_names'][0]}")
    print(f"  融合: {stats_e['num_models']} 模型")

    # 运行跟踪
    print(f"\n[3/4] 运行跟踪...")
    sp, ep = {}, {}
    s_total, e_total = 0, 0

    viz = TemporalVisualizer(
        output_dir=os.path.join(config.OUTPUT_DIR, "qualitative_v13"),
        show_gt=True,
    )

    frames_for_viz = []
    for fi in range(MAX_FRAMES):
        frame = loader.get_next_frame()
        if frame is None:
            break

        gt = gt_data.get(fi, [])

        ds = vs_single.process_frame(frame)
        sp[fi] = [{
            "id": d.get("id", -1),
            "bbox": d["bbox"],
            "confidence": d.get("confidence", 0),
        } for d in ds]
        s_total += len(ds)

        de = vs_ensemble.process_frame(frame)
        ep[fi] = [{
            "id": d.get("id", -1),
            "bbox": d["bbox"],
            "confidence": d.get("confidence", 0),
        } for d in de]
        e_total += len(de)

        if fi in [0, 5, 10, 15, 20, 25, 29]:
            vis_frame = viz.draw_frame(
                frame, de, frame_idx=fi, ground_truth=gt,
                title=f"Frame {fi} - 1.3 Ensemble"
            )
            viz.save_frame(vis_frame, fi, "v13_ensemble")
            frames_for_viz.append(vis_frame)

        if fi % 5 == 0:
            print(f"  Frame {fi:3d}: GT={len(gt):2d}  微调单模型={len(ds):2d}  融合={len(de):2d}")

    print(f"\n  单模型总检测: {s_total} | 融合总检测: {e_total}")

    # MOT 评估
    print(f"\n[4/4] MOT 评估...")
    sum_s = evaluate_mot(gt_data, sp, MAX_FRAMES)
    sum_e = evaluate_mot(gt_data, ep, MAX_FRAMES)

    if not sum_s.empty and not sum_e.empty:
        row_s = sum_s.iloc[0]
        row_e = sum_e.iloc[0]

        print(f"\n{'指标':<16} {'微调单模型':>10} {'1.3融合':>10} {'变化':>10}")
        print("-" * 48)
        for name, key in [
            ("MOTA", "mota"), ("IDF1", "idf1"), ("Recall", "recall"),
            ("Precision", "precision"), ("ID Sw", "num_switches"),
            ("FP", "num_false_positives"), ("FN", "num_misses"),
        ]:
            v1 = float(row_s.get(key, 0))
            v2 = float(row_e.get(key, 0))
            diff = v2 - v1
            if key in ["num_switches", "num_false_positives", "num_misses"]:
                print(f"{name:<16} {int(v1):>10} {int(v2):>10} {int(diff):>+10d}")
            else:
                print(f"{name:<16} {v1:>10.3f} {v2:>10.3f} {diff:>+10.3f}")

        mota_change = (float(row_e.get("mota", 0)) - float(row_s.get("mota", 0))) * 100
        print(f"\n  MOTA 变化: {mota_change:+.1f} pp")

    # 时间序列长条图
    if frames_for_viz:
        strip_path = os.path.join(config.OUTPUT_DIR, "qualitative_v13", "v13_timeline.jpg")
        viz.generate_timeline_strip(frames_for_viz, step=5, max_frames=8, save_path=strip_path)

    # 保存结果
    result = {
        "version": "1.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_frames": MAX_FRAMES,
        "reid_backend": reid_info,
        "single_model": stats_s["model_names"][0],
        "single_detections": s_total,
        "ensemble_detections": e_total,
        "single_metrics": {
            "mota": float(row_s.get("mota", 0)),
            "idf1": float(row_s.get("idf1", 0)),
        },
        "ensemble_metrics": {
            "mota": float(row_e.get("mota", 0)),
            "idf1": float(row_e.get("idf1", 0)),
        },
    }

    result_path = os.path.join(config.OUTPUT_DIR, "benchmark_v13.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nOK 1.3 基准完成: {result_path}")
    print(f"   效果图: {config.OUTPUT_DIR}/qualitative_v13/")


if __name__ == "__main__":
    main()