# E:/UAVagent/benchmark_real_mot.py (v1.2 修复版)
"""真实数据集 MOT 基准测试 - VisDrone"""
import sys, os, cv2, json, numpy as np, time, glob
sys.path.insert(0, "E:/UAVagent")
from config.settings import config
from core.vision_system import VisionSystem
import motmetrics as mm

VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 30

def find_available_dataset():
    seq_path = os.path.join(VISDRONE_ROOT, "sequences", VISDRONE_SEQ)
    ann_path = os.path.join(VISDRONE_ROOT, "annotations", f"{VISDRONE_SEQ}.txt")
    if os.path.isdir(seq_path) and os.path.isfile(ann_path):
        return "visdrone", VISDRONE_ROOT, VISDRONE_SEQ
    ann_path2 = os.path.join(VISDRONE_ROOT, "annotations", VISDRONE_SEQ, "gt.txt")
    if os.path.isdir(seq_path) and os.path.isfile(ann_path2):
        return "visdrone", VISDRONE_ROOT, VISDRONE_SEQ
    return None, None, None

def load_ground_truth(root, seq):
    gt_data = {}
    for ann_name in [f"{seq}.txt", os.path.join(seq, "gt.txt")]:
        ann_path = os.path.join(root, "annotations", ann_name)
        if os.path.isfile(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8: continue
                    frame = int(parts[0]) - 1
                    if frame >= MAX_FRAMES: continue
                    cls = int(parts[7]) if len(parts) > 7 else 0
                    if cls in [0, 11]: continue
                    gt_data.setdefault(frame, []).append({
                        'id': int(parts[1]), 
                        'bbox': [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
                        'class': cls
                    })
            break
    return gt_data

def load_frames(root, seq):
    seq_path = os.path.join(root, "sequences", seq)
    frames = []
    if os.path.isdir(seq_path):
        imgs = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(('.jpg','.png'))])
        for fname in imgs[:MAX_FRAMES]:
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
    print("UAVagent 1.2 真实数据集 MOT 基准测试")
    print("=" * 70)

    ds_type, ds_root, ds_seq = find_available_dataset()
    if ds_type is None:
        print("❌ 未找到真实数据集！")
        return
    print(f"✅ 找到数据集: {ds_type.upper()} ({ds_seq})")

    frames = load_frames(ds_root, ds_seq)
    gt_data = load_ground_truth(ds_root, ds_seq)
    n_frames = len(frames)
    total_gt = sum(len(v) for v in gt_data.values())
    print(f"  图像帧: {n_frames}  标注目标数: {total_gt}")

    # 合理阈值（真实场景）
    config.DETECTION_CONFIDENCE = 0.25
    
    vs_single = VisionSystem(device="cpu", use_ensemble=False)
    vs_ensemble = VisionSystem(device="cpu", use_ensemble=True)
    
    ens_stats = vs_ensemble.get_stats()
    print(f"  单模型: {vs_single.get_stats()['model_names'][0]}")
    print(f"  融合模式: {'✅ ' + str(ens_stats['num_models']) + ' 模型' if ens_stats['ensemble_mode'] else '⚠️ 单模型降级'}")

    single_pred, ensemble_pred = {}, {}
    single_det_total, ensemble_det_total = 0, 0

    for frame_idx, frame in enumerate(frames):
        dets_s = vs_single.process_frame(frame)
        single_pred[frame_idx] = [
            {'id': d.get('id', -1), 'bbox': d['bbox'], 'confidence': d.get('confidence', 0)}
            for d in dets_s
        ]
        single_det_total += len(dets_s)

        dets_e = vs_ensemble.process_frame(frame)
        ensemble_pred[frame_idx] = [
            {'id': d.get('id', -1), 'bbox': d['bbox'], 'confidence': d.get('confidence', 0)}
            for d in dets_e
        ]
        ensemble_det_total += len(dets_e)

        if frame_idx < 3 or frame_idx % 10 == 0:
            gt_n = len(gt_data.get(frame_idx, []))
            print(f"  Frame {frame_idx:3d}: GT={gt_n:2d}  单模型={len(dets_s):2d}  融合={len(dets_e):2d}")

    if total_gt > 0:
        sum_single = evaluate_mot(gt_data, single_pred, n_frames)
        sum_ensemble = evaluate_mot(gt_data, ensemble_pred, n_frames)

        if not sum_single.empty and not sum_ensemble.empty:
            row_s = sum_single.iloc[0]
            row_e = sum_ensemble.iloc[0]

            print(f"\n{'指标':<16} {'单模型':>10} {'多模型融合':>10} {'变化':>10} {'结论':>8}")
            print("-" * 56)
            for name, key in [('MOTA','mota'),('IDF1','idf1'),('Recall','recall'),
                               ('Precision','precision'),('ID Sw','num_switches'),
                               ('FP','num_false_positives'),('FN','num_misses')]:
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
            print(f"\n{'='*70}")
            print(f"  MOTA 变化: {mota_d:+.1f} 百分点")
            print(f"  单模型 MOTA: {float(row_s.get('mota',0)):.3f}")
            print(f"  融合 MOTA: {float(row_e.get('mota',0)):.3f}")

    result_path = os.path.join(config.OUTPUT_DIR, "real_mot_benchmark.json")
    # ... (保存略)

if __name__ == "__main__":
    main()
