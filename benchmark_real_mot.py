# E:/UAVagent1.1/benchmark_real_mot.py
"""真实数据集 MOT 基准测试 - VisDrone / MOT17
定量证明：多模型融合 vs 单模型在真实场景下的精度提升
"""
import sys, os, cv2, json, numpy as np, time, glob
sys.path.insert(0, "E:/UAVagent1.1")

from config.settings import config
from core.vision_system import VisionSystem
from core.visdrone_loader import VisDroneLoader
import motmetrics as mm

# ============================================================
# 配置区
# ============================================================
VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MOT17_ROOT = "E:/datasets/MOT17"
MOT17_SEQ = "MOT17-02"
MAX_FRAMES = 30  # 为快速测试限制帧数（完整评估设为99999）

# ============================================================
# 数据集自动检测与加载
# ============================================================
def find_available_dataset():
    """自动寻找可用的真实数据集"""
    # 优先VisDrone
    seq_path = os.path.join(VISDRONE_ROOT, "sequences", VISDRONE_SEQ)
    ann_path = os.path.join(VISDRONE_ROOT, "annotations", f"{VISDRONE_SEQ}.txt")
    if os.path.isdir(seq_path) and os.path.isfile(ann_path):
        return "visdrone", VISDRONE_ROOT, VISDRONE_SEQ
    
    # 尝试VisDrone annotations在子目录
    ann_path2 = os.path.join(VISDRONE_ROOT, "annotations", VISDRONE_SEQ, "gt.txt")
    if os.path.isdir(seq_path) and os.path.isfile(ann_path2):
        return "visdrone", VISDRONE_ROOT, VISDRONE_SEQ
    
    # MOT17
    mot_seq = os.path.join(MOT17_ROOT, "train", MOT17_SEQ)
    mot_gt = os.path.join(MOT17_ROOT, "train", MOT17_SEQ, "gt", "gt.txt")
    if os.path.isdir(mot_seq) and os.path.isfile(mot_gt):
        return "mot17", MOT17_ROOT, MOT17_SEQ
    
    return None, None, None


def load_ground_truth(dataset_type, root, seq):
    """加载 Ground Truth 标注"""
    gt_data = {}  # frame_idx -> list of {id, bbox, class}
    
    if dataset_type == "visdrone":
        # VisDrone格式: frame, id, x, y, w, h, conf, cls, ...
        ann_path = os.path.join(root, "annotations", f"{seq}.txt")
        if not os.path.isfile(ann_path):
            ann_path = os.path.join(root, "annotations", seq, "gt.txt")
        
        if os.path.isfile(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    frame = int(parts[0]) - 1  # VisDrone从1开始
                    if frame >= MAX_FRAMES:
                        continue
                    obj_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    cls = int(parts[7]) if len(parts) > 7 else 0
                    if cls in [0, 11]:  # 忽略 ignored 和 others
                        continue
                    gt_data.setdefault(frame, []).append({
                        'id': obj_id, 'bbox': [x, y, w, h], 'class': cls
                    })
    
    elif dataset_type == "mot17":
        gt_path = os.path.join(root, "train", seq, "gt", "gt.txt")
        if os.path.isfile(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    frame = int(parts[0]) - 1
                    if frame >= MAX_FRAMES:
                        continue
                    obj_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    cls = int(parts[7]) if len(parts) > 7 else 1
                    # MOT17: cls=1 pedestrian, 其他忽略
                    if cls != 1:
                        continue
                    visible = float(parts[8]) if len(parts) > 8 else 1.0
                    if visible < 0.3:  # 几乎不可见，跳过
                        continue
                    gt_data.setdefault(frame, []).append({
                        'id': obj_id, 'bbox': [x, y, w, h], 'class': cls
                    })
    
    return gt_data


def load_frames(dataset_type, root, seq):
    """加载图像帧"""
    frames = []
    
    if dataset_type == "visdrone":
        seq_path = os.path.join(root, "sequences", seq)
        if os.path.isdir(seq_path):
            imgs = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(('.jpg', '.png'))])
            for fname in imgs[:MAX_FRAMES]:
                img = cv2.imread(os.path.join(seq_path, fname))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    elif dataset_type == "mot17":
        img_path = os.path.join(root, "train", seq, "img1")
        if os.path.isdir(img_path):
            imgs = sorted([f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png'))])
            for fname in imgs[:MAX_FRAMES]:
                img = cv2.imread(os.path.join(img_path, fname))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return frames


def evaluate_mot(gt_data, pred_data, n_frames):
    """MOT 评估"""
    acc = mm.MOTAccumulator(auto_id=True)
    
    for frame_idx in range(n_frames):
        gt = gt_data.get(frame_idx, [])
        pred = pred_data.get(frame_idx, [])
        
        if not gt and not pred:
            continue
        
        gt_ids = [g['id'] for g in gt]
        pred_ids = [p['id'] for p in pred]
        
        if gt and pred:
            ious = np.zeros((len(gt), len(pred)))
            for i, g in enumerate(gt):
                gx, gy, gw, gh = g['bbox']
                for j, p in enumerate(pred):
                    px, py, pw, ph = p['bbox']
                    ix1, iy1 = max(gx, px), max(gy, py)
                    ix2, iy2 = min(gx+gw, px+pw), min(gy+gh, py+ph)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    union = gw*gh + pw*ph - inter
                    ious[i, j] = inter / (union + 1e-8)
        else:
            ious = np.zeros((len(gt), len(pred)))
        
        acc.update(gt_ids, pred_ids, ious)
    
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='eval')
    return summary


def main():
    print("=" * 70)
    print("UAVagent 1.1 真实数据集 MOT 基准测试")
    print("=" * 70)
    
    # ---- 1. 检测可用数据集 ----
    print("\n[1/5] 检测可用数据集...")
    ds_type, ds_root, ds_seq = find_available_dataset()
    
    if ds_type is None:
        print("❌ 未找到真实数据集！")
        print(f"   尝试路径: VisDrone={VISDRONE_ROOT}")
        print(f"             MOT17={MOT17_ROOT}")
        print("\n   请下载数据集后重试：")
        print("   VisDrone: http://aiskyeye.com/download/")
        print("   MOT17: https://motchallenge.net/data/MOT17/")
        return
    
    print(f"✅ 找到数据集: {ds_type.upper()} ({ds_seq})")
    
    # ---- 2. 加载数据 ----
    print("\n[2/5] 加载图像帧和 Ground Truth...")
    frames = load_frames(ds_type, ds_root, ds_seq)
    gt_data = load_ground_truth(ds_type, ds_root, ds_seq)
    
    n_frames = len(frames)
    total_gt = sum(len(v) for v in gt_data.values())
    print(f"  图像帧: {n_frames}")
    print(f"  标注目标数: {total_gt}")
    
    if n_frames == 0:
        print("❌ 未加载到任何图像帧")
        return
    if total_gt == 0:
        print("⚠️ 未加载到 Ground Truth（将仅统计检测数）")
    
    # ---- 3. 加载检测系统 ----
    print("\n[3/5] 加载检测+跟踪系统...")
    config.DETECTION_CONFIDENCE = 0.25  # 恢复合理阈值
    vs_single = VisionSystem(device="cpu", use_ensemble=False)
    vs_ensemble = VisionSystem(device="cpu", use_ensemble=True)
    
    ens_stats = vs_ensemble.get_stats()
    print(f"  单模型: {vs_single.get_stats()['model_names'][0]}")
    print(f"  融合模式: {'✅ ' + str(ens_stats['num_models']) + ' 模型' if ens_stats['ensemble_mode'] else '⚠️ 单模型降级'}")
    
    # ---- 4. 运行跟踪 ----
    print(f"\n[4/5] 运行跟踪 ({n_frames} 帧)...")
    single_pred, ensemble_pred = {}, {}
    single_det_total, ensemble_det_total = 0, 0
    
    for frame_idx, frame in enumerate(frames):
        # 单模型
        dets_s = vs_single.process_frame(frame)
        single_pred[frame_idx] = [
            {'id': d.get('id', -1), 'bbox': d['bbox'], 'confidence': d.get('confidence', 0)}
            for d in dets_s
        ]
        single_det_total += len(dets_s)
        
        # 多模型融合
        dets_e = vs_ensemble.process_frame(frame)
        ensemble_pred[frame_idx] = [
            {'id': d.get('id', -1), 'bbox': d['bbox'], 'confidence': d.get('confidence', 0)}
            for d in dets_e
        ]
        ensemble_det_total += len(dets_e)
        
        if frame_idx < 3 or frame_idx % 5 == 0:
            gt_n = len(gt_data.get(frame_idx, []))
            print(f"  Frame {frame_idx:3d}: GT={gt_n:2d}  单模型={len(dets_s):2d}  融合={len(dets_e):2d}  "
                  f"{'✅' if len(dets_e) >= len(dets_s) else '➖'}")
    
    # ---- 5. MOT 评估 ----
    print(f"\n[5/5] MOT 精度评估...")
    
    if total_gt > 0:
        sum_single = evaluate_mot(gt_data, single_pred, n_frames)
        sum_ensemble = evaluate_mot(gt_data, ensemble_pred, n_frames)
        
        if not sum_single.empty and not sum_ensemble.empty:
            row_s = sum_single.iloc[0]
            row_e = sum_ensemble.iloc[0]
            
            print(f"\n{'指标':<16} {'单模型':>10} {'多模型融合':>10} {'变化':>10} {'结论':>8}")
            print("-" * 56)
            
            key_metrics = [
                ('MOTA', 'mota'), ('MOTP', 'motp'), ('IDF1', 'idf1'),
                ('Recall', 'recall'), ('Precision', 'precision'),
                ('IDP', 'idp'), ('IDR', 'idr'),
                ('ID Switches', 'num_switches'), ('FP', 'num_false_positives'),
                ('FN', 'num_misses')
            ]
            
            improvements = {}
            for name, key in key_metrics:
                v1 = float(row_s.get(key, 0))
                v2 = float(row_e.get(key, 0))
                diff = v2 - v1
                if key in ['num_switches', 'num_false_positives', 'num_misses']:
                    better = '✅' if diff <= 0 else '⚠️'
                    print(f"{name:<16} {int(v1):>10} {int(v2):>10} {int(diff):>+10d} {better:>8}")
                else:
                    better = '✅' if diff >= 0 else '⚠️'
                    print(f"{name:<16} {v1:>10.3f} {v2:>10.3f} {diff:>+10.3f} {better:>8}")
                improvements[key] = diff
            
            mota_change = (float(row_e.get('mota', 0)) - float(row_s.get('mota', 0))) * 100
            idf1_change = (float(row_e.get('idf1', 0)) - float(row_s.get('idf1', 0))) * 100
            
            print(f"\n{'='*70}")
            print("结论")
            print(f"{'='*70}")
            print(f"  数据集: {ds_type.upper()} {ds_seq}")
            print(f"  帧数: {n_frames}")
            print(f"  GT 目标数: {total_gt}")
            print(f"  总检测数: 单模型={single_det_total}  融合={ensemble_det_total}")
            print(f"  MOTA 变化: {mota_change:+.1f} 百分点")
            print(f"  IDF1 变化: {idf1_change:+.1f} 百分点")
            
            if mota_change > 3:
                print(f"\n  ✅ 在真实数据集上，多模型融合 MOTA 超越单模型 {mota_change:.0f}+ 百分点")
                print(f"     这充分证明了 UAVagent 在真实场景中的精度优势。")
            elif mota_change > 0:
                print(f"\n  ✅ 在真实数据集上，多模型融合 MOTA 略优于单模型 (+{mota_change:.1f}pp)")
            else:
                print(f"\n  ⚠️ 当前配置下融合未超越单模型，可能需要调参")
        else:
            print("⚠️ MOT 评估计算失败")
    else:
        print("⚠️ 无 Ground Truth，跳过 MOT 评估")
        print(f"  检测统计: 单模型={single_det_total}  融合={ensemble_det_total}")
    
    # 保存结果
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": f"{ds_type}_{ds_seq}",
        "num_frames": n_frames,
        "total_gt": total_gt,
        "single_detections": single_det_total,
        "ensemble_detections": ensemble_det_total,
        "models_used": ens_stats['model_names'],
        "ensemble_mode": ens_stats['ensemble_mode'],
    }
    if total_gt > 0:
        result["single_metrics"] = {
            'mota': float(row_s.get('mota', 0)), 'idf1': float(row_s.get('idf1', 0))
        }
        result["ensemble_metrics"] = {
            'mota': float(row_e.get('mota', 0)), 'idf1': float(row_e.get('idf1', 0))
        }
        result["mota_improvement_ppt"] = round(mota_change, 1)
        result["idf1_improvement_ppt"] = round(idf1_change, 1)
    
    result_path = os.path.join(config.OUTPUT_DIR, "real_mot_benchmark.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n📁 结果已保存: {result_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
