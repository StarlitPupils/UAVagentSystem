import motmetrics as mm
import numpy as np
import os
import sys
import json
from datetime import datetime

def load_mot_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            data.setdefault(frame, []).append({'id': obj_id, 'bbox': [x, y, w, h], 'conf': conf})
    return data

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1, area2 = w1*h1, w2*h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate(gt_file, pred_file):
    gt = load_mot_file(gt_file)
    pred = load_mot_file(pred_file)
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt.keys()) | set(pred.keys())):
        gt_boxes = gt.get(frame, [])
        pred_boxes = pred.get(frame, [])
        if not gt_boxes and not pred_boxes:
            continue
        gt_ids = [g['id'] for g in gt_boxes]
        pred_ids = [p['id'] for p in pred_boxes]
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, g in enumerate(gt_boxes):
            for j, p in enumerate(pred_boxes):
                iou_matrix[i, j] = compute_iou(g['bbox'], p['bbox'])
        acc.update(gt_ids, pred_ids, iou_matrix)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='UAVagent')
    print(summary)
    return summary

def save_metrics(summary, output_dir):
    """将评估结果保存为 JSON 文件"""
    if summary is None or summary.empty:
        return
    record = summary.iloc[0].to_dict()
    # 提取关键指标
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "idf1": float(record.get('idf1', 0)),
        "idp": float(record.get('idp', 0)),
        "idr": float(record.get('idr', 0)),
        "recall": float(record.get('recall', 0)),
        "precision": float(record.get('precision', 0)),
        "mota": float(record.get('mota', 0)),
        "motp": float(record.get('motp', 0)),
        "num_frames": int(record.get('num_frames', 0)),
        "num_objects": int(record.get('num_objects', 0)),
        "num_matches": int(record.get('num_matches', 0)),
        "num_switches": int(record.get('num_switches', 0)),
        "num_transfer": int(record.get('num_transfer', 0)),
        "num_ascend": int(record.get('num_ascend', 0)),
        "num_migrate": int(record.get('num_migrate', 0)),
    }
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tracking_metrics.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ 评估结果已保存至: {save_path}")

def find_latest_session_dir(base_dir: str = "output"):
    """查找最新的 session_* 目录"""
    import glob
    candidates = glob.glob(os.path.join(base_dir, "session_*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        gt_file = sys.argv[1]
        pred_file = sys.argv[2]
    else:
        # 默认路径：自动查找最新会话目录下的跟踪结果
        base_dir = "output"
        session_dir = find_latest_session_dir(base_dir)
        if session_dir is None:
            print("错误：未找到任何会话目录，且未指定参数")
            print("用法: python evaluate_tracking.py <gt_file> <pred_file>")
            sys.exit(1)
        pred_file = os.path.join(session_dir, "tracking", "uav0000086_00000_v.txt")
        gt_file = r"E:\datasets\VisDrone\VisDrone2019-MOT-val\annotations\uav0000086_00000_v.txt"
        print(f"自动使用会话目录: {session_dir}")
        print(f"真值文件: {gt_file}")
        print(f"预测文件: {pred_file}")

    if not os.path.exists(pred_file):
        print(f"错误：跟踪结果文件不存在：{pred_file}")
        print("请先运行 'python run_visdrone_real.py' 生成跟踪结果。")
        sys.exit(1)
    if not os.path.exists(gt_file):
        print(f"错误：真值文件不存在：{gt_file}")
        sys.exit(1)

    summary = evaluate(gt_file, pred_file)

    # 保存结果到会话目录
    session_dir = os.path.dirname(os.path.dirname(pred_file)) if 'pred_file' in locals() else "output"
    save_metrics(summary, session_dir)
