# E:/UAVagent/training/evaluate_finetuned.py (UAVagent 1.3)
"""微调前后模型性能对比评估"""
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from config.settings import config
from core.vision_system import VisionSystem
from core.visdrone_loader import VisDroneLoader
import motmetrics as mm


def evaluate_on_visdrone(model_path: str, seq_name: str = "uav0000086_00000_v",
                          max_frames: int = 30, device: str = "cpu") -> dict:
    """
    在VisDrone序列上评估指定模型
    """
    # 临时切换模型
    original_model = config.YOLO_MODEL_PATH
    original_name = config.YOLO_MODEL_NAME
    config.YOLO_MODEL_PATH = model_path
    config.YOLO_MODEL_NAME = os.path.basename(model_path).replace('.pt', '')
    
    vs = VisionSystem(device=device, use_ensemble=False)
    
    visdrone_root = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
    loader = VisDroneLoader(visdrone_root, seq_name)
    
    detections_per_frame = []
    total_dets = 0
    
    for i in range(max_frames):
        frame = loader.get_next_frame()
        if frame is None:
            break
        
        dets = vs.process_frame(frame)
        detections_per_frame.append({
            "frame": i,
            "count": len(dets),
            "dets": dets,
        })
        total_dets += len(dets)
    
    # 恢复原模型
    config.YOLO_MODEL_PATH = original_model
    config.YOLO_MODEL_NAME = original_name
    
    return {
        "model": os.path.basename(model_path),
        "frames": len(detections_per_frame),
        "total_detections": total_dets,
        "avg_per_frame": round(total_dets / max(len(detections_per_frame), 1), 1),
        "detections_per_frame": detections_per_frame,
    }


def compare_models(baseline_model: str = None, finetuned_model: str = None,
                   seq_name: str = "uav0000086_00000_v", max_frames: int = 30):
    """对比基线模型和微调模型"""
    
    if baseline_model is None:
        baseline_model = str(Path(__file__).resolve().parent.parent / "models" / "yolo11x.pt")
    if finetuned_model is None:
        # 查找最新的微调模型
        training_dir = Path("E:/UAVagent/output/training")
        best_models = list(training_dir.glob("**/weights/best.pt"))
        if best_models:
            finetuned_model = str(max(best_models, key=os.path.getmtime))
        else:
            print("❌ 未找到微调模型，请先运行训练")
            return
    
    print("=" * 70)
    print("微调前后模型性能对比")
    print("=" * 70)
    print(f"基线模型: {baseline_model}")
    print(f"微调模型: {finetuned_model}")
    print(f"测试序列: {seq_name} ({max_frames}帧)")
    
    # 评估基线
    print(f"\n[1/2] 评估基线模型 (YOLO11x 原始)...")
    baseline_result = evaluate_on_visdrone(baseline_model, seq_name, max_frames)
    print(f"  检测数: {baseline_result['total_detections']} "
          f"(平均 {baseline_result['avg_per_frame']}/帧)")
    
    # 评估微调
    print(f"\n[2/2] 评估微调模型 (YOLO11x VisDrone专训)...")
    finetuned_result = evaluate_on_visdrone(finetuned_model, seq_name, max_frames)
    print(f"  检测数: {finetuned_result['total_detections']} "
          f"(平均 {finetuned_result['avg_per_frame']}/帧)")
    
    # 对比
    delta = finetuned_result['total_detections'] - baseline_result['total_detections']
    delta_pct = (delta / max(baseline_result['total_detections'], 1)) * 100
    
    print(f"\n{'='*70}")
    print("对比结果")
    print(f"{'='*70}")
    print(f"{'指标':<20} {'基线':>10} {'微调':>10} {'变化':>10}")
    print(f"-" * 50)
    print(f"{'总检测数':<20} {baseline_result['total_detections']:>10} "
          f"{finetuned_result['total_detections']:>10} {delta:>+10d}")
    print(f"{'平均/帧':<20} {baseline_result['avg_per_frame']:>10.1f} "
          f"{finetuned_result['avg_per_frame']:>10.1f} {delta_pct:>+9.1f}%")
    
    if delta > 0:
        print(f"\n✅ 微调模型检测数提升 {delta_pct:+.1f}%")
    else:
        print(f"\n⚠️ 微调模型检测数变化 {delta_pct:+.1f}% (可能需要更多训练轮数)")
    
    # 保存结果
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_model": os.path.basename(baseline_model),
        "finetuned_model": os.path.basename(finetuned_model),
        "sequence": seq_name,
        "max_frames": max_frames,
        "baseline": {k: v for k, v in baseline_result.items() if k != 'detections_per_frame'},
        "finetuned": {k: v for k, v in finetuned_result.items() if k != 'detections_per_frame'},
        "delta": delta,
        "delta_pct": round(delta_pct, 1),
    }
    
    output_dir = Path("E:/UAVagent/output/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "finetune_comparison.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 对比结果已保存: {result_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=None, help="基线模型路径")
    parser.add_argument("--finetuned", type=str, default=None, help="微调模型路径")
    parser.add_argument("--frames", type=int, default=30, help="评估帧数")
    args = parser.parse_args()
    
    compare_models(args.baseline, args.finetuned, max_frames=args.frames)
