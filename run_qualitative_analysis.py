# E:/UAVagent/run_qualitative_analysis.py (UAVagent 1.3)
"""一键定性分析：生成30帧 VisDrone 检测/跟踪效果图"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import json
import time
from config.settings import config
from core.vision_system import VisionSystem
from core.visdrone_loader import VisDroneLoader
from evaluation.temporal_visualizer import TemporalVisualizer

def main():
    print("=" * 70)
    print("UAVagent 1.3 定性分析 - 时间序列检测跟踪效果图")
    print("=" * 70)

    # 配置
    config.setup_session()
    config.DETECTION_CONFIDENCE = 0.25
    config.ENSEMBLE_IOU_THR = 0.50

    VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
    VISDRONE_SEQ = "uav0000086_00000_v"
    MAX_FRAMES = 30

    output_dir = os.path.join(config.OUTPUT_DIR, "qualitative_analysis")
    visualizer = TemporalVisualizer(output_dir=output_dir, show_gt=True)

    print(f"\n[1/4] 加载数据集: {VISDRONE_SEQ}")
    try:
        loader = VisDroneLoader(VISDRONE_ROOT, VISDRONE_SEQ)
        print(f"  ✅ {loader.get_frame_count()} 帧可用")
    except Exception as e:
        print(f"  ❌ 数据集加载失败: {e}")
        print("  提示: 请确保 VisDrone 数据集在 E:/datasets/VisDrone/VisDrone2019-MOT-val")
        return

    print(f"\n[2/4] 初始化视觉系统 (5模型融合)...")
    vs_ensemble = VisionSystem(device="cpu", use_ensemble=True)
    vs_single = VisionSystem(device="cpu", use_ensemble=False)
    print(f"  融合: {vs_ensemble.get_stats()['num_models']} models")
    print(f"  单模型: {vs_single.get_stats()['model_names'][0]}")

    print(f"\n[3/4] 运行检测跟踪 ({MAX_FRAMES} 帧)...")
    frame_data_single = []
    frame_data_ensemble = []
    visualizer.reset_track_history()

    for i in range(MAX_FRAMES):
        frame = loader.get_next_frame()
        if frame is None:
            break

        # 单模型
        dets_single = vs_single.process_frame(frame)
        gt = loader.get_ground_truth(i + 1)

        # 融合模型
        dets_ensemble = vs_ensemble.process_frame(frame)

        # 可视化 - 单模型
        vis_single = visualizer.draw_frame(
            frame, dets_single, frame_idx=i,
            ground_truth=gt,
            title=f"Frame {i} - Single Model (YOLO11x)"
        )
        visualizer.save_frame(vis_single, i, "single")

        # 可视化 - 融合模型
        vis_ensemble = visualizer.draw_frame(
            frame, dets_ensemble, frame_idx=i,
            ground_truth=gt,
            title=f"Frame {i} - 5-Model Ensemble"
        )
        visualizer.save_frame(vis_ensemble, i, "ensemble")

        frame_data_single.append({
            'image': frame, 'detections': dets_single, 'ground_truth': gt,
        })
        frame_data_ensemble.append({
            'image': frame, 'detections': dets_ensemble, 'ground_truth': gt,
        })

        if i % 5 == 0:
            print(f"  Frame {i:3d}: 单模型={len(dets_single):3d}  融合={len(dets_ensemble):3d}  GT={len(gt):2d}")

    print(f"\n[4/4] 生成分析报告...")

    # 对比网格图
    single_vis = []
    ensemble_vis = []
    for i in range(len(frame_data_ensemble)):
        sv = visualizer.draw_frame(
            frame_data_single[i]['image'],
            frame_data_single[i]['detections'],
            frame_idx=i,
            ground_truth=frame_data_single[i].get('ground_truth'),
            title=f"Single"
        )
        ev = visualizer.draw_frame(
            frame_data_ensemble[i]['image'],
            frame_data_ensemble[i]['detections'],
            frame_idx=i,
            ground_truth=frame_data_ensemble[i].get('ground_truth'),
            title=f"Ensemble"
        )
        single_vis.append(sv)
        ensemble_vis.append(ev)

    # 生成对比网格
    grid_path = os.path.join(output_dir, "comparison", "single_vs_ensemble_grid.jpg")
    visualizer.generate_comparison_grid(single_vis, ensemble_vis, save_path=grid_path)
    print(f"  ✅ 对比网格: {grid_path}")

    # 生成时间序列长条图
    strip_path = os.path.join(output_dir, "ensemble_timeline.jpg")
    visualizer.generate_timeline_strip(ensemble_vis, step=3, max_frames=15, save_path=strip_path)
    print(f"  ✅ 时间序列长条: {strip_path}")

    # 生成逐帧指标图
    metrics = []
    for i in range(len(frame_data_ensemble)):
        de = frame_data_ensemble[i].get('detections', [])
        gt = frame_data_ensemble[i].get('ground_truth', [])
        metrics.append({
            "frame": i,
            "det_count": len(de),
            "gt_count": len(gt),
            "multi_2": sum(1 for d in de if d.get('num_models', 1) >= 2),
            "multi_3": sum(1 for d in de if d.get('num_models', 1) >= 3),
        })

    chart_path = os.path.join(output_dir, "per_frame_metrics.png")
    visualizer.generate_metric_chart(metrics, save_path=chart_path)
    print(f"  ✅ 逐帧指标: {chart_path}")

    # 保存完整报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence": VISDRONE_SEQ,
        "num_frames": len(frame_data_ensemble),
        "total_detections_single": sum(len(fd['detections']) for fd in frame_data_single),
        "total_detections_ensemble": sum(len(fd['detections']) for fd in frame_data_ensemble),
        "outputs": {
            "comparison_grid": os.path.relpath(grid_path, output_dir),
            "timeline_strip": os.path.relpath(strip_path, output_dir),
            "metrics_chart": os.path.relpath(chart_path, output_dir),
        }
    }

    report_path = os.path.join(output_dir, "qualitative_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✅ 定性分析完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   报告文件: {report_path}")
    print(f"\n   生成的文件:")
    print(f"   - 单帧效果图: {output_dir}/frames/single_*.jpg")
    print(f"   - 融合效果图: {output_dir}/frames/ensemble_*.jpg")
    print(f"   - 对比网格图: {grid_path}")
    print(f"   - 时间序列长条: {strip_path}")
    print(f"   - 指标图表: {chart_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
