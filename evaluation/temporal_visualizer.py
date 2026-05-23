# E:/UAVagent/evaluation/temporal_visualizer.py (UAVagent 1.3)
"""时间序列检测与跟踪效果图生成器 - 定性分析核心模块
功能:
  1. 逐帧绘制检测框 (带类别标签/置信度/模型确认数)
  2. 绘制跟踪ID + 轨迹线 (连接历史位置)
  3. 添加时间戳/帧号标注
  4. 支持多帧拼接对比图 (前后版本对比)
  5. 生成视频/GIF动图
  6. 叠加 Ground Truth 对比 (绿色=TP, 红色=FP, 蓝色=FN)
"""
import cv2
import numpy as np
import os
import json
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================= 颜色映射 =======================
CLASS_COLORS = {
    0:  (0, 255, 0),     # pedestrian - 绿色
    1:  (255, 255, 0),   # people - 青色
    2:  (255, 0, 0),     # bicycle - 蓝色
    3:  (0, 0, 255),     # car - 红色
    4:  (255, 0, 255),   # van - 品红
    5:  (0, 255, 255),   # truck - 黄色
    6:  (128, 0, 128),   # tricycle - 紫色
    7:  (128, 128, 0),   # awning-tricycle
    8:  (0, 128, 128),   # bus
    9:  (128, 128, 128), # motor
    10: (255, 128, 0),   # others - 橙色
}

COLOR_CONSENSUS = {
    1: (80, 80, 255),    # 单模型 - 浅红
    2: (0, 200, 255),    # 2模型确认 - 橙黄
    3: (0, 255, 0),      # 3+模型确认 - 绿色
}

CLASS_NAMES = {
    0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car',
    4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle',
    8: 'bus', 9: 'motor', 10: 'others'
}


class TemporalVisualizer:
    """时间序列检测跟踪可视化器"""
    
    def __init__(self, output_dir: str = "output/qualitative", 
                 show_gt: bool = True, show_tracks: bool = True,
                 show_consensus: bool = True, font_scale: float = 0.5):
        self.output_dir = output_dir
        self.show_gt = show_gt
        self.show_tracks = show_tracks
        self.show_consensus = show_consensus
        self.font_scale = font_scale
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.max_history = 30  # 保留最近30帧的轨迹
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)
    
    def draw_frame(self, image: np.ndarray, detections: List[Dict],
                   frame_idx: int = 0, ground_truth: List[Dict] = None,
                   title: str = None) -> np.ndarray:
        """
        在单帧上绘制检测框+跟踪ID+轨迹线+时间戳
        
        Args:
            image: RGB图像 (H, W, 3)
            detections: 检测结果列表 [{"bbox":[cx,cy,w,h], "id":int, "class":int, 
                        "confidence":float, "num_models":int, "class_name":str}, ...]
            frame_idx: 帧编号
            ground_truth: GT标注 (可选)
            title: 帧标题 (可选)
        Returns:
            带标注的BGR图像 (可直接用cv2.imwrite保存)
        """
        # 转BGR用于OpenCV绘制
        if image.shape[-1] == 3 and image.dtype == np.uint8:
            vis = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        else:
            vis = image.copy()
            if vis.shape[-1] == 3:
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        h, w = vis.shape[:2]
        
        # ---- 时间戳 ----
        timestamp = f"Frame: {frame_idx:04d} | {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(vis, timestamp, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, timestamp, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # ---- 帧标题 ----
        if title:
            cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # ---- Ground Truth (半透明框) ----
        if self.show_gt and ground_truth:
            for gt in ground_truth:
                bbox = gt.get('bbox', [0, 0, 0, 0])
                x, y, bw, bh = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + bw), int(y + bh)
                # GT用虚线半透明框
                overlay = vis.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                cv2.putText(vis, f"GT:{gt.get('id','?')}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        # ---- 检测框 + 跟踪ID ----
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            cx, cy, bw, bh = bbox
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            
            cls_id = det.get('class', det.get('class_id', 3))
            conf = det.get('confidence', 0)
            tid = det.get('id', '?')
            num_models = det.get('num_models', 1)
            
            # 根据共识模型数选择颜色
            if self.show_consensus:
                color = COLOR_CONSENSUS.get(num_models, COLOR_CONSENSUS[1])
            else:
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            # 绘制检测框
            thickness = 2 if num_models >= 3 else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            class_name = det.get('class_name', CLASS_NAMES.get(cls_id, f'cls{cls_id}'))
            label = f"ID:{tid} {class_name} {conf:.2f}"
            if num_models >= 2:
                label += f" [{num_models}m]"
            
            # 标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                          self.font_scale, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(vis, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                       self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            # ---- 轨迹线 ----
            if self.show_tracks and tid is not None and tid != '?':
                tid_int = int(tid) if isinstance(tid, (int, float)) else hash(tid) % 10000
                if tid_int not in self.track_history:
                    self.track_history[tid_int] = []
                self.track_history[tid_int].append((cx, cy))
                if len(self.track_history[tid_int]) > self.max_history:
                    self.track_history[tid_int] = self.track_history[tid_int][-self.max_history:]
                
                # 绘制轨迹线
                pts = self.track_history[tid_int]
                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        alpha = 0.3 + 0.7 * (i / len(pts))
                        pt1 = (int(pts[i][0]), int(pts[i][1]))
                        pt2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                        faded_color = tuple(int(c * alpha) for c in color)
                        cv2.line(vis, pt1, pt2, faded_color, 1, cv2.LINE_AA)
        
        # ---- 统计信息 ----
        stats_y = 50
        n_gt = len(ground_truth) if ground_truth else 0
        cv2.putText(vis, f"Detections: {len(detections)} | GT: {n_gt}", 
                   (w - 280, stats_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # 多模型统计
        multi_2 = sum(1 for d in detections if d.get('num_models', 1) >= 2)
        multi_3 = sum(1 for d in detections if d.get('num_models', 1) >= 3)
        cv2.putText(vis, f"2+Models: {multi_2} | 3+Models: {multi_3}", 
                   (w - 280, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return vis
    
    def save_frame(self, image: np.ndarray, frame_idx: int, prefix: str = "frame"):
        """保存单帧图像"""
        path = os.path.join(self.output_dir, "frames", 
                           f"{prefix}_{frame_idx:04d}.jpg")
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
    
    def generate_comparison_grid(self, frames_single: List[np.ndarray],
                                  frames_ensemble: List[np.ndarray],
                                  frame_indices: List[int] = None,
                                  save_path: str = None) -> np.ndarray:
        """
        生成单模型 vs 融合模型的对比网格图
        
        Args:
            frames_single: 单模型检测帧列表
            frames_ensemble: 融合模型检测帧列表
            frame_indices: 要展示的帧索引 (默认取均匀5帧)
            save_path: 保存路径
        """
        n_frames = len(frames_single)
        if frame_indices is None:
            # 取均匀5帧
            if n_frames <= 5:
                frame_indices = list(range(n_frames))
            else:
                step = n_frames // 5
                frame_indices = [i * step for i in range(5)]
        
        n_cols = len(frame_indices)
        # 行: 单模型 / 融合模型 / GT
        n_rows = 2
        
        h, w = frames_single[0].shape[:2]
        grid = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
        
        for col, fi in enumerate(frame_indices):
            if fi < len(frames_single):
                grid[0:h, col*w:(col+1)*w] = cv2.resize(
                    frames_single[fi], (w, h))
            if fi < len(frames_ensemble):
                grid[h:2*h, col*w:(col+1)*w] = cv2.resize(
                    frames_ensemble[fi], (w, h))
        
        # 添加行标签
        cv2.putText(grid, "Single Model", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(grid, "Ensemble Fusion", (10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 添加列标签 (帧号)
        for col, fi in enumerate(frame_indices):
            cv2.putText(grid, f"Frame {fi}", (col*w + 10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        if save_path:
            cv2.imwrite(save_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return grid
    
    def generate_timeline_strip(self, frames: List[np.ndarray], 
                                 step: int = 5, max_frames: int = 20,
                                 save_path: str = None) -> np.ndarray:
        """
        生成时间序列长条图 (多帧横向拼接)
        适合展示连续帧的跟踪效果
        """
        n_total = len(frames)
        if n_total <= max_frames:
            selected = frames
        else:
            indices = list(range(0, n_total, max(1, n_total // max_frames)))
            selected = [frames[i] for i in indices[:max_frames]]
        
        n = len(selected)
        if n == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        h, w = selected[0].shape[:2]
        # 缩放到统一宽度200
        target_w = 200
        target_h = int(h * target_w / w)
        
        strip = np.zeros((target_h + 40, n * target_w, 3), dtype=np.uint8)
        
        for i, frame in enumerate(selected):
            resized = cv2.resize(frame, (target_w, target_h))
            strip[0:target_h, i*target_w:(i+1)*target_w] = resized
            # 帧号标注
            fn = i * step if step > 0 else i
            cv2.putText(strip, f"F{fn}", (i*target_w + 5, target_h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if save_path:
            cv2.imwrite(save_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return strip
    
    def generate_metric_chart(self, metrics_per_frame: List[Dict],
                               save_path: str = None) -> str:
        """
        生成逐帧指标变化图 (检测数/Recall/Precision)
        """
        if not metrics_per_frame:
            return ""
        
        frames = [m.get('frame', i) for i, m in enumerate(metrics_per_frame)]
        det_counts = [m.get('det_count', 0) for m in metrics_per_frame]
        gt_counts = [m.get('gt_count', 0) for m in metrics_per_frame]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 检测数 vs GT
        ax = axes[0]
        ax.plot(frames, det_counts, 'b-', label='Detections', linewidth=1.5)
        if any(g > 0 for g in gt_counts):
            ax.plot(frames, gt_counts, 'g--', label='Ground Truth', linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Count')
        ax.set_title('Detection Count per Frame')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 多模型确认比例
        ax2 = axes[1]
        multi_2 = [m.get('multi_2', 0) for m in metrics_per_frame]
        multi_3 = [m.get('multi_3', 0) for m in metrics_per_frame]
        ax2.fill_between(frames, 0, det_counts, alpha=0.2, color='gray', label='Total')
        ax2.fill_between(frames, 0, multi_2, alpha=0.4, color='orange', label='2+Models')
        ax2.fill_between(frames, 0, multi_3, alpha=0.6, color='green', label='3+Models')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Count')
        ax2.set_title('Multi-Model Consensus per Frame')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path or ""
    
    def create_video_from_frames(self, frame_paths: List[str], 
                                  output_name: str = "tracking_result",
                                  fps: int = 10) -> str:
        """从帧序列创建MP4视频"""
        if not frame_paths:
            return ""
        
        output_path = os.path.join(self.output_dir, f"{output_name}.mp4")
        first = cv2.imread(frame_paths[0])
        if first is None:
            return ""
        
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is not None:
                writer.write(frame)
        
        writer.release()
        return output_path
    
    def reset_track_history(self):
        """重置轨迹历史（新序列开始时调用）"""
        self.track_history.clear()
    
    def generate_full_report(self, 
                              frames_single: List[Dict],
                              frames_ensemble: List[Dict],
                              sequence_name: str = "VisDrone"):
        """
        一键生成完整定性分析报告
        - 单帧效果图 (Frame 0, 5, 10, 15, 20, 25)
        - 对比网格图
        - 时间序列长条图
        - 逐帧指标图
        - MP4视频
        """
        report = {
            "sequence": sequence_name,
            "timestamp": datetime.now().isoformat(),
            "num_frames": len(frames_ensemble),
            "outputs": {}
        }
        
        # 提取关键帧
        key_indices = [0, 5, 10, 15, 20, 25, 29]
        key_indices = [i for i in key_indices if i < len(frames_ensemble)]
        
        print(f"\n{'='*60}")
        print(f"📊 生成定性分析报告: {sequence_name}")
        print(f"{'='*60}")
        
        # 逐帧指标收集
        metrics = []
        for i in range(len(frames_ensemble)):
            se = frames_ensemble[i] if i < len(frames_ensemble) else {}
            si = frames_single[i] if i < len(frames_single) else {}
            metrics.append({
                "frame": i,
                "det_count": len(se.get('detections', [])),
                "gt_count": len(se.get('ground_truth', [])),
                "multi_2": sum(1 for d in se.get('detections', []) if d.get('num_models', 1) >= 2),
                "multi_3": sum(1 for d in se.get('detections', []) if d.get('num_models', 1) >= 3),
                "single_det_count": len(si.get('detections', [])),
            })
        
        # 单帧效果图
        for fi in key_indices:
            if fi < len(frames_ensemble):
                frame_data = frames_ensemble[fi]
                if frame_data.get('image') is not None:
                    vis = self.draw_frame(
                        frame_data['image'],
                        frame_data.get('detections', []),
                        frame_idx=fi,
                        ground_truth=frame_data.get('ground_truth'),
                        title=f"Frame {fi} - Ensemble Fusion"
                    )
                    path = self.save_frame(vis, fi, "ensemble")
                    report['outputs'][f"frame_{fi}"] = path
        
        # 对比网格图
        single_vis = []
        ensemble_vis = []
        for i in range(len(frames_ensemble)):
            if i < len(frames_ensemble) and frames_ensemble[i].get('image') is not None:
                ev = self.draw_frame(frames_ensemble[i]['image'],
                                    frames_ensemble[i].get('detections', []),
                                    frame_idx=i,
                                    ground_truth=frames_ensemble[i].get('ground_truth'))
                ensemble_vis.append(ev)
            if i < len(frames_single) and frames_single[i].get('image') is not None:
                sv = self.draw_frame(frames_single[i]['image'],
                                    frames_single[i].get('detections', []),
                                    frame_idx=i,
                                    ground_truth=frames_single[i].get('ground_truth'))
                single_vis.append(sv)
        
        if single_vis and ensemble_vis:
            grid_path = os.path.join(self.output_dir, "comparison", "single_vs_ensemble.jpg")
            self.generate_comparison_grid(single_vis, ensemble_vis, save_path=grid_path)
            report['outputs']['comparison_grid'] = grid_path
        
        # 时间序列长条图
        if ensemble_vis:
            strip_path = os.path.join(self.output_dir, "timeline_strip.jpg")
            self.generate_timeline_strip(ensemble_vis, save_path=strip_path)
            report['outputs']['timeline_strip'] = strip_path
        
        # 逐帧指标图
        if metrics:
            chart_path = os.path.join(self.output_dir, "metrics_per_frame.png")
            self.generate_metric_chart(metrics, save_path=chart_path)
            report['outputs']['metrics_chart'] = chart_path
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "qualitative_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 定性分析报告已生成: {report_path}")
        print(f"   输出目录: {self.output_dir}")
        for key, path in report['outputs'].items():
            if path:
                print(f"   - {key}: {os.path.basename(path)}")
        
        return report


# ======================= 便捷函数 =======================
def quick_visualize(vision_system, visdrone_loader, max_frames=30,
                    output_dir="output/qualitative"):
    """快速可视化：给定VisionSystem和数据集，生成效果图"""
    from config.settings import config
    
    visualizer = TemporalVisualizer(output_dir=output_dir)
    frame_data_ensemble = []
    
    for i in range(max_frames):
        frame = visdrone_loader.get_next_frame()
        if frame is None:
            break
        
        # 检测+跟踪
        detections = vision_system.process_frame(frame)
        
        # 获取GT (如果有)
        gt = []
        try:
            gt = visdrone_loader.get_ground_truth(i + 1)
        except:
            pass
        
        # 可视化
        vis = visualizer.draw_frame(frame, detections, frame_idx=i, 
                                     ground_truth=gt)
        visualizer.save_frame(vis, i, "tracking")
        
        frame_data_ensemble.append({
            'image': frame,
            'detections': detections,
            'ground_truth': gt,
        })
        
        if i % 10 == 0:
            print(f"  处理帧 {i}/{max_frames}")
    
    # 生成时间序列长条图
    vis_frames = []
    for i, fd in enumerate(frame_data_ensemble):
        vis = visualizer.draw_frame(fd['image'], fd['detections'], 
                                     frame_idx=i, ground_truth=fd.get('ground_truth'))
        vis_frames.append(vis)
    
    visualizer.generate_timeline_strip(vis_frames)
    print(f"\n✅ 可视化完成，输出: {output_dir}")
    
    return frame_data_ensemble


if __name__ == "__main__":
    # 测试：创建空可视化器
    viz = TemporalVisualizer(output_dir="E:/UAVagent/output/qualitative")
    print("✅ TemporalVisualizer 初始化成功")
    print(f"   输出目录: {viz.output_dir}")
