
# core/detection/sahi.py (UAVagent 1.3 Phase 3)
"""SAHI (Slicing Aided Hyper Inference) — 小目标检测增强
用于无人机航拍大图中密集小目标的检测。
"""
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from config.settings import config


class SAHIInference:
    """SAHI 切片推理器"""
    
    def __init__(self, detector, slice_height: int = 640, slice_width: int = 640,
                 overlap_ratio: float = 0.2, min_area_ratio: float = 0.0001):
        self.detector = detector          # VisionSystem 或 EnsembleDetector
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio  # 相邻切片重叠比例
        self.min_area_ratio = min_area_ratio  # 保留检测的最小尺寸比（相对于原图）
        
    def detect(self, image: np.ndarray, conf_thr: float = None,
               iou_thr: float = 0.5) -> List[Dict]:
        """执行 SAHI 切片推理"""
        h, w = image.shape[:2]
        print(f"[SAHI] 输入尺寸: {w}x{h}, 切片: {self.slice_width}x{self.slice_height}")
        
        # 如果图像尺寸小于切片尺寸，直接检测
        if h <= self.slice_height and w <= self.slice_width:
            dets = self._detect_whole(image, conf_thr); print(f"[SAHI] 整图检测: {len(dets)} 个目标"); return dets
        
        # 计算切片参数
        y_steps = self._compute_steps(h, self.slice_height)
        x_steps = self._compute_steps(w, self.slice_width)
        
        all_detections = []
        
        for y_start in y_steps:
            for x_start in x_steps:
                # 裁剪切片
                y_end = min(y_start + self.slice_height, h)
                x_end = min(x_start + self.slice_width, w)
                slice_img = image[y_start:y_end, x_start:x_end, :]
                
                # 检测
                dets = self._detect_whole(slice_img, conf_thr)
                
                # 映射坐标回原图
                for d in dets:
                    bbox = d['bbox']  # [cx, cy, w, h]
                    d['bbox'] = [
                        bbox[0] + x_start,
                        bbox[1] + y_start,
                        bbox[2],
                        bbox[3]
                    ]
                all_detections.extend(dets)
        
        print(f"[SAHI] 切片合并前: {len(all_detections)} 个检测"); # 全图 NMS 合并重叠框
        if len(all_detections) > 1:
            all_detections = self._nms(all_detections, iou_thr); print(f"[SAHI] NMS后: {len(all_detections)}")
        
        # 过滤极小框（可能是切片边缘噪声）
        img_area = h * w
        all_detections = [
            d for d in all_detections 
            if (d['bbox'][2] * d['bbox'][3]) / img_area >= self.min_area_ratio
        ]
        
        return all_detections
    
    def _compute_steps(self, total: int, slice_size: int) -> List[int]:
        """计算切片起始位置"""
        if total <= slice_size:
            return [0]
        step = int(slice_size * (1 - self.overlap_ratio))
        steps = list(range(0, total - slice_size, step))
        steps.append(total - slice_size)  # 确保最后一块覆盖边缘
        return sorted(set(steps))  # 去重排序
    
    def _detect_whole(self, image, conf_thr):
        """对单张图像检测（兼容 VisionSystem 和 EnsembleDetector）"""
        if hasattr(self.detector, 'detect_only'):
            return self.detector.detect_only(image)
        elif hasattr(self.detector, 'detect'):
            return self.detector.detect(image, conf_thr)
        else:
            return []
    
    def _nms(self, detections: List[Dict], iou_thr: float) -> List[Dict]:
        """对 SAHI 合并结果执行 NMS"""
        if not detections:
            return detections
        
        # 提取框和置信度
        boxes = np.array([d['bbox'] for d in detections])  # [cx, cy, w, h]
        scores = np.array([d.get('confidence', 0.5) for d in detections])
        
        # 转为 [x1, y1, x2, y2]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]


def create_sahi_detector(vision_system, **kwargs):
    """为现有 VisionSystem 包裹 SAHI"""
    return SAHIInference(vision_system, **kwargs)
