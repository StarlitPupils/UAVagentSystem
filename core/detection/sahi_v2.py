# E:/UAVagent/core/detection/sahi_v2.py
"""SAHI v2 — VisDrone 优化版：切片内直接检测，不做共识过滤"""
import numpy as np, cv2
from typing import List, Dict
from config.settings import config

class SAHIInferenceV2:
    """SAHI v2: 适用于小目标密集场景"""
    
    def __init__(self, ensemble_detector, slice_size=640, overlap=0.25, 
                 slice_conf_boost=1.3):
        self.detector = ensemble_detector   # EnsembleDetector 实例
        self.slice_size = slice_size
        self.overlap = overlap
        self.slice_conf_boost = slice_conf_boost  # 切片内置信度放大系数
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        h, w = image.shape[:2]
        
        # 如果图像接近切片尺寸，直接全图检测
        if h <= self.slice_size * 1.2 and w <= self.slice_size * 1.2:
            return self._detect_full(image)
        
        # 计算切片步长
        stride = int(self.slice_size * (1 - self.overlap))
        
        all_dets = []
        for y0 in range(0, h - self.slice_size + 1, stride):
            for x0 in range(0, w - self.slice_size + 1, stride):
                # 裁剪切片
                y1, x1 = y0 + self.slice_size, x0 + self.slice_size
                slice_img = image[y0:y1, x0:x1]
                
                # 对切片检测（使用降低的阈值 + 无共识过滤）
                dets = self._detect_slice(slice_img, x0, y0)
                all_dets.extend(dets)
        
        # 处理右边缘和下边缘的残留区域
        if w % stride > 0:
            x0 = w - self.slice_size
            for y0 in range(0, h - self.slice_size + 1, stride):
                y1 = y0 + self.slice_size
                slice_img = image[y0:y1, x0:w]
                dets = self._detect_slice(slice_img, x0, y0)
                all_dets.extend(dets)
        
        if h % stride > 0:
            y0 = h - self.slice_size
            for x0 in range(0, w - self.slice_size + 1, stride):
                x1 = x0 + self.slice_size
                slice_img = image[y0:h, x0:x1]
                dets = self._detect_slice(slice_img, x0, y0)
                all_dets.extend(dets)
        
        # 全图 NMS
        return self._global_nms(all_dets)
    
    def _detect_full(self, image):
        return self.detector.detect(image)
    
    def _detect_slice(self, slice_img, offset_x, offset_y):
        """切片内检测：降低阈值 + 提升置信度"""
        # 临时降低检测阈值
        saved_conf = config.DETECTION_CONFIDENCE
        config.DETECTION_CONFIDENCE = max(0.15, saved_conf * 0.7)  # 切片内更宽松
        
        # 直接调用模型检测（绕过共识过滤）
        all_dets_raw = []
        for model in self.detector.models:
            try:
                results = model(slice_img, conf=config.DETECTION_CONFIDENCE, 
                               device=self.detector.device, verbose=False)
                if results and results[0].boxes is not None:
                    for i in range(len(results[0].boxes)):
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
                        all_dets_raw.append({
                            "bbox": [(x1+x2)/2 + offset_x, (y1+y2)/2 + offset_y, 
                                     x2-x1, y2-y1],
                            "confidence": float(results[0].boxes.conf[i]) * self.slice_conf_boost,
                            "class": int(results[0].boxes.cls[i]),
                        })
            except:
                pass
        
        config.DETECTION_CONFIDENCE = saved_conf
        return all_dets_raw
    
    def _global_nms(self, detections: List[Dict], iou_thr: float = 0.45) -> List[Dict]:
        """全图 NMS 去重"""
        if len(detections) < 2:
            return detections
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        boxes = np.array([d['bbox'] for d in detections])
        # 转 xyxy
        x1 = boxes[:, 0] - boxes[:, 2]/2
        y1 = boxes[:, 1] - boxes[:, 3]/2
        x2 = boxes[:, 0] + boxes[:, 2]/2
        y2 = boxes[:, 1] + boxes[:, 3]/2
        areas = (x2 - x1) * (y2 - y1)
        
        keep = []
        for i in range(len(detections)):
            if any(self._compute_iou(
                [x1[i], y1[i], x2[i], y2[i]],
                [x1[j], y1[j], x2[j], y2[j]]
            ) > iou_thr for j in keep):
                continue
            keep.append(i)
            if len(keep) >= 200:  # 上限
                break
        
        return [detections[i] for i in keep]
    
    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-8)