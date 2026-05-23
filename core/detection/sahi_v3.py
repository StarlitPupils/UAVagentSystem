# E:/UAVagent/core/detection/sahi_v3.py
"""SAHI v3 — 混合策略：切片单模型 + 全图融合合并"""
import numpy as np, cv2, os
from typing import List, Dict
from config.settings import config

class SAHIInferenceV3:
    """SAHI v3: 切片用微调模型快速扫描，全图用融合验证"""
    
    def __init__(self, ensemble_detector, finetuned_model_idx=0,
                 slice_size=640, overlap=0.3, slice_conf=0.15,
                 merge_iou=0.40, max_dets=200):
        self.ensemble = ensemble_detector
        self.slice_model = ensemble_detector.models[finetuned_model_idx]  # 微调模型
        self.slice_size = slice_size
        self.overlap = overlap
        self.slice_conf = slice_conf       # 切片内低阈值
        self.merge_iou = merge_iou         # 合并 IoU 阈值
        self.max_dets = max_dets
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        h, w = image.shape[:2]
        
        # === 第一步：全图融合检测（高精度基线） ===
        dets_full = self.ensemble.detect(image)
        
        # === 第二步：SAHI 切片检测（高召回补充） ===
        dets_slices = self._detect_slices(image)
        
        # === 第三步：合并 ===
        if not dets_slices:
            return dets_full
        
        # 将切片检测中与全图检测重叠的框剔除（优先信任全图检测）
        merged = list(dets_full)
        for sd in dets_slices:
            # 检查是否与已有检测重复
            is_duplicate = False
            for md in merged:
                if self._iou(sd['bbox'], md['bbox']) > 0.35:
                    is_duplicate = True
                    break
            if not is_duplicate:
                # 标记为 SAHI 补充检测
                sd['source'] = 'sahi'
                merged.append(sd)
        
        # 置信度排序，保留 top-k
        merged.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        if len(merged) > self.max_dets:
            merged = merged[:self.max_dets]
        
        return merged
    
    def _detect_slices(self, image: np.ndarray) -> List[Dict]:
        """切片检测：仅用微调单模型，低阈值快速扫描"""
        h, w = image.shape[:2]
        stride = int(self.slice_size * (1 - self.overlap))
        
        all_dets = []
        
        # 计算所有切片位置
        y_starts = list(range(0, h - self.slice_size + 1, stride))
        x_starts = list(range(0, w - self.slice_size + 1, stride))
        
        # 确保覆盖边缘
        if not y_starts or y_starts[-1] < h - self.slice_size:
            y_starts.append(max(0, h - self.slice_size))
        if not x_starts or x_starts[-1] < w - self.slice_size:
            x_starts.append(max(0, w - self.slice_size))
        
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + self.slice_size, h)
                x1 = min(x0 + self.slice_size, w)
                slice_img = image[y0:y1, x0:x1]
                
                # 用微调单模型检测（低阈值）
                try:
                    results = self.slice_model(
                        slice_img, conf=self.slice_conf,
                        device=self.ensemble.device,
                        verbose=False
                    )
                    if results and results[0].boxes is not None:
                        for i in range(len(results[0].boxes)):
                            bx1, by1, bx2, by2 = results[0].boxes.xyxy[i].tolist()
                            conf = float(results[0].boxes.conf[i])
                            cls = int(results[0].boxes.cls[i])
                            
                            # 映射回原图坐标
                            det = {
                                'bbox': [
                                    (bx1 + bx2) / 2 + x0,
                                    (by1 + by2) / 2 + y0,
                                    bx2 - bx1,
                                    by2 - by1,
                                ],
                                'confidence': conf,
                                'class': cls,
                                'num_models': 1,
                                'id': None,
                            }
                            
                            # 过滤边缘检测（容易不完整）
                            margin = 10
                            if (bx1 > margin and by1 > margin and 
                                bx2 < self.slice_size - margin and 
                                by2 < self.slice_size - margin):
                                all_dets.append(det)
                except Exception:
                    pass
        
        # 切片间 NMS
        return self._nms(all_dets, self.merge_iou)
    
    def _iou(self, box1, box2):
        """计算两个 [cx,cy,w,h] 框的 IoU"""
        def to_xyxy(b):
            return [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
        b1, b2 = to_xyxy(box1), to_xyxy(box2)
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1, area2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (area1 + area2 - inter + 1e-8)
    
    def _nms(self, detections: List[Dict], iou_thr: float) -> List[Dict]:
        if len(detections) < 2:
            return detections
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        for i in range(len(detections)):
            keep_i = True
            for j in keep:
                if self._iou(detections[i]['bbox'], detections[j]['bbox']) > iou_thr:
                    keep_i = False
                    break
            if keep_i:
                keep.append(i)
        return [detections[i] for i in keep]