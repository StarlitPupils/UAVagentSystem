# E:/UAVagent/core/detection/sahi_v3.py (UAVagent 1.4 P1)
"""SAHI v3 — 大图优化版：自适应切片 + 全图融合合并"""
import numpy as np, cv2, os, time
from typing import List, Dict, Tuple
from config.settings import config

class SAHIInferenceV3:
    """SAHI v3: 适用于大尺度航拍图（2000-4000px+）
    
    改进:
    - 自适应切片大小：根据图像尺寸自动选择最优切片
    - 渐进式重叠：边缘区域重叠更大
    - 智能合并：全图融合检测 + 切片补充
    """
    
    # 切片大小预设（根据图像尺寸自动选择）
    SLICE_PRESETS = {
        "small":  (640, 0.25),   # < 1500px
        "medium": (800, 0.20),   # 1500-2500px
        "large":  (1024, 0.15),  # 2500-4000px
        "xlarge": (1280, 0.10),  # > 4000px
    }
    
    def __init__(self, ensemble_detector, finetuned_model_idx=0,
                 slice_size=None, overlap=None, slice_conf=0.12,
                 merge_iou=0.35, max_dets=300):
        self.ensemble = ensemble_detector
        self.finetuned_model_idx = finetuned_model_idx
        self.slice_conf = slice_conf
        self.merge_iou = merge_iou
        self.max_dets = max_dets
        
        # 手动指定覆盖自动选择
        self.manual_slice_size = slice_size
        self.manual_overlap = overlap
        
        # 统计
        self.stats = {"total_calls": 0, "total_slices": 0, "total_time_ms": 0}
    
    def _get_slice_params(self, image_h: int, image_w: int) -> Tuple[int, float]:
        """根据图像尺寸自适应选择切片参数"""
        if self.manual_slice_size is not None and self.manual_overlap is not None:
            return self.manual_slice_size, self.manual_overlap
        
        max_dim = max(image_h, image_w)
        
        if max_dim < 1500:
            preset = self.SLICE_PRESETS["small"]
        elif max_dim < 2500:
            preset = self.SLICE_PRESETS["medium"]
        elif max_dim < 4000:
            preset = self.SLICE_PRESETS["large"]
        else:
            preset = self.SLICE_PRESETS["xlarge"]
        
        return preset[0], preset[1]
    
    def _compute_slice_positions(self, total: int, slice_size: int,
                                  overlap_ratio: float) -> List[int]:
        """计算切片起始位置，确保覆盖边缘"""
        if total <= slice_size:
            return [0]
        
        stride = int(slice_size * (1 - overlap_ratio))
        positions = list(range(0, total - slice_size, stride))
        
        # 确保最后一块覆盖右/下边缘
        last = total - slice_size
        if last > 0 and (not positions or positions[-1] < last - stride // 2):
            positions.append(last)
        
        return sorted(set(positions))
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """执行 SAHI 切片推理 + 全图融合合并"""
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        
        # 自适应切片参数
        slice_size, overlap = self._get_slice_params(h, w)
        
        print(f"  [SAHI v3] 图像={w}x{h} | 切片={slice_size} | 重叠={overlap:.0%}")
        
        # === 第一步：全图融合检测（高精度基线） ===
        dets_full = self.ensemble.detect(image)
        print(f"  [SAHI v3] 全图融合: {len(dets_full)} dets")
        
        # === 第二步：SAHI 切片检测（高召回补充） ===
        dets_slices = self._detect_slices(image, slice_size, overlap)
        print(f"  [SAHI v3] 切片检测: {len(dets_slices)} dets ({self.stats['total_slices']} slices)")
        
        # === 第三步：智能合并 ===
        merged = self._merge_detections(dets_full, dets_slices)
        
        elapsed = (time.perf_counter() - t0) * 1000
        self.stats["total_calls"] += 1
        self.stats["total_time_ms"] += elapsed
        
        return merged
    
    def _detect_slices(self, image: np.ndarray, slice_size: int,
                       overlap: float) -> List[Dict]:
        """切片检测：仅用微调单模型，低阈值快速扫描"""
        h, w = image.shape[:2]
        
        y_positions = self._compute_slice_positions(h, slice_size, overlap)
        x_positions = self._compute_slice_positions(w, slice_size, overlap)
        
        all_dets = []
        self.stats["total_slices"] = len(y_positions) * len(x_positions)
        
        for y0 in y_positions:
            for x0 in x_positions:
                y1 = min(y0 + slice_size, h)
                x1 = min(x0 + slice_size, w)
                slice_img = image[y0:y1, x0:x1]
                
                # 用微调单模型检测（低阈值）
                try:
                    slice_model = self.ensemble.models[self.finetuned_model_idx]
                    results = slice_model(
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
                                'source': 'sahi_slice',
                            }
                            
                            # 过滤边缘检测（容易不完整）
                            margin = max(5, int(slice_size * 0.02))
                            if (bx1 > margin and by1 > margin and 
                                bx2 < slice_size - margin and 
                                by2 < slice_size - margin):
                                all_dets.append(det)
                except Exception:
                    pass
        
        # 切片间 NMS
        return self._nms(all_dets, self.merge_iou)
    
    def _merge_detections(self, dets_full: List[Dict],
                          dets_slices: List[Dict]) -> List[Dict]:
        """合并全图检测和切片检测
        
        策略:
        1. 全图检测优先级最高（多模型确认）
        2. 切片检测中与全图检测不重叠的作为补充
        3. 根据置信度排序保留 top-k
        """
        if not dets_slices:
            return dets_full
        
        merged = list(dets_full)
        
        for sd in dets_slices:
            is_duplicate = False
            for md in merged:
                if self._iou(sd['bbox'], md['bbox']) > self.merge_iou * 0.8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                sd['source'] = 'sahi_supplement'
                merged.append(sd)
        
        # 置信度排序，保留 top-k
        merged.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        if len(merged) > self.max_dets:
            merged = merged[:self.max_dets]
        
        added = len(merged) - len(dets_full)
        if added > 0:
            print(f"  [SAHI v3] 合并: +{added} 补充检测 (总计 {len(merged)})")
        
        return merged
    
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
        """NMS 去重"""
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
    
    def get_stats(self) -> dict:
        """返回 SAHI 统计信息"""
        return {
            **self.stats,
            "avg_time_ms": round(self.stats["total_time_ms"] / max(self.stats["total_calls"], 1), 1),
            "avg_slices": round(self.stats["total_slices"] / max(self.stats["total_calls"], 1), 1),
        }