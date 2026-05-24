# core/detection/preprocessing.py (UAVagent 1.4 P3.1)
"""图像预处理 + 场景自适应阈值估计"""
import cv2
import numpy as np
from typing import Tuple, Dict
from config.settings import config


class ImagePreprocessor:
    """图像预处理器 - CLAHE增强 + 场景分析"""
    
    def __init__(self):
        self.enabled = getattr(config, 'PREPROCESSING_ENABLED', True)
        self.clahe_clip = getattr(config, 'CLAHE_CLIP_LIMIT', 2.0)
        self.clahe_tile = getattr(config, 'CLAHE_TILE_GRID', (8, 8))
        
        if self.enabled:
            self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
        else:
            self.clahe = None
    
    def enhance(self, image: np.ndarray, strategy: str = "auto") -> np.ndarray:
        if not self.enabled or strategy == "none":
            return image
        if strategy == "auto":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = gray.mean()
            if mean_brightness < 60:
                strategy = "night"
            elif mean_brightness < 100:
                strategy = "clahe"
            else:
                strategy = "light_clahe"
        if strategy in ("clahe", "light_clahe"):
            return self._clahe_enhance(image, strength=1.0 if strategy == "clahe" else 0.5)
        elif strategy == "night":
            return self._night_enhance(image)
        return image
    
    def _clahe_enhance(self, image, strength=1.0):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        l_mix = cv2.addWeighted(l, 1-strength, l_eq, strength, 0)
        return cv2.cvtColor(cv2.merge([l_mix, a, b]), cv2.COLOR_LAB2RGB)
    
    def _night_enhance(self, image):
        enhanced = self._clahe_enhance(image, 1.0)
        gamma = 1.5
        table = np.array([(i/255.0)**(1.0/gamma)*255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(enhanced, table)
    
    # ========== P3.1 新增：场景自适应阈值估计 ==========
    
    def analyze_scene(self, image: np.ndarray) -> Dict[str, float]:
        """分析场景特征，返回自适应参数建议
        
        Returns:
            {
                'mean_brightness': float,    # 平均亮度 0-255
                'std_brightness': float,     # 亮度标准差
                'estimated_density': float,  # 预估目标密度 (0=稀疏, 1=密集)
                'suggested_conf': float,     # 建议检测置信度阈值
                'suggested_iou': float,      # 建议NMS IoU阈值
                'scene_type': str,           # 场景类型
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        mean_b = float(gray.mean())
        std_b = float(gray.std())
        
        # ---- 场景类型判定 ----
        if mean_b < 60:
            scene_type = "night"
            base_conf = 0.18
            base_iou = 0.35
        elif mean_b < 100:
            scene_type = "dim"
            base_conf = 0.22
            base_iou = 0.38
        elif mean_b > 200:
            scene_type = "bright"
            base_conf = 0.30
            base_iou = 0.45
        else:
            scene_type = "normal"
            base_conf = 0.25
            base_iou = 0.40
        
        # ---- 密度估计（基于边缘密度） ----
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (h * w)
        
        # 边缘密度映射到目标密度
        if edge_density > 0.15:
            estimated_density = 1.0   # 密集场景
            conf_adj = -0.05          # 降低阈值避免FP
            iou_adj = -0.05           # 降低IoU避免过度合并
        elif edge_density > 0.08:
            estimated_density = 0.6
            conf_adj = -0.02
            iou_adj = 0.0
        elif edge_density > 0.03:
            estimated_density = 0.3
            conf_adj = 0.0
            iou_adj = 0.0
        else:
            estimated_density = 0.1   # 稀疏场景
            conf_adj = +0.03          # 提高阈值减少FP
            iou_adj = +0.05
        
        # ---- 对比度补偿 ----
        if std_b < 30:
            conf_adj -= 0.03  # 低对比度，放宽阈值
        
        suggested_conf = max(0.10, min(0.50, base_conf + conf_adj))
        suggested_iou = max(0.25, min(0.60, base_iou + iou_adj))
        
        return {
            'mean_brightness': round(mean_b, 1),
            'std_brightness': round(std_b, 1),
            'estimated_density': round(estimated_density, 2),
            'edge_density': round(edge_density, 4),
            'suggested_conf': round(suggested_conf, 2),
            'suggested_iou': round(suggested_iou, 2),
            'scene_type': scene_type,
        }
    
    def get_adaptive_thresholds(self, image: np.ndarray) -> Tuple[float, float]:
        """便捷方法：返回 (建议conf, 建议IoU)"""
        analysis = self.analyze_scene(image)
        return analysis['suggested_conf'], analysis['suggested_iou']


preprocessor = ImagePreprocessor()