# core/detection/preprocessing.py
"""图像预处理增强模块 - UAVagent 1.2
支持: CLAHE自适应直方图均衡 / 全局直方图均衡 / 去雾 / 锐化
专为无人机航拍低光照/雾霾场景设计
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from config.settings import config


class ImagePreprocessor:
    """图像预处理器 - 多策略增强"""
    
    def __init__(self):
        self.enabled = config.PREPROCESSING_ENABLED
        self.clahe_clip = config.CLAHE_CLIP_LIMIT
        self.clahe_tile = config.CLAHE_TILE_GRID
        self.hist_eq = config.HIST_EQUALIZATION
        
        # 初始化CLAHE
        if self.enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip, 
                tileGridSize=self.clahe_tile
            )
        else:
            self.clahe = None
    
    def enhance(self, image: np.ndarray, strategy: str = "auto") -> np.ndarray:
        """
        增强图像
        strategy: "auto" | "clahe" | "hist_eq" | "night" | "fog" | "none"
        """
        if not self.enabled or strategy == "none":
            return image
        
        # 自动模式：根据图像亮度选择策略
        if strategy == "auto":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = gray.mean()
            
            if mean_brightness < 60:
                strategy = "night"  # 极暗场景
            elif mean_brightness < 100:
                strategy = "clahe"  # 偏暗场景
            elif mean_brightness > 200:
                strategy = "clahe"  # 过曝也做均衡
            else:
                strategy = "light_clahe"  # 正常光照轻微增强
        
        # 执行增强
        if strategy == "clahe":
            return self._clahe_enhance(image)
        elif strategy == "light_clahe":
            return self._clahe_enhance(image, strength=0.5)
        elif strategy == "hist_eq":
            return self._histogram_equalization(image)
        elif strategy == "night":
            return self._night_enhance(image)
        elif strategy == "fog":
            return self._dehaze(image)
        
        return image
    
    def _clahe_enhance(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """CLAHE增强 - 在LAB空间处理亮度通道"""
        # 转换到LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道做CLAHE
        l_eq = self.clahe.apply(l)
        
        # 按强度混合原图和增强图
        l_mix = cv2.addWeighted(l, 1 - strength, l_eq, strength, 0)
        
        # 合并
        lab_eq = cv2.merge([l_mix, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return result
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """全局直方图均衡"""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    
    def _night_enhance(self, image: np.ndarray) -> np.ndarray:
        """夜间增强：CLAHE + Gamma校正"""
        # 先CLAHE
        enhanced = self._clahe_enhance(image, strength=1.0)
        
        # Gamma校正提升暗部
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 
                         for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        """简易去雾（暗通道先验的简化版）"""
        # 转float
        img_float = image.astype(np.float32) / 255.0
        
        # 暗通道
        dark_channel = img_float.min(axis=2)
        
        # 大气光估计
        atm_light = np.percentile(dark_channel, 95)
        
        # 传输图估计
        omega = 0.8
        transmission = 1 - omega * dark_channel / (atm_light + 1e-8)
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # 恢复
        result = np.zeros_like(img_float)
        for c in range(3):
            result[:, :, c] = (img_float[:, :, c] - atm_light) / (transmission + 1e-8) + atm_light
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    def get_stats(self, image: np.ndarray) -> dict:
        """获取图像统计信息"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return {
            "mean_brightness": float(gray.mean()),
            "std_brightness": float(gray.std()),
            "min_brightness": int(gray.min()),
            "max_brightness": int(gray.max()),
            "shape": image.shape,
        }


# 全局单例
preprocessor = ImagePreprocessor()
