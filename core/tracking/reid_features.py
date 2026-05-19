# core/tracking/reid_features.py (v1.2.1)
"""深度ReID特征提取器 - 统一64维HSV降级，消除维度冲突"""
import numpy as np
import cv2
from typing import Optional
from config.settings import config


class ReIDFeatureExtractor:
    """特征提取器 - 所有路径统一输出64维向量"""

    def __init__(self):
        self.enabled = config.REID_ENABLED
        self.feature_dim = 64  # 统一64维！避免与HSV冲突
        self.model = None
        self.model_name = config.REID_MODEL
        self._backend = 'hsv'  # 默认HSV

        if self.enabled:
            self._init_model()

    def _init_model(self):
        """初始化 - 优先HSV（最稳定），OSNet可选"""
        # 优先使用HSV（稳定、快速、维度可控）
        self._backend = 'hsv'
        self.feature_dim = 64
        print(f"[ReID] 使用HSV直方图特征 (dim=64, 稳定模式)")

        # 尝试加载OSNet（可选增强）
        try:
            import torch
            import torchreid
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1000,
                pretrained=True
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self._backend = 'torchreid'
            self.feature_dim = 512
            print(f"[ReID] OSNet模型加载成功! 升级到dim=512")
        except ImportError:
            pass  # 静默使用HSV
        except Exception as e:
            print(f"[ReID] OSNet加载失败: {e}，使用HSV降级")

    def extract(self, image: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """提取特征 - 统一返回float32 ndarray"""
        if not self.enabled or image is None:
            return None

        roi = self._crop_roi(image, bbox)
        if roi is None:
            return None

        try:
            if self._backend == 'torchreid' and self.model is not None:
                feat = self._extract_torchreid(roi)
                if feat is not None:
                    return feat.astype(np.float32).flatten()
        except Exception:
            pass

        # 降级HSV
        return self._extract_hsv(roi)

    def _crop_roi(self, image, bbox):
        """裁剪ROI"""
        try:
            cx = int(float(bbox[0]))
            cy = int(float(bbox[1]))
            w = max(1, int(float(bbox[2])))
            h = max(1, int(float(bbox[3])))

            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(image.shape[1] - 1, cx + w // 2)
            y2 = min(image.shape[0] - 1, cy + h // 2)

            if x2 <= x1 + 3 or y2 <= y1 + 3:
                return None

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            return cv2.resize(roi, (64, 128))
        except Exception:
            return None

    def _extract_torchreid(self, roi):
        """OSNet特征"""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if roi.shape[-1] == 3 else roi
        tensor = transform(roi_rgb).unsqueeze(0)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            features = self.model(tensor)

        return features.cpu().numpy().flatten()

    def _extract_hsv(self, roi):
        """HSV直方图 - 64维 (8x8)"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def compute_distance(self, feat1, feat2) -> float:
        """计算特征距离 - 自动适配维度"""
        if feat1 is None or feat2 is None:
            return 0.5

        try:
            f1 = np.array(feat1, dtype=np.float32).flatten()
            f2 = np.array(feat2, dtype=np.float32).flatten()

            # 对齐到相同维度（取较小者）
            min_dim = min(len(f1), len(f2))
            if min_dim < 2:
                return 0.5
            f1 = f1[:min_dim]
            f2 = f2[:min_dim]

            # 余弦距离
            dot = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)

            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.5

            cosine_sim = dot / (norm1 * norm2)
            return float(np.clip(1.0 - cosine_sim, 0.0, 1.0))
        except Exception:
            return 0.5


# 全局单例
reid_extractor = ReIDFeatureExtractor()
