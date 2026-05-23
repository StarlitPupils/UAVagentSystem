# -*- coding: utf-8 -*-
"""ReID 特征提取器 v1.3 - 多后端自适应（torchvision 保证成功）"""
import os, sys, numpy as np, cv2

class ReIDFeatureExtractor:
    def __init__(self):
        self.enabled = True
        self.backend = None          # 'osnet', 'torchvision', 'hsv'
        self.feature_dim = 64        # 默认 HSV 维度
        self.model = None
        self.transform = None
        self._init_model()

    def _init_model(self):
        # ---- 第一优先：OSNet（torchreid）----
        try:
            import torch, torchreid
            self.model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.backend = 'osnet'
            self.feature_dim = 512
            print('[ReID] OSNet 512-dim 已启用')
            return
        except Exception:
            print('[ReID] OSNet 不可用，尝试 torchvision 备选...')

        # ---- 第二优先：torchvision ResNet50（绝对可行）----
        try:
            import torch, torchvision
            from torchvision import models, transforms
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()  # 去掉分类头
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.backend = 'torchvision'
            self.feature_dim = 2048
            print('[ReID] torchvision ResNet50 2048-dim 已启用')
            return
        except Exception:
            print('[ReID] torchvision 不可用，降级到 HSV 64-dim')

        # ---- 兜底：HSV 直方图 ----
        self.backend = 'hsv'
        self.feature_dim = 64
        print('[ReID] 使用 HSV 直方图 64-dim')

    def extract(self, image, bbox):
        if not self.enabled or image is None:
            return None
        roi = self._crop_roi(image, bbox)
        if roi is None:
            return None

        if self.backend == 'osnet':
            return self._extract_osnet(roi)
        elif self.backend == 'torchvision':
            return self._extract_torchvision(roi)
        else:
            return self._extract_hsv(roi)

    def _crop_roi(self, image, bbox):
        try:
            cx, cy = int(float(bbox[0])), int(float(bbox[1]))
            w, h = max(1, int(float(bbox[2]))), max(1, int(float(bbox[3])))
            x1, y1 = max(0, cx-w//2), max(0, cy-h//2)
            x2, y2 = min(image.shape[1]-1, cx+w//2), min(image.shape[0]-1, cy+h//2)
            if x2 <= x1+3 or y2 <= y1+3:
                return None
            return image[y1:y2, x1:x2]
        except:
            return None

    def _extract_osnet(self, roi):
        import torch
        try:
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = transform(roi_rgb).unsqueeze(0)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                feat = self.model(tensor)
            return feat.cpu().numpy().flatten()
        except:
            return None

    def _extract_torchvision(self, roi):
        import torch
        try:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self.transform(roi_rgb).unsqueeze(0)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                feat = self.model(tensor)
            return feat.cpu().numpy().flatten()
        except:
            return None

    def _extract_hsv(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def compute_distance(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.5
        try:
            f1 = np.array(f1, dtype=np.float32).flatten()
            f2 = np.array(f2, dtype=np.float32).flatten()
            min_dim = min(len(f1), len(f2))
            if min_dim < 2:
                return 0.5
            f1, f2 = f1[:min_dim], f2[:min_dim]
            dot = np.dot(f1, f2)
            n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.5
            return float(np.clip(1.0 - dot/(n1*n2), 0.0, 1.0))
        except:
            return 0.5

    def get_backend_info(self):
        return {'backend': self.backend, 'feature_dim': self.feature_dim}

reid_extractor = ReIDFeatureExtractor()
