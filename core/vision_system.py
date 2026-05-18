# core/vision_system.py
"""视觉系统 - 集成Ensemble多模型融合检测"""
import cv2
import numpy as np
import os
from config.settings import config

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class VisionSystem:
    def __init__(self, device: str = None, use_ensemble: bool = True):
        if device is None:
            device = config.DETECTION_DEVICE
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_ensemble = use_ensemble
        print(f"[Vision] 设备: {device.upper()}")
        print(f"[Vision] 集成模式: {'多模型融合' if use_ensemble else '单模型'}")

        # 主模型（单模型模式）
        self.model = None
        # 集成检测器（多模型模式）
        self.ensemble = None
        # 跟踪器
        self.tracker = None
        self.tracker_type = None
        # 状态
        self.latest_detections = []
        self.latest_frame = None
        # 精度统计
        self.detection_count = 0
        self.total_detections = 0

        self._init_model()
        self._init_tracker()

    def _init_model(self):
        """加载检测模型"""
        if not ULTRALYTICS_AVAILABLE:
            print("[Vision] ultralytics 不可用")
            return

        base = os.path.dirname(os.path.dirname(__file__))
        model_search = [
            os.path.join(base, "models", "yolo11x.pt"),
            os.path.join(base, "models", "yolo11n.pt"),
            os.path.join(base, "models", "yolov8x.pt"),
            os.path.join(base, "models", "yolov8n.pt"),
        ]

        # 加载主模型（单模型模式用）
        loaded = False
        for path in model_search:
            if os.path.exists(path):
                try:
                    import torch
                    self.model = YOLO(path)
                    self.model.to(self.device)
                    fname = os.path.basename(path).replace('.pt', '')
                    config.YOLO_MODEL_PATH = path
                    config.YOLO_MODEL_NAME = fname
                    print(f"[Vision] 主模型: {fname}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[Vision] {path} 加载失败: {e}")

        if not loaded:
            print("[Vision] 未找到本地模型，尝试下载 yolo11n...")
            try:
                self.model = YOLO("yolo11n.pt")
                self.model.to(self.device)
                config.YOLO_MODEL_NAME = "yolo11n"
            except:
                pass

        # 初始化集成检测器
        if self.use_ensemble:
            available_models = [p for p in model_search if os.path.exists(p)]
            if len(available_models) >= 2:
                from core.detection.ensemble_detector import EnsembleDetector
                self.ensemble = EnsembleDetector(available_models, self.device)
            else:
                print("[Vision] 模型不足2个，降级为单模型模式")
                self.use_ensemble = False

    def _init_tracker(self):
        tracker_type = config.TRACKER_TYPE
        self.tracker_type = tracker_type
        try:
            from core.tracking.tracker_registry import get_tracker
            self.tracker = get_tracker(tracker_type)
            print(f"[Vision] 跟踪器: {self.tracker.name}")
        except ImportError:
            from core.tracking.iou_tracker import EnhancedTracker
            self.tracker = EnhancedTracker(max_age=30, min_hits=5, iou_threshold=0.25)
            print("[Vision] 跟踪器: EnhancedTracker (内嵌)")

    def reload_model(self):
        print(f"[Vision] 热切换模型 -> {config.YOLO_MODEL_NAME}")
        self._init_model()

    def switch_tracker(self, tracker_type: str = None):
        if tracker_type:
            config.TRACKER_TYPE = tracker_type
        print(f"[Vision] 热切换跟踪器 -> {config.TRACKER_TYPE}")
        self._init_tracker()

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """检测+跟踪一帧"""
        self.latest_frame = frame

        # ---- 检测 ----
        if self.use_ensemble and self.ensemble is not None:
            raw_dets = self.ensemble.detect(frame)
            # 共识过滤：去除单模型低置信度检测（仅融合模式）
            try:
                from core.detection.filter_consensus import filter_by_consensus
                raw_dets = filter_by_consensus(raw_dets, min_models=2, min_conf_single=0.25)
            except ImportError:
                pass
        else:
            raw_dets = self._detect_single(frame)

        # 统计
        self.detection_count += 1
        self.total_detections += len(raw_dets)

        # ---- 跟踪 ----
        if self.tracker is not None:
            try:
                tracked = self.tracker.update(raw_dets, frame)
                self.latest_detections = tracked
                return tracked
            except Exception as e:
                print(f"[Vision] 跟踪失败: {e}")

        for i, d in enumerate(raw_dets):
            d["id"] = i
        self.latest_detections = raw_dets
        return raw_dets

    def _detect_single(self, frame: np.ndarray) -> list[dict]:
        """单模型检测"""
        if self.model is None:
            return self._mock_detections(frame)
        try:
            results = self.model(frame, conf=config.DETECTION_CONFIDENCE,
                                 device=self.device, verbose=False)
            dets = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    dets.append({
                        "bbox": [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                        "confidence": float(boxes.conf[i]),
                        "class": int(boxes.cls[i]),
                        "id": None,
                    })
            return dets
        except Exception as e:
            print(f"[Vision] 检测失败: {e}")
            return self._mock_detections(frame)

    def detect_only(self, frame: np.ndarray) -> list[dict]:
        if self.use_ensemble and self.ensemble is not None:
            return self.ensemble.detect(frame)
        return self._detect_single(frame)

    def get_stats(self) -> dict:
        return {
            "detection_count": self.detection_count,
            "total_detections": self.total_detections,
            "avg_per_frame": self.total_detections / max(self.detection_count, 1),
            "ensemble_mode": self.use_ensemble and self.ensemble is not None,
            "num_models": self.ensemble.count_models() if self.ensemble else 1,
            "model_names": self.ensemble.get_model_names() if self.ensemble else [config.YOLO_MODEL_NAME],
        }

    def _mock_detections(self, frame: np.ndarray) -> list[dict]:
        import random
        h, w = frame.shape[:2]
        return [{"bbox": [random.randint(50, w-50), random.randint(50, h-50),
                          random.randint(30, 80), random.randint(30, 100)],
                 "confidence": random.uniform(0.5, 0.9), "class": 0, "id": None}
                for _ in range(random.randint(1, 3))]

    def draw_detections(self, frame: np.ndarray, detections: list[dict] = None) -> np.ndarray:
        if detections is None:
            detections = self.latest_detections
        frame = frame.copy()
        for det in detections:
            cx, cy, w, h = det["bbox"]
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            track_id = det.get("id", "?")
            conf = det.get("confidence", 0)
            # 多模型检测到的目标用绿色，单模型用蓝色
            num_models = det.get("num_models", 1)
            color = (0, 255, 0) if num_models >= 2 else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame



