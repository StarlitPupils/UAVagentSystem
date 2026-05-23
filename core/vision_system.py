# core/vision_system.py (v1.2.5 - 简洁版)
"""视觉系统 - ensemble内部已含共识过滤"""
import cv2, numpy as np, os, time
from config.settings import config
try:
    from core.edge.tensorrt_exporter import TensorRTInference
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class VisionSystem:
    def __init__(self, device=None, use_ensemble=True):
        if device is None: device=config.DETECTION_DEVICE
        if device=="auto" or device=="cuda":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device=device; self.use_ensemble=use_ensemble
        self.fp16=config.DETECTION_FP16 and device=="cuda"
        self.preprocessor=None
        try:
            from core.detection.preprocessing import preprocessor
            self.preprocessor=preprocessor
        except: pass
        self.model=None; self.ensemble=None; self.tracker=None; self.tensorrt_model=None
        self.latest_detections=[]; self.detection_count=0; self.total_detections=0
        self.processing_times=[]
        self._init_model(); self._init_tensorrt(); self._init_tracker()

    def _init_model(self):
        if not ULTRALYTICS_AVAILABLE: return
        if os.path.exists(config.YOLO_MODEL_PATH):
            try:
                self.model=YOLO(config.YOLO_MODEL_PATH)
                if self.fp16: self.model.model.half()
                self.model.to(self.device)
            except Exception: self._try_download()
        else: self._try_download()
        if self.use_ensemble:
            from core.detection.ensemble_detector import EnsembleDetector
            self.ensemble=EnsembleDetector(device=self.device)
            if self.ensemble.count_models()<2: self.use_ensemble=False

    def _try_download(self):
        try:
            # 优先 engine
            engine_path = config.YOLO_MODEL_PATH.replace('.pt', '.engine')
            if os.path.exists(engine_path):
                self.model = YOLO(engine_path)
            else:
                self.model = YOLO("yolo11n.pt")
                if self.fp16: self.model.model.half()
                self.model.to(self.device)
        except: pass


    def _init_tensorrt(self):
        """尝试加载 TensorRT 引擎加速"""
        if not TENSORRT_AVAILABLE:
            return
        # 优先查找专用 engine 路径
        engine_path = getattr(config, 'TENSORRT_ENGINE_PATH', '')
        if not engine_path or not os.path.exists(engine_path):
            engine_path = config.YOLO_MODEL_PATH.replace('.pt', '.engine')
        if os.path.exists(engine_path):
            try:
                self.tensorrt_model = TensorRTInference(engine_path)
                print(f"[Vision] TensorRT 引擎已加载: {os.path.basename(engine_path)}")
            except Exception as e:
                print(f"[Vision] TensorRT 加载失败: {e}")

    def _init_tracker(self):
        from core.tracking.iou_tracker import EnhancedTracker
        cfg=config.TRACKER_REGISTRY.get(config.TRACKER_TYPE,{})
        self.tracker=EnhancedTracker(
            max_age=cfg.get('max_age',30), min_hits=cfg.get('min_hits',3),
            iou_threshold=cfg.get('iou_threshold',0.25),
            feature_weight=cfg.get('feature_weight',0.35),
            use_ekf=cfg.get('use_ekf',True),
            interpolate_gaps=cfg.get('interpolate_gaps',True),
            max_gap_frames=cfg.get('max_gap_frames',10))

    def process_frame(self, frame):
        start=time.time()
        if self.preprocessor: frame=self.preprocessor.enhance(frame,"auto")
        # 检测 (ensemble.detect已内置共识过滤)
        if self.use_ensemble and self.ensemble:
            raw_dets=self.ensemble.detect(frame)
        else:
            raw_dets=self._detect_single(frame)
        self.detection_count+=1; self.total_detections+=len(raw_dets)
        # 跟踪
        if self.tracker:
            try:
                tracked=self.tracker.update(raw_dets,frame)
                self.latest_detections=tracked
                self.processing_times.append((time.time()-start)*1000)
                if len(self.processing_times)>100: self.processing_times=self.processing_times[-100:]
                return tracked
            except Exception as e:
                print(f"[Vision] 跟踪失败: {e}")
        for i,d in enumerate(raw_dets): d["id"]=i
        self.latest_detections=raw_dets; return raw_dets

    def _detect_single(self, frame):
        if self.model is None: return self._mock_detections(frame)
        try:
            # TRT 引擎不能指定 device
            is_trt = '.engine' in str(getattr(self.model, 'model_name', ''))
            if is_trt:
                results = self.model(frame, conf=config.DETECTION_CONFIDENCE, verbose=False)
            else:
                results = self.model(frame, conf=config.DETECTION_CONFIDENCE, device=self.device, verbose=False, half=self.fp16)
            dets=[]
            if results and results[0].boxes is not None:
                for i in range(len(results[0].boxes)):
                    x1,y1,x2,y2=results[0].boxes.xyxy[i].tolist()
                    dets.append({"bbox":[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],
                                "confidence":float(results[0].boxes.conf[i]),
                                "class":int(results[0].boxes.cls[i]),"id":None,"num_models":1})
            return dets
        except: return self._mock_detections(frame)

    def detect_only(self, frame):
        if self.preprocessor: frame=self.preprocessor.enhance(frame,"auto")
        # 1. 多模型融合优先（内部已对各模型使用 TensorRT 引擎加速）
        if self.use_ensemble and self.ensemble:
            return self.ensemble.detect(frame)
        # 2. 单模型模式：优先 TensorRT
        if self.tensorrt_model is not None:
            try:
                dets = self.tensorrt_model.infer(frame, conf=config.DETECTION_CONFIDENCE)
                if dets:
                    return dets
            except Exception:
                pass
        # 3. 兜底 PyTorch
        return self._detect_single(frame)

    def get_stats(self):
        avg=np.mean(self.processing_times) if self.processing_times else 0
        return {"detection_count":self.detection_count,"total_detections":self.total_detections,
                "avg_per_frame":self.total_detections/max(self.detection_count,1),
                "avg_processing_ms":round(float(avg),1),
                "ensemble_mode":self.use_ensemble and self.ensemble is not None,
                "num_models":self.ensemble.count_models() if self.ensemble else 1,
                "model_names":self.ensemble.get_model_names() if self.ensemble else [config.YOLO_MODEL_NAME]}

    def reload_model(self): self._init_model()
    def switch_tracker(self, t=None):
        if t: config.TRACKER_TYPE=t
        self._init_tracker()

    def _mock_detections(self, frame):
        import random
        h,w=frame.shape[:2] if hasattr(frame,'shape') else (640,640)
        return [{"bbox":[random.randint(50,w-50),random.randint(50,h-50),
                random.randint(30,80),random.randint(30,100)],"confidence":random.uniform(0.5,0.9),
                "class":0,"id":None,"num_models":1} for _ in range(random.randint(1,3))]

    def draw_detections(self, frame, detections=None):
        if detections is None: detections=self.latest_detections
        frame=frame.copy()
        for det in detections:
            cx,cy,w,h=[float(v) for v in det["bbox"]]
            x1,y1=int(cx-w/2),int(cy-h/2); x2,y2=int(cx+w/2),int(cy+h/2)
            n=det.get("num_models",1); tid=det.get("id","?"); conf=det.get("confidence",0)
            color=(0,255,0) if n>=3 else ((255,200,0) if n>=2 else (255,100,0))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID:{tid} {conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
        return frame
