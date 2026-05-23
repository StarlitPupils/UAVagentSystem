# core/detection/ensemble_detector.py (v1.2.8 stable)
import numpy as np, os, cv2
from config.settings import config

def compute_iou_matrix(b1,b2):
    if len(b1)==0 or len(b2)==0: return np.zeros((len(b1),len(b2)),dtype=np.float32)
    b1_=np.zeros_like(b1); b2_=np.zeros_like(b2)
    b1_[:,0]=b1[:,0]-b1[:,2]/2; b1_[:,1]=b1[:,1]-b1[:,3]/2
    b1_[:,2]=b1[:,0]+b1[:,2]/2; b1_[:,3]=b1[:,1]+b1[:,3]/2
    b2_[:,0]=b2[:,0]-b2[:,2]/2; b2_[:,1]=b2[:,1]-b2[:,3]/2
    b2_[:,2]=b2[:,0]+b2[:,2]/2; b2_[:,3]=b2[:,1]+b2[:,3]/2
    ix1=np.maximum(b1_[:,None,0],b2_[None,:,0]); iy1=np.maximum(b1_[:,None,1],b2_[None,:,1])
    ix2=np.minimum(b1_[:,None,2],b2_[None,:,2]); iy2=np.minimum(b1_[:,None,3],b2_[None,:,3])
    inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    a1=(b1_[:,2]-b1_[:,0])*(b1_[:,3]-b1_[:,1]); a2=(b2_[:,2]-b2_[:,0])*(b2_[:,3]-b2_[:,1])
    return inter/(a1[:,None]+a2[None,:]-inter+1e-8)

def weighted_box_fusion(dets_list, iou_thr=0.40, weights=None, model_names=None):
    all_boxes, all_confs, all_cls, all_w, all_mid, all_mname = [], [], [], [], [], []
    if weights is None:
        weights = config.MODEL_WEIGHTS[:len(dets_list)]
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(dets_list))]
    for mi, dets in enumerate(dets_list):
        if not dets:
            continue
        w = weights[mi] if mi < len(weights) else 0.5
        mname = model_names[mi] if mi < len(model_names) else f"model_{mi}"
        for d in dets:
            all_boxes.append(d['bbox'])
            all_confs.append(d.get('confidence', 0.5))
            all_cls.append(d.get('class', 0))
            all_w.append(w)
            all_mid.append(mi)
            all_mname.append(mname)
    if not all_boxes:
        return []
    boxes = np.array(all_boxes, dtype=np.float32)
    confs = np.array(all_confs, dtype=np.float32)
    classes = np.array(all_cls, dtype=np.int32)
    wts = np.array(all_w, dtype=np.float32)
    mids = np.array(all_mid, dtype=np.int32)
    mnames = np.array(all_mname)
    idx = np.argsort(-confs)
    boxes = boxes[idx]; confs = confs[idx]; classes = classes[idx]
    wts = wts[idx]; mids = mids[idx]; mnames = mnames[idx]
    fused = []
    used = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if used[i]:
            continue
        cluster = [i]
        models = {mids[i]}
        model_names_cluster = [mnames[i]]
        for j in range(i + 1, len(boxes)):
            if used[j] or classes[i] != classes[j]:
                continue
            if compute_iou_matrix(boxes[i:i+1], boxes[j:j+1])[0, 0] >= iou_thr:
                cluster.append(j)
                models.add(mids[j])
                model_names_cluster.append(mnames[j])
        for c in cluster:
            used[c] = True
        cb = boxes[cluster]
        cc = confs[cluster]
        cw = wts[cluster]
        fc = np.average(cc, weights=cw)
        tw = np.sum(cw * cc)
        fb = np.sum(cb * (cw * cc)[:, None], axis=0) / (tw + 1e-8)
        fused.append({
            'bbox': fb.tolist(),
            'confidence': float(fc),
            'class': int(classes[i]),
            'num_models': len(models),
            'id': None,
            'model_names': list(set(model_names_cluster)),
        })
    fused.sort(key=lambda x: x['confidence'], reverse=True)
    return fused


def soft_consensus_filter_v13(detections, primary_model="yolo11x_visdrone"):
    """v1.3.1 自适应共识过滤：主力模型放宽阈值"""
    kept=[]
    stats={'3+':0,'2+':0,'primary':0,'hi_conf':0,'DROP':0}
    for d in detections:
        n=d.get('num_models',1); c=d.get('confidence',0)
        model_names = d.get('model_names', [])
        has_primary = primary_model in model_names if model_names else False
        
        if n>=3:
            kept.append(d); stats['3+']+=1
        elif n>=2 and c>=0.30:
            kept.append(d); stats['2+']+=1
        elif has_primary and c>=0.45:    # 主力模型检测，降低阈值
            kept.append(d); stats['primary']+=1
        elif c>=0.60:
            kept.append(d); stats['hi_conf']+=1
        else:
            stats['DROP']+=1
    
    if stats['DROP']:
        print(f"  [FILTER v1.3] kept:{len(kept)} drop:{stats['DROP']}/{len(detections)} "
              f"(3+:{stats['3+']} 2+:{stats['2+']} primary:{stats['primary']} hi:{stats['hi_conf']})")
    return kept
class EnsembleDetector:
    def __init__(self, model_paths=None, device=None):
        if device is None: device=config.DETECTION_DEVICE
        if device=="cuda":
            import torch
            if not torch.cuda.is_available(): device="cpu"
        self.device=device; self.models=[]; self.model_names=[]
        self.fp16=config.DETECTION_FP16 and device=="cuda"
        if model_paths is None:
            base=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_dir=os.path.join(base,"models")
            if os.path.isdir(model_dir):
                model_paths=[os.path.join(model_dir,f) for f in os.listdir(model_dir) if f.endswith('.pt')]
                priority=['yolo11x_visdrone.pt','yolo11x.pt','yolov8x.pt','yolo11n.pt','yolov10n.pt','rtdetr-l.pt']
                model_paths.sort(key=lambda p: priority.index(os.path.basename(p)) if os.path.basename(p) in priority else 999)
                model_paths=model_paths[:5]
        for p in (model_paths or []):
            if os.path.exists(p): self._load_model(p)
        if not self.models: self._auto_download()
        print(f"[Ensemble] {len(self.models)} models, conf={config.DETECTION_CONFIDENCE}, IoU={config.ENSEMBLE_IOU_THR}")

    def _load_model(self, path):
        """加载模型：优先 TensorRT engine，否则 PyTorch"""
        try:
            from ultralytics import YOLO
            engine_path = path.replace('.pt', '.engine')
            if os.path.exists(engine_path):
                # TensorRT engine：不能调用 .to() 或 .half()
                m = YOLO(engine_path)
                self.models.append(m)
                self.model_names.append(os.path.basename(engine_path).replace('.engine', ''))
                print(f"  [TRT engine] {os.path.basename(engine_path)}")
            else:
                # PyTorch 模型
                m = YOLO(path)
                if self.fp16 and self.device == 'cuda':
                    m.model.half()
                m.to(self.device)
                self.models.append(m)
                self.model_names.append(os.path.basename(path).replace('.pt', ''))
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
    def _auto_download(self):
        for name in ['yolo11n.pt','yolo11x.pt']:
            try:
                from ultralytics import YOLO
                m=YOLO(name)
                if self.fp16: m.model.half()
                m.to(self.device)
                self.models.append(m); self.model_names.append(name.replace('.pt',''))
            except: pass

    def detect(self, frame, conf_thr=None):
        if conf_thr is None: conf_thr=config.DETECTION_CONFIDENCE
        if not self.models: return []
        all_dets=[]
        for i, model in enumerate(self.models):
            dets=[]
            try:
                # 优先尝试无 device/half 参数（TRT 引擎兼容）
                results=model(frame, conf=conf_thr, verbose=False)
            except Exception:
                try:
                    # 回退到带 device/half 参数（PyTorch 模型）
                    results=model(frame, conf=conf_thr, device=self.device, verbose=False, half=self.fp16)
                except Exception as e:
                    print(f"  [Warn] Model {self.model_names[i]}: {e}")
                    results=None
            if results is not None and len(results)>0 and results[0].boxes is not None:
                for k in range(len(results[0].boxes)):
                    x1,y1,x2,y2=results[0].boxes.xyxy[k].tolist()
                    dets.append({"bbox":[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],
                                 "confidence":float(results[0].boxes.conf[k]),
                                 "class":int(results[0].boxes.cls[k]),"id":None})
            all_dets.append(dets)
        det_counts=[len(d) for d in all_dets]
        print(f"  [Detect] per-model dets: {det_counts}")
        fused=weighted_box_fusion(all_dets, iou_thr=0.40, model_names=self.model_names)
        fused=soft_consensus_filter(fused)
        MAX_DETS=150
        if len(fused)>MAX_DETS:
            fused=sorted(fused,key=lambda x:x['confidence'],reverse=True)[:MAX_DETS]
        print(f"  [Ensemble] {det_counts} -> {len(fused)} dets")
        return fused
    def count_models(self): return len(self.models)
    def get_model_names(self): return self.model_names
    def get_stats(self): return {"num_models":len(self.models),"model_names":self.model_names}

# 别名：保持向后兼容
soft_consensus_filter = soft_consensus_filter_v13
