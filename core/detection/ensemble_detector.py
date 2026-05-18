# core/detection/ensemble_detector.py
"""多模型集成检测器 - Weighted Box Fusion (WBF)"""
import numpy as np
import os
from typing import List

from config.settings import config


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    b1 = np.zeros_like(boxes1)
    b2 = np.zeros_like(boxes2)
    b1[:,0]=boxes1[:,0]-boxes1[:,2]/2; b1[:,1]=boxes1[:,1]-boxes1[:,3]/2
    b1[:,2]=boxes1[:,0]+boxes1[:,2]/2; b1[:,3]=boxes1[:,1]+boxes1[:,3]/2
    b2[:,0]=boxes2[:,0]-boxes2[:,2]/2; b2[:,1]=boxes2[:,1]-boxes2[:,3]/2
    b2[:,2]=boxes2[:,0]+boxes2[:,2]/2; b2[:,3]=boxes2[:,1]+boxes2[:,3]/2
    inter_x1=np.maximum(b1[:,None,0],b2[None,:,0])
    inter_y1=np.maximum(b1[:,None,1],b2[None,:,1])
    inter_x2=np.minimum(b1[:,None,2],b2[None,:,2])
    inter_y2=np.minimum(b1[:,None,3],b2[None,:,3])
    inter=np.maximum(0,inter_x2-inter_x1)*np.maximum(0,inter_y2-inter_y1)
    area1=(b1[:,2]-b1[:,0])*(b1[:,3]-b1[:,1])
    area2=(b2[:,2]-b2[:,0])*(b2[:,3]-b2[:,1])
    return inter/(area1[:,None]+area2[None,:]-inter+1e-8)


def weighted_box_fusion(detections_list, iou_thr=0.55, conf_type="weighted_avg"):
    all_boxes,all_confs,all_classes,all_weights=[],[],[],[]
    for mi,dets in enumerate(detections_list):
        if not dets: continue
        w=getattr(config,'MODEL_WEIGHTS',[1.0,0.8,0.6])[mi] if getattr(config,'MODEL_WEIGHTS',None) else 1.0
        for d in dets:
            all_boxes.append(d.get('bbox',[0,0,0,0]))
            all_confs.append(d.get('confidence',0.5))
            all_classes.append(d.get('class',0))
            all_weights.append(w)
    if not all_boxes: return []
    boxes=np.array(all_boxes,dtype=np.float32); confs=np.array(all_confs,dtype=np.float32)
    classes=np.array(all_classes,dtype=np.int32); weights=np.array(all_weights,dtype=np.float32)
    idx=np.argsort(-confs)
    boxes=boxes[idx]; confs=confs[idx]; classes=classes[idx]; weights=weights[idx]
    fused=[]; used=np.zeros(len(boxes),dtype=bool)
    for i in range(len(boxes)):
        if used[i]: continue
        cluster=[i]
        for j in range(i+1,len(boxes)):
            if used[j] or classes[i]!=classes[j]: continue
            if compute_iou_matrix(boxes[i:i+1],boxes[j:j+1])[0,0]>=iou_thr:
                cluster.append(j)
        for c in cluster: used[c]=True
        cb=boxes[cluster]; cc=confs[cluster]; cw=weights[cluster]
        if conf_type=="avg": fc=np.mean(cc)
        elif conf_type=="max": fc=np.max(cc)
        else: fc=np.average(cc,weights=cw)
        tw=np.sum(cw*cc); fb=np.sum(cb*(cw*cc)[:,None],axis=0)/(tw+1e-8)
        fused.append({'bbox':fb.tolist(),'confidence':float(fc),'class':int(classes[i]),'num_models':len(cluster),'id':None})
    fused.sort(key=lambda x:x['confidence'],reverse=True)
    return fused


class EnsembleDetector:
    def __init__(self, model_paths=None, device=None):
        if device is None: device=config.DETECTION_DEVICE
        self.device=device; self.models=[]; self.model_names=[]
        if model_paths is None:
            base=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_paths=[os.path.join(base,"models","yolo11x.pt"),
                         os.path.join(base,"models","yolov8x.pt"),
                         os.path.join(base,"models","yolo11n.pt")]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    m=YOLO(path); m.to(device)
                    self.models.append(m)
                    self.model_names.append(os.path.basename(path).replace('.pt',''))
                    print(f"[Ensemble] + {os.path.basename(path)}")
                except Exception as e:
                    print(f"[Ensemble] fail {path}: {e}")
        print(f"[Ensemble] {len(self.models)} models: {self.model_names}")

    def detect(self, frame, conf_thr=None):
        if conf_thr is None: conf_thr=config.DETECTION_CONFIDENCE
        if not self.models: return []
        all_dets=[]
        for model in self.models:
            try:
                results=model(frame,conf=conf_thr,device=self.device,verbose=False)
                dets=[]
                if results and results[0].boxes is not None:
                    for i in range(len(results[0].boxes)):
                        x1,y1,x2,y2=results[0].boxes.xyxy[i].tolist()
                        dets.append({"bbox":[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],
                                     "confidence":float(results[0].boxes.conf[i]),
                                     "class":int(results[0].boxes.cls[i]),"id":None})
                all_dets.append(dets)
            except Exception as e:
                print(f"[Ensemble] err: {e}"); all_dets.append([])
        fused=weighted_box_fusion(all_dets,iou_thr=0.55,conf_type="weighted_avg")
        print(f"[Ensemble] {[len(d) for d in all_dets]} -> {len(fused)}")
        return fused

    def count_models(self): return len(self.models)
    def get_model_names(self): return self.model_names
