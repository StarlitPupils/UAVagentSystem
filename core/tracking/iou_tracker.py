# core/tracking/iou_tracker.py (v1.2.8 stable - min_hits=1)
import numpy as np, cv2
from typing import List

def _safe_float(val):
    if isinstance(val, (np.ndarray,)):
        if val.size == 0: return 0.0
        return float(val.flat[0])
    return float(val)

class KalmanTracker:
    count = 0
    def __init__(self, bbox, confidence, cls, frame=None):
        self.id = KalmanTracker.count; KalmanTracker.count += 1
        self.state = np.zeros((8,1),dtype=np.float32)
        self.state[0]=_safe_float(bbox[0]); self.state[1]=_safe_float(bbox[1])
        self.state[2]=max(1.0,_safe_float(bbox[2])); self.state[3]=max(1.0,_safe_float(bbox[3]))
        self.P=np.eye(8,dtype=np.float32)*10.0; self.P[4:,4:]*=100.0
        self.Q=np.eye(8,dtype=np.float32)*0.01; self.Q[4:,4:]*=0.1
        self.R=np.eye(4,dtype=np.float32)*0.1
        self.bbox=[_safe_float(v) for v in bbox]; self.confidence=_safe_float(confidence)
        self.cls=int(cls) if cls else 0; self.age=0; self.hits=1; self.time_since_update=0
        self.features=None; self.trajectory=[]
        if frame is not None: self._extract_features(frame,bbox)

    def _extract_features(self, frame, bbox):
        # v1.3: 优先使用 ReID 特征提取器
        try:
            from core.tracking.reid_features import reid_extractor
            feat = reid_extractor.extract(frame, bbox)
            if feat is not None and len(feat) > 0:
                self.features = feat.astype(np.float32)
                return
        except Exception:
            pass
        
        # 降级: HSV直方图
        try:
            cx=int(_safe_float(bbox[0])); cy=int(_safe_float(bbox[1]))
            w=max(1,int(_safe_float(bbox[2]))); h=max(1,int(_safe_float(bbox[3])))
            x1=max(0,cx-w//2); y1=max(0,cy-h//2)
            x2=min(frame.shape[1]-1,cx+w//2); y2=min(frame.shape[0]-1,cy+h//2)
            if x2>x1+3 and y2>y1+3:
                roi=frame[y1:y2,x1:x2]
                if roi.size>0:
                    hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
                    hist=cv2.calcHist([hsv],[0,1],None,[8,8],[0,180,0,256])
                    cv2.normalize(hist,hist)
                    self.features=hist.flatten().astype(np.float32)
        except: self.features=np.ones(64,dtype=np.float32)*0.5

    def predict(self):
        F=np.eye(8,dtype=np.float32)
        F[0,4]=1.0; F[1,5]=1.0; F[2,6]=1.0; F[3,7]=1.0
        self.state=F@self.state; self.P=F@self.P@F.T+self.Q
        self.age+=1; self.time_since_update+=1
        self.state[2]=max(1.0,_safe_float(self.state[2]))
        self.state[3]=max(1.0,_safe_float(self.state[3]))
        return [_safe_float(self.state[0]),_safe_float(self.state[1]),
                _safe_float(self.state[2]),_safe_float(self.state[3])]

    def update(self, bbox, confidence, cls, frame=None):
        self.time_since_update=0; self.hits+=1
        self.confidence=_safe_float(confidence); self.cls=int(cls) if cls else 0
        self.bbox=[_safe_float(v) for v in bbox]
        H=np.zeros((4,8),dtype=np.float32)
        H[0,0]=1.0; H[1,1]=1.0; H[2,2]=1.0; H[3,3]=1.0
        z=np.array([[_safe_float(bbox[0])],[_safe_float(bbox[1])],
                     [_safe_float(bbox[2])],[_safe_float(bbox[3])]],dtype=np.float32)
        y=z-H@self.state; S=H@self.P@H.T+self.R
        try:
            K=self.P@H.T@np.linalg.inv(S+np.eye(4,dtype=np.float32)*1e-4)
            self.state=self.state+K@y; self.P=(np.eye(8,dtype=np.float32)-K@H)@self.P
        except:
            self.state[0]=_safe_float(bbox[0]); self.state[1]=_safe_float(bbox[1])
        self.trajectory.append(bbox)
        if len(self.trajectory)>50: self.trajectory=self.trajectory[-50:]
        if frame is not None: self._extract_features(frame,bbox)

def iou(box1,box2):
    a=[_safe_float(v) for v in box1]; b=[_safe_float(v) for v in box2]
    x1,y1=a[0]-a[2]/2,a[1]-a[3]/2; x2,y2=b[0]-b[2]/2,b[1]-b[3]/2
    ix1,iy1=max(x1,x2),max(y1,y2); ix2,iy2=min(x1+a[2],x2+b[2]),min(y1+a[3],y2+b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); area1,area2=a[2]*a[3],b[2]*b[3]
    return inter/(area1+area2-inter+1e-8)

def feature_distance(f1,f2):
    if f1 is None or f2 is None: return 0.5
    try:
        f1=np.array(f1,dtype=np.float32).flatten(); f2=np.array(f2,dtype=np.float32).flatten()
        dim=min(len(f1),len(f2))
        if dim<1: return 0.5
        f1=f1[:dim]; f2=f2[:dim]
        dot=np.dot(f1,f2); n1=np.linalg.norm(f1); n2=np.linalg.norm(f2)
        if n1<1e-8 or n2<1e-8: return 0.5
        return float(np.clip(1.0-dot/(n1*n2),0.0,1.0))
    except: return 0.5

class EnhancedTracker:
    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.25, feature_weight=0.35,
                 use_ekf=True, interpolate_gaps=True, max_gap_frames=10):
        self.name='enhanced'; self.max_age=max_age; self.min_hits=min_hits
        self.iou_threshold=iou_threshold; self.feature_weight=feature_weight
        self.interpolate_gaps=interpolate_gaps; self.max_gap_frames=max_gap_frames
        self.tracks=[]; self.frame_count=0; self.lost_tracks={}

    def update(self, detections, frame=None):
        self.frame_count+=1
        predictions=[t.predict() for t in self.tracks]
        if len(detections)==0:
            for t in self.tracks: t.update(t.bbox,t.confidence,t.cls,None)
            self._cleanup(); return self._get_active(frame)
        n_dets=len(detections); n_tracks=len(self.tracks)
        cost=np.full((n_dets,max(1,n_tracks)),1e9,dtype=np.float32)
        det_feats=[]
        for det in detections:
            try:
                tmp=KalmanTracker(det['bbox'],det.get('confidence',0.5),det.get('class',0),frame)
                det_feats.append(tmp.features)
            except: det_feats.append(None)
        for i,det in enumerate(detections):
            for j,trk in enumerate(self.tracks):
                iou_val=iou(det['bbox'],predictions[j])
                if iou_val<0.01: continue
                feat_cost=feature_distance(det_feats[i],trk.features)
                cost[i,j]=(1-self.feature_weight)*(1-iou_val)+self.feature_weight*feat_cost
        matched_d,matched_t=set(),set()
        if n_dets>0 and n_tracks>0:
            try:
                from scipy.optimize import linear_sum_assignment
                ri,ci=linear_sum_assignment(cost)
            except:
                ri=list(range(min(n_dets,n_tracks))); ci=list(range(min(n_dets,n_tracks)))
            for r,c in zip(ri,ci):
                if cost[r,c]<1e8 and cost[r,c]<0.7:
                    self.tracks[c].update(detections[r]['bbox'],
                        detections[r].get('confidence',0.5),detections[r].get('class',0),frame)
                    matched_d.add(r); matched_t.add(c)
        for i in range(n_dets):
            if i not in matched_d:
                self.tracks.append(KalmanTracker(detections[i]['bbox'],
                    detections[i].get('confidence',0.5),detections[i].get('class',0),frame))
        for j in range(n_tracks):
            if j not in matched_t:
                self.tracks[j].update(predictions[j],self.tracks[j].confidence,self.tracks[j].cls,None)
        self._cleanup()
        return self._get_active(frame)

    def _cleanup(self):
        active=[]
        for t in self.tracks:
            if t.time_since_update<=self.max_age: active.append(t)
            elif self.interpolate_gaps:
                self.lost_tracks[t.id]={'track':t,'lost_frame':self.frame_count,'last_bbox':t.bbox}
        self.tracks=active
        expired=[tid for tid,info in self.lost_tracks.items() if self.frame_count-info['lost_frame']>self.max_gap_frames*2]
        for tid in expired: del self.lost_tracks[tid]

    def _get_active(self,frame=None):
        res=[]
        for t in self.tracks:
            if t.hits>=self.min_hits and t.time_since_update==0:
                res.append({'bbox':[_safe_float(v) for v in t.bbox],'id':int(t.id),
                           'confidence':_safe_float(t.confidence),'class':int(t.cls),
                           'class_name':'object','hits':int(t.hits),'age':int(t.age),'num_models':1})
        if self.interpolate_gaps and frame is not None:
            for tid,info in list(self.lost_tracks.items()):
                gap=self.frame_count-info['lost_frame']
                if gap>self.max_gap_frames: continue
                t=info['track']
                if len(t.trajectory)>=2:
                    try:
                        p1=np.array(t.trajectory[-2]); p2=np.array(t.trajectory[-1])
                        v=p2-p1; interp=p2+v*gap
                        interp_bbox=[_safe_float(interp[0]),_safe_float(interp[1]),
                                     max(5.0,min(500.0,_safe_float(interp[2]))),
                                     max(5.0,min(500.0,_safe_float(interp[3])))]
                        res.append({'bbox':interp_bbox,'id':int(tid),
                                   'confidence':_safe_float(t.confidence)*0.7,'class':int(t.cls),
                                   'class_name':'object','hits':int(t.hits),'age':int(t.age),'interpolated':True})
                    except: pass
        return res

    def reset(self):
        self.tracks.clear(); self.lost_tracks.clear(); self.frame_count=0; KalmanTracker.count=0
