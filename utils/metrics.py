import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TrackingMetrics:
    MOTA: float = 0.0; MOTP: float = 0.0; IDF1: float = 0.0; HOTA: float = 0.0
    DetA: float = 0.0; AssA: float = 0.0; MT: int = 0; ML: int = 0; PT: int = 0
    IDSW: int = 0; Frag: int = 0; FP: int = 0; FN: int = 0; TP: int = 0
    GT_Objects: int = 0; Recall: float = 0.0; Precision: float = 0.0
    def to_dict(self) -> Dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}

class MOTEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    @staticmethod
    def compute_iou(box1: Tuple, box2: Tuple) -> float:
        x1,y1,w1,h1=box1; x2,y2,w2,h2=box2
        b1=[x1-w1/2,y1-h1/2,x1+w1/2,y1+h1/2]
        b2=[x2-w2/2,y2-h2/2,x2+w2/2,y2+h2/2]
        inter_x1,inter_y1=max(b1[0],b2[0]),max(b1[1],b2[1])
        inter_x2,inter_y2=min(b1[2],b2[2]),min(b1[3],b2[3])
        inter_area=max(0,inter_x2-inter_x1)*max(0,inter_y2-inter_y1)
        area1,area2=w1*h1,w2*h2
        union=area1+area2-inter_area
        return inter_area/union if union>0 else 0.0
    def evaluate(self, gt_tracks: Dict[int, Dict[int, Tuple]], pred_tracks: Dict[int, Dict[int, Tuple]]) -> TrackingMetrics:
        metrics=TrackingMetrics()
        all_frames=sorted(set(gt_tracks.keys())|set(pred_tracks.keys()))
        total_FP=total_FN=total_TP=total_gt_objects=motp_count=id_switches=0
        total_motp_sum=0.0
        gt_traj_lengths=defaultdict(int); gt_traj_matched=defaultdict(int)
        prev_matches: Dict[int,int]={}
        for frame_id in all_frames:
            gt_frame=gt_tracks.get(frame_id,{})
            pred_frame=pred_tracks.get(frame_id,{})
            total_gt_objects+=len(gt_frame)
            for gt_id in gt_frame: gt_traj_lengths[gt_id]+=1
            if not gt_frame and not pred_frame: continue
            gt_ids=list(gt_frame.keys()); pred_ids=list(pred_frame.keys())
            iou_matrix=np.zeros((len(gt_ids),len(pred_ids)))
            for i,gt_id in enumerate(gt_ids):
                for j,pred_id in enumerate(pred_ids):
                    iou_matrix[i,j]=self.compute_iou(gt_frame[gt_id],pred_frame[pred_id])
            matched_gt=set(); matched_pred=set(); matches=[]
            iou_flat=[]
            for i in range(len(gt_ids)):
                for j in range(len(pred_ids)):
                    if iou_matrix[i,j]>=self.iou_threshold:
                        iou_flat.append((iou_matrix[i,j],i,j))
            iou_flat.sort(reverse=True,key=lambda x:x[0])
            for iou,i,j in iou_flat:
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i); matched_pred.add(j)
                    gt_id=gt_ids[i]; pred_id=pred_ids[j]
                    matches.append((gt_id,pred_id,iou))
                    total_motp_sum+=iou; motp_count+=1
                    gt_traj_matched[gt_id]+=1
            TP=len(matches); FP=len(pred_frame)-TP; FN=len(gt_frame)-TP
            total_TP+=TP; total_FP+=FP; total_FN+=FN
            current_matches={gt_id:pred_id for gt_id,pred_id,_ in matches}
            for gt_id,pred_id in current_matches.items():
                if gt_id in prev_matches and prev_matches[gt_id]!=pred_id:
                    id_switches+=1
            prev_matches=current_matches
        metrics.TP=total_TP; metrics.FP=total_FP; metrics.FN=total_FN; metrics.GT_Objects=total_gt_objects
        if total_gt_objects>0: metrics.Recall=total_TP/total_gt_objects
        if total_TP+total_FP>0: metrics.Precision=total_TP/(total_TP+total_FP)
        metrics.MOTA=1.0-(total_FN+total_FP+id_switches)/max(1,total_gt_objects)
        metrics.MOTA=max(0.0,metrics.MOTA)
        metrics.MOTP=total_motp_sum/max(1,motp_count)
        idtp=total_TP; idfp=total_FP; idfn=total_FN
        if idtp>0: metrics.IDF1=2*idtp/(2*idtp+idfp+idfn)
        else: metrics.IDF1=0.0
        total_gt_trajs=len(gt_traj_lengths)
        for gt_id,length in gt_traj_lengths.items():
            coverage=gt_traj_matched[gt_id]/length if length>0 else 0
            if coverage>=0.8: metrics.MT+=1
            elif coverage<=0.2: metrics.ML+=1
            else: metrics.PT+=1
        metrics.IDSW=id_switches; metrics.Frag=0
        metrics.HOTA=np.sqrt(metrics.DetA*metrics.AssA) if metrics.DetA and metrics.AssA else 0.0
        return metrics