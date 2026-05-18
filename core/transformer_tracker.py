import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

class TransformerTracker:
    def __init__(self, device: str = 'cpu', feat_dim: int = 256, num_heads: int = 4):
        self.device = device
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True, device=device)
        self.norm = nn.LayerNorm(feat_dim).to(device)
        self.tracks: Dict[int, Dict] = {}
        self.next_id: int = 0
        self.max_age: int = 30
        self.min_hits: int = 3

    def extract_features(self, bboxes: List[Tuple[float, float, float, float]], image: Optional[np.ndarray] = None) -> torch.Tensor:
        if len(bboxes) == 0:
            return torch.empty((0, self.feat_dim), device=self.device)
        feats = []
        for bbox in bboxes:
            x, y, w, h = bbox
            pos = torch.tensor([x/640, y/640, w/640, h/640, w/(h+1e-6)], dtype=torch.float32, device=self.device)
            feat = pos.repeat(self.feat_dim // 5 + 1)[:self.feat_dim]
            feats.append(feat)
        return torch.stack(feats)

    def update(self, detections: List[Dict], image: Optional[np.ndarray] = None) -> Dict[int, Dict]:
        det_bboxes = [d['bbox'] for d in detections]
        det_feats = self.extract_features(det_bboxes, image)
        track_ids = list(self.tracks.keys())
        track_feats = torch.stack([self.tracks[tid]['features'] for tid in track_ids]) if track_ids else torch.empty((0, self.feat_dim), device=self.device)
        if len(det_feats) > 0 and len(track_feats) > 0:
            combined = torch.cat([det_feats, track_feats], dim=0)
            attn_output, _ = self.self_attn(combined, combined, combined)
            attn_output = self.norm(attn_output + combined)
            det_feats_enh = attn_output[:len(det_feats)]
            track_feats_enh = attn_output[len(det_feats):]
            det_feats_norm = nn.functional.normalize(det_feats_enh, dim=1)
            track_feats_norm = nn.functional.normalize(track_feats_enh, dim=1)
            cost_matrix = 1 - torch.mm(det_feats_norm, track_feats_norm.t()).detach().cpu().numpy()
        else:
            cost_matrix = np.zeros((len(detections), len(track_ids)))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches, unmatched_det, unmatched_track = [], set(range(len(detections))), set(range(len(track_ids)))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.5:
                matches.append((r, c))
                unmatched_det.discard(r)
                unmatched_track.discard(c)
        for det_idx, track_idx in matches:
            tid = track_ids[track_idx]
            det = detections[det_idx]
            self.tracks[tid].update({'bbox': det['bbox'], 'confidence': det['confidence'], 'class_id': det['class_id'], 'features': det_feats[det_idx].detach(), 'age': 0, 'hits': self.tracks[tid]['hits']+1})
        for det_idx in unmatched_det:
            det = detections[det_idx]
            self.tracks[self.next_id] = {'bbox': det['bbox'], 'confidence': det['confidence'], 'class_id': det['class_id'], 'features': det_feats[det_idx].detach() if len(det_feats)>0 else torch.zeros(self.feat_dim, device=self.device), 'age': 0, 'hits': 1}
            self.next_id += 1
        for track_idx in unmatched_track:
            tid = track_ids[track_idx]
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]
        results = {}
        for tid, track in self.tracks.items():
            if track['hits'] >= self.min_hits and track['age'] == 0:
                results[tid] = {'bbox': track['bbox'], 'id': tid, 'confidence': track['confidence'], 'class_id': track['class_id'], 'class_name': 'object'}
        return results

    def reset(self):
        self.tracks.clear()
        self.next_id = 0