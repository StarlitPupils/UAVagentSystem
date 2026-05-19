import json
import time
import os
import csv
from typing import Dict, Optional
from config.settings import config
from core.vision_system import VisionSystem
from simulation.ground_truth import AirSimGroundTruth
from utils.metrics import MOTEvaluator, TrackingMetrics

class TrackingEvaluator:
    def __init__(self, vision_system: VisionSystem, client=None):
        self.vision = vision_system
        self.gt_collector = AirSimGroundTruth(client)
        self.evaluator = MOTEvaluator(iou_threshold=0.5)
        self.gt_tracks: Dict[int, Dict[int, tuple]] = {}
        self.pred_tracks: Dict[int, Dict[int, tuple]] = {}
    def reset(self):
        self.gt_tracks.clear()
        self.pred_tracks.clear()
    def record_frame(self, frame_id: int):
        gt = self.gt_collector.get_frame_ground_truth()
        if gt:
            self.gt_tracks[frame_id] = gt
        pred = {}
        for det in self.vision.latest_detections:
            track_id = det['id']
            bbox = det['bbox']
            pred[track_id] = bbox
        self.pred_tracks[frame_id] = pred
    def evaluate(self) -> TrackingMetrics:
        return self.evaluator.evaluate(self.gt_tracks, self.pred_tracks)
    def save_report(self, metrics: TrackingMetrics, filename: str = None):
        if filename is None:
            filename = f"eval_{int(time.time())}"
        json_path = os.path.join(config.REPORT_DIR, f"{filename}.json")
        csv_path = os.path.join(config.REPORT_DIR, f"{filename}.csv")
        report = {"timestamp": time.time(), "settings": {"iou_threshold": self.evaluator.iou_threshold}, "metrics": metrics.to_dict()}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.to_dict().keys())
            writer.writeheader()
            writer.writerow(metrics.to_dict())
        print(f"[Evaluator] 报告已保存至 {json_path} 和 {csv_path}")
        return report