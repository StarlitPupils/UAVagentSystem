import os
import cv2
import numpy as np
from typing import List, Dict, Optional

class VisDroneLoader:
    def __init__(self, dataset_root: str, sequence_name: str):
        self.sequence_path = os.path.join(dataset_root, "sequences", sequence_name)
        self.annotations_path = os.path.join(dataset_root, "annotations", sequence_name)
        if not os.path.exists(self.sequence_path):
            raise FileNotFoundError(f"序列路径不存在: {self.sequence_path}")
        self.image_files = sorted([f for f in os.listdir(self.sequence_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if not self.image_files:
            raise ValueError(f"序列 {sequence_name} 中没有图像文件")
        self.frame_idx = 0
        print(f"[VisDrone] 已加载序列 {sequence_name}，共 {len(self.image_files)} 帧")
    def get_next_frame(self) -> Optional[np.ndarray]:
        if self.frame_idx >= len(self.image_files):
            return None
        img_path = os.path.join(self.sequence_path, self.image_files[self.frame_idx])
        frame = cv2.imread(img_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_idx += 1
        return frame
    def reset(self):
        self.frame_idx = 0
    def get_frame_count(self) -> int:
        return len(self.image_files)
    def get_ground_truth(self, frame_number: int) -> List[Dict]:
        gt_file = os.path.join(self.annotations_path, "gt.txt")
        if not os.path.exists(gt_file):
            return []
        gts = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                fno = int(parts[0])
                if fno != frame_number: continue
                obj_id = int(parts[1])
                x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
                cls_id = int(parts[7])
                gts.append({"id": obj_id, "bbox": (x, y, w, h), "class_id": cls_id, "class_name": self._class_id_to_name(cls_id)})
        return gts
    def _class_id_to_name(self, cls_id: int) -> str:
        names = {0:'ignored',1:'pedestrian',2:'people',3:'bicycle',4:'car',5:'van',6:'truck',7:'tricycle',8:'awning-tricycle',9:'bus',10:'motor',11:'others'}
        return names.get(cls_id, 'unknown')