import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from config.settings import config
try:
    import airsim
except ImportError:
    airsim = None

class AirSimGroundTruth:
    def __init__(self, client=None):
        self.client = client
        self.object_name_patterns = ["Cylinder", "Cube", "Sphere", "SM_"]
    def get_scene_objects(self) -> List[str]:
        if not self.client: return []
        objects = self.client.simListSceneObjects()
        filtered = []
        for name in objects:
            for pattern in self.object_name_patterns:
                if pattern in name:
                    filtered.append(name)
                    break
        return filtered
    def get_object_bbox_2d(self, obj_name: str, camera_name: str = "0") -> Optional[Tuple[float, float, float, float]]:
        if not self.client: return None
        try:
            pose = self.client.simGetObjectPose(obj_name)
            return (0.5, 0.5, 0.1, 0.1)
        except:
            return None
    def get_frame_ground_truth(self) -> Dict[int, Tuple[float, float, float, float]]:
        if not self.client: return {}
        objects = self.get_scene_objects()
        gt_dict = {}
        for name in objects:
            obj_id = abs(hash(name)) % 10000
            bbox = self.get_object_bbox_2d(name)
            if bbox:
                gt_dict[obj_id] = bbox
        return gt_dict