# core/tracking/tracker_registry.py
from .iou_tracker import EnhancedTracker

class BaseTracker:
    def __init__(self, **kwargs):
        self.name = "base"
        self.kwargs = kwargs
    def update(self, detections, frame=None):
        return detections
    def reset(self):
        pass

class ByteTrackWrapper(BaseTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bytetrack"
    def update(self, detections, frame=None):
        if not hasattr(self, '_next_id'):
            self._next_id = 0
        for d in detections:
            if d.get('id') is None:
                d['id'] = self._next_id
                self._next_id += 1
        return detections

class StrongSORTWrapper(BaseTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "strongsort"
        kw = kwargs if kwargs else {}
        self._tracker = EnhancedTracker(**kw)
    def update(self, detections, frame=None):
        return self._tracker.update(detections, frame)

class TransformerTrackerWrapper(BaseTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "transformer"
    def update(self, detections, frame=None):
        if not hasattr(self, '_next_id'):
            self._next_id = 0
        for d in detections:
            if d.get('id') is None:
                d['id'] = self._next_id
                self._next_id += 1
        return detections

TRACKER_MAP = {
    "bytetrack": ByteTrackWrapper,
    "strongsort": StrongSORTWrapper,
    "transformer": TransformerTrackerWrapper,
    "deepsort": StrongSORTWrapper,
    "botsort": ByteTrackWrapper,
    "iou": lambda **kw: EnhancedTracker(**kw),
}

class TrackerRegistry:
    def __init__(self):
        self.trackers = {}
    def get_tracker(self, tracker_type=None):
        from config.settings import config
        if tracker_type is None:
            tracker_type = config.TRACKER_TYPE
        if tracker_type not in self.trackers:
            cls = TRACKER_MAP.get(tracker_type, ByteTrackWrapper)
            kw = config.TRACKER_REGISTRY.get(tracker_type, {})
            self.trackers[tracker_type] = cls(**kw)
        return self.trackers[tracker_type]
    def list_available(self):
        return list(TRACKER_MAP.keys())

tracker_registry = TrackerRegistry()
def get_tracker(tracker_type=None):
    return tracker_registry.get_tracker(tracker_type)
