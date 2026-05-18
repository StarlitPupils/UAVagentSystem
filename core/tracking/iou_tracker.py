# core/tracking/iou_tracker.py
"""增强跟踪器 v4 - 稳健卡尔曼 + 外观匹配 + IoU"""
import numpy as np
import cv2

class KalmanBoxTracker:
    """单目标卡尔曼跟踪器（8 状态：cx,cy,w,h,vx,vy,vw,vh）"""
    count = 0

    def __init__(self, bbox, confidence, cls, frame=None):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # 8 状态 (cx,cy,w,h,vx,vy,vw,vh)  4 观测 (cx,cy,w,h)
        self.kf = cv2.KalmanFilter(8, 4)

        # ---- 状态转移矩阵 ----
        # 位置 = 位置 + 速度（使用 dt=1）
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1.0  # cx += vx
        self.kf.transitionMatrix[1, 5] = 1.0  # cy += vy
        self.kf.transitionMatrix[2, 6] = 1.0  # w  += vw
        self.kf.transitionMatrix[3, 7] = 1.0  # h  += vh

        # ---- 观测矩阵 ----
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0
        self.kf.measurementMatrix[3, 3] = 1.0

        # ---- 噪声 ----
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:, 4:] *= 0.01   # 速度噪声更小
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # ---- 初始状态 ----
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[0, 0] = bbox[0]
        self.kf.statePost[1, 0] = bbox[1]
        self.kf.statePost[2, 0] = bbox[2]
        self.kf.statePost[3, 0] = bbox[3]

        self.bbox = list(bbox)
        self.confidence = confidence
        self.cls = cls
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.features = self._extract_features(frame, bbox) if frame is not None else None

    def _extract_features(self, frame, bbox):
        if frame is None:
            return None
        try:
            cx, cy, w, h = [int(v) for v in bbox]
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(frame.shape[1] - 1, cx + w // 2)
            y2 = min(frame.shape[0] - 1, cy + h // 2)
            if x2 <= x1 + 3 or y2 <= y1 + 3:
                return None
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except:
            return None

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        s = self.kf.statePre
        return [float(s[0, 0]), float(s[1, 0]), float(s[2, 0]), float(s[3, 0])]

    def update(self, bbox, confidence, cls, frame=None):
        self.time_since_update = 0
        self.hits += 1
        self.confidence = confidence
        self.cls = cls
        self.bbox = list(bbox)

        meas = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float32)
        self.kf.correct(meas)

        if frame is not None:
            feat = self._extract_features(frame, bbox)
            if feat is not None:
                self.features = feat


def iou(box1, box2):
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    x1, y1, x2, y2 = cx1 - w1/2, cy1 - h1/2, cx1 + w1/2, cy1 + h1/2
    x3, y3, x4, y4 = cx2 - w2/2, cy2 - h2/2, cx2 + w2/2, cy2 + h2/2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1, area2 = w1 * h1, w2 * h2
    return inter / (area1 + area2 - inter + 1e-8)


def feature_distance(f1, f2):
    if f1 is None or f2 is None:
        return 0.5
    return float(cv2.compareHist(f1, f2, cv2.HISTCMP_BHATTACHARYYA))


class EnhancedTracker:
    def __init__(self, max_age=30, min_hits=5, iou_threshold=0.25, feature_weight=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_weight = feature_weight
        self.tracks = []
        self.frame_count = 0

    def update(self, detections, frame=None):
        self.frame_count += 1

        # ---- 预测所有轨迹 ----
        predictions = []
        for track in self.tracks:
            predictions.append(track.predict())

        n_dets = len(detections)
        n_tracks = len(self.tracks)

        # ---- 无检测情况 ----
        if n_dets == 0:
            for track in self.tracks:
                track.update(track.bbox, track.confidence, track.cls, None)
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._get_active_tracks()

        # ---- 计算代价矩阵 ----
        cost_matrix = np.full((n_dets, max(1, n_tracks)), 1e9, dtype=np.float32)

        for i, det in enumerate(detections):
            det_bbox = det['bbox']

            # 提取检测外观特征
            det_feat = None
            if frame is not None:
                try:
                    cx, cy, w, h = [int(v) for v in det_bbox]
                    x1 = max(0, cx - w // 2)
                    y1 = max(0, cy - h // 2)
                    x2 = min(frame.shape[1] - 1, cx + w // 2)
                    y2 = min(frame.shape[0] - 1, cy + h // 2)
                    if x2 > x1 + 3 and y2 > y1 + 3:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                            cv2.normalize(hist, hist)
                            det_feat = hist.flatten()
                except:
                    pass

            for j, track in enumerate(self.tracks):
                iou_val = iou(det_bbox, predictions[j])
                if iou_val < 0.02:
                    continue

                iou_cost = 1.0 - iou_val

                if det_feat is not None and track.features is not None:
                    feat_cost = feature_distance(det_feat, track.features)
                else:
                    feat_cost = 0.5

                cost_matrix[i, j] = (1 - self.feature_weight) * iou_cost + self.feature_weight * feat_cost

        # ---- 匈牙利匹配 ----
        matched_dets, matched_tracks = set(), set()
        if n_dets > 0 and n_tracks > 0:
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except:
                row_ind, col_ind = [], []

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e8:
                    self.tracks[c].update(
                        detections[r]['bbox'],
                        detections[r].get('confidence', 0.5),
                        detections[r].get('class', 0),
                        frame
                    )
                    matched_dets.add(r)
                    matched_tracks.add(c)

        # ---- 新轨迹 ----
        for i in range(n_dets):
            if i not in matched_dets:
                self.tracks.append(KalmanBoxTracker(
                    detections[i]['bbox'],
                    detections[i].get('confidence', 0.5),
                    detections[i].get('class', 0),
                    frame
                ))

        # ---- 未匹配轨迹用预测值更新 ----
        for j in range(n_tracks):
            if j not in matched_tracks:
                self.tracks[j].update(
                    predictions[j],
                    self.tracks[j].confidence,
                    self.tracks[j].cls,
                    None
                )

        # ---- 清理 ----
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self._get_active_tracks()

    def _get_active_tracks(self):
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                results.append({
                    'bbox': track.bbox,
                    'id': track.id,
                    'confidence': track.confidence,
                    'class': track.cls,
                    'class_name': 'object',
                    'hits': track.hits,
                    'age': track.age,
                })
        return results

    def reset(self):
        self.tracks.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
