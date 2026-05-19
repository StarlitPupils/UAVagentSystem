# core/detection/filter_consensus.py (v1.2.2 - 最终稳定版)
"""自适应共识过滤器 v5 - 异常模型熔断 + 密度感知"""
import numpy as np


def filter_by_consensus(detections: list[dict],
                         min_models: int = 2,
                         min_conf_single: float = 0.25) -> list[dict]:
    """
    共识过滤 v5:
    - 3+模型确认 → 无条件保留
    - 2模型确认 + conf>0.3 → 保留
    - 单模型高置信度 → 保留
    - 小目标保护
    """
    if not detections:
        return detections

    n = len(detections)

    # 密度自适应
    if n > 20:
        density_factor = 0.7
    elif n > 12:
        density_factor = 0.85
    elif n < 6:
        density_factor = 1.4
    else:
        density_factor = 1.0

    kept = []
    dropped = 0
    stats = {"multi3": 0, "multi2": 0, "high_conf": 0, "small_obj": 0, "dropped": 0}

    for d in detections:
        num_models = d.get('num_models', 1)
        conf = d.get('confidence', 0)
        bbox = d.get('bbox', [0, 0, 0, 0])
        area = bbox[2] * bbox[3]

        # 规则1: 3+模型确认 → 无条件保留
        if num_models >= 3:
            kept.append(d)
            stats["multi3"] += 1
            continue

        # 规则2: 2模型确认 + 合理置信度
        if num_models >= 2 and conf >= 0.25:
            kept.append(d)
            stats["multi2"] += 1
            continue

        # 规则3: 单模型 + 高置信度（自适应阈值）
        eff_threshold = max(0.35, min_conf_single * density_factor)
        if conf >= eff_threshold:
            kept.append(d)
            stats["high_conf"] += 1
            continue

        # 规则4: 小目标保护（尺寸合理 + 中等置信度）
        if 80 < area < 4000 and conf > 0.25:
            kept.append(d)
            stats["small_obj"] += 1
            continue

        dropped += 1
        stats["dropped"] += 1

    if dropped:
        print(f"  [Filter] keep={len(kept)} drop={dropped}/{n} "
              f"(3+: {stats['multi3']}, 2: {stats['multi2']}, "
              f"high: {stats['high_conf']}, small: {stats['small_obj']})")

    return kept


def detect_anomalous_model(det_counts: list, model_names: list) -> list:
    """
    异常模型检测 - 用于熔断
    返回: 应该被熔断（完全排除）的模型索引列表
    """
    if not det_counts or len(det_counts) < 3:
        return []

    median = np.median(det_counts)
    if median == 0:
        return []

    anomalous = []
    for i, count in enumerate(det_counts):
        # 规则: 检测数超过中位数5倍且>10个 → 熔断
        if count > 10 and count > median * 5:
            print(f"  [Meltdown] {model_names[i]}: {count} dets vs median {median:.0f} → 熔断")
            anomalous.append(i)

    return anomalous
