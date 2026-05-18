# E:/UAVagent1.1/core/detection/filter_consensus.py
"""自适应共识过滤器 v3"""
import numpy as np

def adaptive_consensus_filter(detections: list[dict],
                               min_models: int = 2,
                               base_conf: float = 0.25,
                               adaptive_factor: float = 1.5) -> list[dict]:
    """
    自适应阈值：根据当前帧检测密度动态调整
    - 密集场景（>15 dets）：收紧阈值，降低 FP
    - 稀疏场景（<8 dets）：放宽阈值，保留更多真实检测
    """
    if not detections:
        return detections

    n = len(detections)
    
    # 密度因子：稀疏场景放宽，密集场景收紧
    if n > 20:
        density_factor = 0.8   # 密集 → 更严格
    elif n < 10:
        density_factor = 1.2   # 稀疏 → 更宽松
    else:
        density_factor = 1.0

    # 置信度阈值随密度调整
    effective_conf = base_conf * density_factor

    kept = []
    dropped = 0
    for d in detections:
        models = d.get('num_models', 1)
        conf = d.get('confidence', 0)

        # 多模型确认 → 无条件保留
        if models >= min_models:
            kept.append(d)
            continue

        # 单模型 + 高于自适应阈值 → 保留
        if conf >= effective_conf:
            kept.append(d)
            continue

        # 单模型 + 极低置信度但尺寸合理（可能是小目标）→ 保留
        bbox = d.get('bbox', [0, 0, 0, 0])
        area = bbox[2] * bbox[3]
        if 100 < area < 5000 and conf > effective_conf * 0.7:
            kept.append(d)
            continue

        dropped += 1

    if dropped:
        print(f"  [Filter] -{dropped} dets (eff_conf={effective_conf:.2f}, n={n})")

    return kept
