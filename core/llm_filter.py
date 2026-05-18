# E:/UAVagent1.1/core/llm_filter.py
"""LLM 推理降噪器 v2 - 多模型确认优先策略
规则：
1. num_models >= 2 的目标直接保留（多模型交叉验证）
2. num_models == 1 但 confidence > 0.5 保留（高置信度单检）
3. 仅对 num_models == 1 且 confidence < 0.5 的目标进行 LLM 审查
4. LLM 失败时按 confidence=0.3 阈值兜底
"""
import json, re
from core.llm.llm_client import llm_client

class LLMDetectionFilter:
    def __init__(self):
        self.enabled = True
        self.low_conf_threshold = 0.5
        self.fallback_threshold = 0.3
        self.llm_batch_max = 8
        self.cache = {}

    def filter(self, detections: list[dict], frame_description: str = "") -> list[dict]:
        if not self.enabled or not detections:
            return detections

        keep = []
        candidates_for_llm = []

        for i, d in enumerate(detections):
            num_models = d.get('num_models', 1)
            conf = d.get('confidence', 0)

            # 规则1: 多模型确认 → 直接保留
            if num_models >= 2:
                keep.append(d)
                continue

            # 规则2: 高置信度单检 → 直接保留
            if conf >= self.low_conf_threshold:
                keep.append(d)
                continue

            # 规则3: 低置信度单检 → 待 LLM 审查
            candidates_for_llm.append((i, d))

        if not candidates_for_llm:
            return keep

        # LLM 审查候选
        decisions = self._llm_review(candidates_for_llm, frame_description)

        for (idx, det), decision in zip(candidates_for_llm, decisions):
            if decision:
                keep.append(det)
            # 否则丢弃

        removed = len(detections) - len(keep)
        if removed > 0:
            print(f"[LLM-Filter] 审查{len(candidates_for_llm)}个, 移除{removed}个 "
                  f"(保留{len(keep)}/{len(detections)})")

        return keep

    def _llm_review(self, candidates: list, frame_desc: str) -> list[bool]:
        """LLM 批量审查低置信度单检"""
        if not candidates:
            return []

        # 构建简洁描述
        items = []
        for idx, (i, d) in enumerate(candidates):
            bbox = d.get('bbox', [0, 0, 0, 0])
            items.append(
                f"#{idx}: cls={d.get('class',0)} conf={d.get('confidence',0):.2f} "
                f"pos=({bbox[0]:.0f},{bbox[1]:.0f}) size={bbox[2]:.0f}x{bbox[3]:.0f}"
            )

        prompt = (
            f"无人机航拍图{frame_desc}。以下是低置信度单模型检测，请判断每个是否合理。"
            f"标准：尺寸在15-300像素、宽高比0.2-5之间、位置不超出图像边界(640x640)。"
            f"回复JSON: [true, false, ...]（与输入顺序对应）\n"
            + "\n".join(items)
        )

        try:
            resp = llm_client.chat(
                [{"role": "user", "content": prompt}],
                system_prompt="仅输出JSON数组。"
            )
            content = resp.get("content", "[]")
            result = self._parse_bool_array(content, len(candidates))
            if result and len(result) == len(candidates):
                return result
        except Exception as e:
            print(f"[LLM-Filter] 审查失败: {e}")

        # 降级：按阈值过滤
        return [d.get('confidence', 0) >= self.fallback_threshold for _, d in candidates]

    def _parse_bool_array(self, text: str, expected_len: int) -> list[bool]:
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [bool(v) for v in arr[:expected_len]]
        except json.JSONDecodeError:
            pass
        # 尝试提取 []
        match = re.search(r'\[([^\]]*)\]', text)
        if match:
            inner = match.group(1)
            # 统计 true/false 数量
            trues = inner.lower().count('true')
            falses = inner.lower().count('false')
            if trues + falses >= expected_len:
                result = []
                for word in inner.split(','):
                    if 'true' in word.lower():
                        result.append(True)
                    elif 'false' in word.lower():
                        result.append(False)
                return result[:expected_len]
        return []
