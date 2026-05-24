# core/llm/llm_filter.py (UAVagent 1.4)
"""LLM 推理降噪器 — 多模型确认优先策略 (v2)
规则：
1. num_models >= 2 的直接保留
2. num_models == 1 且 confidence > 0.5 保留
3. 低置信度单检用 LLM 审查
4. LLM 失败时按 confidence=0.3 兜底
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
        candidates = []
        for i, d in enumerate(detections):
            num_models = d.get('num_models', 1)
            conf = d.get('confidence', 0)
            if num_models >= 2:
                keep.append(d)
            elif conf >= self.low_conf_threshold:
                keep.append(d)
            else:
                candidates.append((i, d))

        if not candidates:
            return keep

        decisions = self._llm_review(candidates, frame_description)
        for (_, det), decision in zip(candidates, decisions):
            if decision:
                keep.append(det)

        removed = len(detections) - len(keep)
        if removed > 0:
            print(f"[LLM-Filter] kept {len(keep)}/{len(detections)}, removed {removed}")
        return keep

    def _llm_review(self, candidates, frame_desc):
        items = []
        for idx, (_, d) in enumerate(candidates):
            bbox = d.get('bbox', [0, 0, 0, 0])
            items.append(
                f"#{idx}: cls={d.get('class',0)} conf={d.get('confidence',0):.2f} "
                f"pos=({bbox[0]:.0f},{bbox[1]:.0f}) size={bbox[2]:.0f}x{bbox[3]:.0f}"
            )
        prompt = (
            f"无人机航拍{frame_desc}。低置信度单模型检测，判断是否合理（JSON [true,false,...]）\n"
            + "\n".join(items)
        )
        try:
            resp = llm_client.chat([{"role": "user", "content": prompt}],
                                   system_prompt="仅输出JSON数组。")
            content = resp.get("content", "[]")
            result = self._parse_bool_array(content, len(candidates))
            if result and len(result) == len(candidates):
                return result
        except Exception as e:
            print(f"[LLM-Filter] 审查失败: {e}")
        return [d.get('confidence', 0) >= self.fallback_threshold for _, d in candidates]

    def _parse_bool_array(self, text: str, expected_len: int) -> list:
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [bool(v) for v in arr[:expected_len]]
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[([^\]]*)\]', text)
        if match:
            inner = match.group(1)
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