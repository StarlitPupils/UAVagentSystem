# agents/reasoning_agent.py
"""推理智能体 - 兼容1.0接口 + 2.0向量记忆 + 单模型deepseek-v4-flash"""
import json
import asyncio
from config.settings import config

class ReasoningAgent:
    def __init__(self):
        self.name = "ReasoningAgent"
        self._llm = None
        self._memory = None
        # 1.0 兼容
        self.last_call_details = {"success": False, "model": "", "latency_ms": 0}

    @property
    def llm(self):
        if self._llm is None:
            from core.llm.llm_client import llm_client
            self._llm = llm_client
        return self._llm

    @property
    def llm_client(self):
        """1.0 兼容别名"""
        return self.llm

    @property
    def memory(self):
        if self._memory is None:
            from core.memory.memory_manager import memory_manager
            self._memory = memory_manager
        return self._memory

    # ========== 1.0 兼容接口 ==========
    async def parse_command(self, command: str, visual_state: dict = None) -> dict:
        return await self.reason(command, visual_state)

    # ========== 2.0 核心接口 ==========
    async def reason(self, command: str, visual_state: dict = None) -> dict:
        vis = visual_state or {}

        # 安全获取字段（兼容不同来源的键名）
        count = vis.get("num_objects", vis.get("count", 0))
        targets = vis.get("detections", vis.get("targets", []))

        # 构建目标信息文本
        targets_info = ""
        if targets:
            # 简化每个目标的信息，便于 LLM 理解
            simplified = []
            for t in targets:
                bbox = t.get("bbox", [])
                class_name = t.get("class_name", t.get("class", "object"))
                conf = t.get("confidence", 0)
                tid = t.get("id", "?")
                simplified.append({
                    "id": tid,
                    "class": class_name,
                    "conf": round(conf, 2),
                    "bbox_center": [round(b, 1) for b in bbox[:2]] if bbox else []
                })
            targets_info = f"目标列表: {json.dumps(simplified, ensure_ascii=False)[:800]}\n"

        messages = [{
            "role": "user",
            "content": f"命令: {command}\n检测到 {count} 个目标\n{targets_info}"
        }]

        # 检索记忆上下文（失败不影响主流程）
        try:
            ctx = self.memory.get_context_for_llm(command)
            if ctx:
                messages.insert(0, {"role": "system", "content": f"相关历史经验:\n{ctx}"})
        except Exception:
            pass

        system_prompt = (
            "你是无人机目标检测与跟踪系统的推理智能体。\n"
            "根据用户命令和当前检测到的目标列表，制定行动计划。\n"
            "输出严格JSON（不要markdown代码块）：\n"
            '{"action_type":"search|track|report", "target_description":"...", '
            '"target_id":null, "confidence":0.5, "reasoning":"..."}'
        )

        try:
            response = self.llm.chat(messages, system_prompt=system_prompt)
            plan = self.llm.extract_json(response)

            # 更新调用详情
            self.last_call_details = {
                "success": not response.get("fallback", False),
                "model": response.get("model", config.LLM_MODEL),
                "latency_ms": response.get("latency_ms", 0),
                "fallback": response.get("fallback", False),
            }

            if "parse_error" in plan:
                plan = self.get_fallback_plan(command)
                self.last_call_details["success"] = False

            # 存储记忆
            try:
                self.memory.remember(
                    f"命令:{command} -> {plan.get('action_type','?')}:{plan.get('target_description','?')}",
                    memory_type="reasoning"
                )
            except Exception:
                pass

            return plan

        except Exception as e:
            print(f"[ReasoningAgent] LLM调用失败: {e}")
            self.last_call_details = {"success": False, "model": "", "latency_ms": 0, "error": str(e)}
            return self.get_fallback_plan(command)

    def reason_sync(self, command: str, visual_state: dict = None) -> dict:
        """同步推理"""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.reason(command, visual_state))
                )
                return future.result(timeout=60)
        except RuntimeError:
            return asyncio.run(self.reason(command, visual_state))

    def get_fallback_plan(self, command: str) -> dict:
        """本地规则降级"""
        cmd_lower = command.lower()

        if 'search' in cmd_lower or '搜索' in command:
            action_type = "search"
        elif 'report' in cmd_lower or '报告' in command:
            action_type = "report"
        else:
            action_type = "track"

        target_map = [
            (['car', 'vehicle', '车辆', '汽车', '轿车'], '车辆'),
            (['person', 'pedestrian', '行人', '人'], '行人'),
            (['truck', '卡车', '货车'], '卡车'),
            (['bus', '公交', '巴士'], '公交车'),
            (['bicycle', 'bike', '自行车'], '自行车'),
            (['uav', 'drone', '无人机'], '无人机'),
        ]
        target_desc = "目标"
        for keywords, desc in target_map:
            if any(kw in cmd_lower or kw in command for kw in keywords):
                target_desc = desc
                break

        return {
            "action_type": action_type,
            "target_description": target_desc,
            "target_id": None,
            "confidence": 0.5,
            "reasoning": f"本地规则降级 - 关键词匹配>{target_desc}<",
            "fallback": True,
        }
