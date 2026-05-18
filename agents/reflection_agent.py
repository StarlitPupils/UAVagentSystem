# agents/reflection_agent.py
"""反思智能体 - 分析日志并生成优化建议（仅 deepseek-v4-flash）"""
import os
from core.llm.llm_client import llm_client
from core.data_logger import DataLogger

class ReflectionAgent:
    def __init__(self, logger: DataLogger):
        self.logger = logger
        self.llm_client = llm_client
        self.reflection_timeout = int(os.getenv("LLM_REFLECTION_TIMEOUT", "120"))

    async def analyze_logs(self) -> str:
        events = self.logger.get_recent_events(50)
        prompt = f"分析以下无人机系统日志，找出可优化的参数或逻辑，并给出具体修改建议。建议应包含目标文件、修改内容和预期效果。\n日志: {events}\n"
        try:
            return await self.llm_client.generate(prompt, timeout_override=self.reflection_timeout)
        except Exception as e:
            return f"反思失败: {e}"
