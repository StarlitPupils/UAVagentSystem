# agents/orchestrator_agent.py
"""编排智能体 — 负责任务分解、调度以及多机协同逻辑"""
import asyncio, time, uuid
from typing import Dict, List

class OrchestratorAgent:
    def __init__(self):
        self.name = "OrchestratorAgent"
        self.task_queue: List[Dict] = []
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: List[Dict] = []

    async def decompose_mission(self, command: str, context: dict = None) -> List[dict]:
        """将高层指令分解为原子子任务"""
        sub_tasks = []
        cmd_lower = command.lower()
        if "search" in cmd_lower or "搜索" in command:
            sub_tasks.append({"action": "search", "priority": 1})
        if "track" in cmd_lower or "跟踪" in command:
            sub_tasks.append({"action": "track", "priority": 1})
        if "report" in cmd_lower or "报告" in command:
            sub_tasks.append({"action": "report", "priority": 2})
        if not sub_tasks:
            sub_tasks.append({"action": command, "priority": 1})
        return sub_tasks

    async def schedule(self, tasks: List[dict]) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.active_tasks[task_id] = {"tasks": tasks, "started": time.time(), "status": "running"}
        return task_id

    def mark_complete(self, task_id: str, success: bool = True):
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "done" if success else "failed"
            self.completed_tasks.append(self.active_tasks.pop(task_id))