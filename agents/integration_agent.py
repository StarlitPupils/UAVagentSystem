# agents/integration_agent.py（关键部分修正）
import asyncio
import traceback
import time
import uuid
import os
from agents.perception_agent import PerceptionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.action_agent import ActionAgent
from agents.reporting_agent import ReportingAgent
from agents.learning_agent import LearningAgent
from core.data_logger import DataLogger
from evaluation.metrics_collector import MetricsCollector, TaskMetrics
from core.agent_registry import registry

class IntegrationAgent:
    def __init__(self, perception=None, reasoning=None, action=None, reporting=None, learning=None, logger=None):
        if perception is None:
            perception_cls = registry.get_agent_class('PerceptionAgent')
            from core.vision_system import VisionSystem
            from core.uav_controller import UavController
            vision = VisionSystem()
            uav = UavController()
            perception = perception_cls(vision, uav)
        if reasoning is None:
            reasoning_cls = registry.get_agent_class('ReasoningAgent')
            reasoning = reasoning_cls()
        if action is None:
            action_cls = registry.get_agent_class('ActionAgent')
            from core.uav_controller import UavController
            uav = UavController()
            action = action_cls(uav, logger or DataLogger())
        if reporting is None:
            reporting_cls = registry.get_agent_class('ReportingAgent')
            reporting = reporting_cls(logger or DataLogger())
        if learning is None:
            learning_cls = registry.get_agent_class('LearningAgent')
            learning = learning_cls(logger or DataLogger())
        if logger is None:
            logger = DataLogger()
        self.perception = perception
        self.reasoning = reasoning
        self.action = action
        self.reporting = reporting
        self.learning = learning
        self.logger = logger
        self.metrics_collector = MetricsCollector()
        self.task_count = 0
        self.auto_evolve = True

    async def execute_mission(self, command: str):
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        print(f"[集成智能体] 开始执行任务: {command} (ID: {task_id})")

        metrics = TaskMetrics(
            task_id=task_id, command=command, start_time=start_time, end_time=0.0, success=False,
            perception_objects=0, reasoning_plan={}, action_type="", llm_called=False, llm_success=False,
            llm_latency=0.0, fallback_used=False, total_latency=0.0, collaboration_rounds=0,
            agent_idle_ratios={}, llm_prompt_tokens=0, llm_completion_tokens=0, llm_reasoning_length=0,
            llm_ttft=0.0, llm_error_type=""
        )
        try:
            # 感知
            print("[集成智能体] 调用感知智能体...")
            self.perception.receive_command(command)
            vis_task = asyncio.create_task(asyncio.to_thread(self.perception.perceive))
            vis = await vis_task
            metrics.perception_objects = vis.get('num_objects', 0)
            print(f"[集成智能体] 感知完成，检测到 {metrics.perception_objects} 个目标")

            # 推理
            print("[集成智能体] 调用推理智能体...")
            llm_start = time.time()
            metrics.llm_called = True
            plan = await self.reasoning.parse_command(command, vis)
            llm_latency = time.time() - llm_start
            metrics.llm_latency = llm_latency

            # 兼容两种推理智能体接口
            if hasattr(self.reasoning, 'llm_client') and hasattr(self.reasoning.llm_client, 'get_last_call_details'):
                details = self.reasoning.llm_client.get_last_call_details()
            elif hasattr(self.reasoning, 'llm_client') and hasattr(self.reasoning.llm_client, 'last_call_details'):
                details = self.reasoning.llm_client.last_call_details
            else:
                details = {}

            metrics.llm_success = details.get('success', True)
            metrics.fallback_used = plan.get('fallback', False) or plan.get('reasoning', '') == '本地降级推理'
            metrics.llm_prompt_tokens = details.get('prompt_tokens', 0)
            metrics.llm_completion_tokens = details.get('completion_tokens', 0)
            metrics.llm_reasoning_length = len(details.get('reasoning_content', ''))
            metrics.llm_ttft = details.get('ttft', 0.0)
            metrics.llm_error_type = details.get('error_type', '')
            metrics.reasoning_plan = plan
            metrics.action_type = plan.get('action_type', 'unknown')
            print(f"[集成智能体] 推理完成，计划: {plan}")

            # 行动
            print("[集成智能体] 调用行动智能体...")
            action_result = self.action.execute(plan)
            # ✅ 关键修复：显式转换为布尔值
            success = bool(action_result) if action_result is not None else True
            metrics.success = success
            print(f"[集成智能体] 行动完成，结果: {success}")

            # 学习
            print("[集成智能体] 调用学习智能体...")
            self.learning.record(vis, plan, success)
            if success:
                try:
                    self.reasoning.record_successful_plan(command, vis, plan, success)
                except AttributeError:
                    pass  # 部分推理智能体无此方法

            self.logger.log_event("mission_done", {"cmd": command, "success": success, "task_id": task_id})
            metrics.collaboration_rounds = 3
        except Exception as e:
            print(f"[集成智能体] 任务执行异常: {e}")
            traceback.print_exc()
            metrics.success = False
        finally:
            metrics.end_time = time.time()
            metrics.total_latency = metrics.end_time - metrics.start_time
            self.metrics_collector.record_task(metrics)
            print(f"[集成智能体] 任务结束: {metrics.success} (耗时 {metrics.total_latency:.2f}s)")

        self.task_count += 1
        if self.auto_evolve and self.task_count % 5 == 0:
            await self._trigger_reflection()
        return metrics.success

    async def _trigger_reflection(self):
        print("[Integration] Triggering reflection...")
        from agents.reflection_agent import ReflectionAgent
        from agents.meta_agent import MetaAgent
        ref = ReflectionAgent(self.logger)
        meta = MetaAgent()
        suggestions = await ref.analyze_logs()
        if not suggestions or "失败" in suggestions:
            print("[Integration] Reflection failed, skipping patch generation.")
            return
        patch = await meta.generate_patch(suggestions)
        target = os.path.join(os.path.dirname(__file__), 'perception_agent.py')
        if await meta.apply_and_test(patch, target):
            print("[Integration] System evolved successfully!")
        else:
            print("[Integration] Evolution rolled back.")
