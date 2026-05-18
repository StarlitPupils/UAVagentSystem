# evaluation/batch_runner.py
"""批量测试运行器 - 自动初始化会话"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import json
from config.settings import config
from agents.perception_agent import PerceptionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.action_agent import ActionAgent
from agents.integration_agent import IntegrationAgent
from agents.reporting_agent import ReportingAgent
from agents.learning_agent import LearningAgent
from core.vision_system import VisionSystem
from core.uav_controller import UavController
from core.data_logger import DataLogger

class BatchRunner:
    def __init__(self):
        # 确保会话已设置（这样所有组件使用同一个 OUTPUT_DIR）
        if not config.OUTPUT_DIR:
            config.setup_session()

        self.vision = VisionSystem()
        self.uav = UavController()
        self.logger = DataLogger()
        self.perception = PerceptionAgent(self.vision, self.uav)
        self.reasoning = ReasoningAgent()
        self.action = ActionAgent(self.uav, self.logger)
        self.reporting = ReportingAgent(self.logger)
        self.learning = LearningAgent(self.logger)
        self.integration = IntegrationAgent(
            self.perception, self.reasoning, self.action,
            self.reporting, self.learning, self.logger
        )

    async def run_scenarios(self, scenarios, repeats=3):
        for scenario in scenarios:
            for i in range(repeats):
                print(f"\n{'='*50}")
                print(f"执行场景: {scenario['name']} (第 {i+1}/{repeats} 次)")
                print(f"命令: {scenario['command']}")
                print(f"{'='*50}")
                await self.integration.execute_mission(scenario['command'])
                await asyncio.sleep(1)


async def main():
    # 首先初始化会话
    config.setup_session()
    print(f"[BatchRunner] 会话目录: {config.OUTPUT_DIR}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    scenarios_path = os.path.join(project_root, 'tests', 'test_scenarios.json')

    with open(scenarios_path, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)

    runner = BatchRunner()
    await runner.run_scenarios(scenarios, repeats=5)
    print("\n批量测试完成！指标已保存至", config.OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
