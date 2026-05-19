import asyncio
import random
from typing import Dict, List

class RobustnessTester:
    def __init__(self, integration_agent):
        self.integration = integration_agent
        self.fault_types = ['perception_fail', 'reasoning_timeout', 'action_error']
    async def inject_fault_and_test(self, command: str, fault_type: str = None) -> Dict:
        if fault_type is None: fault_type = random.choice(self.fault_types)
        original_perceive = self.integration.perception.perceive
        original_parse = self.integration.reasoning.parse_command
        original_execute = self.integration.action.execute
        try:
            if fault_type == 'perception_fail':
                self.integration.perception.perceive = lambda: {'num_objects': 0, 'detections': [], 'image': None}
            elif fault_type == 'reasoning_timeout':
                async def timeout_parse(*args, **kwargs):
                    await asyncio.sleep(100); return {}
                self.integration.reasoning.parse_command = timeout_parse
            elif fault_type == 'action_error':
                self.integration.action.execute = lambda plan: False
            success = await self.integration.execute_mission(command)
            return {'fault_type': fault_type, 'recovered': success, 'command': command}
        finally:
            self.integration.perception.perceive = original_perceive
            self.integration.reasoning.parse_command = original_parse
            self.integration.action.execute = original_execute
    async def run_robustness_suite(self, commands: List[str], repeats: int = 3) -> List[Dict]:
        results = []
        for cmd in commands:
            for _ in range(repeats):
                for fault in self.fault_types:
                    res = await self.inject_fault_and_test(cmd, fault)
                    results.append(res)
                    await asyncio.sleep(1)
        return results