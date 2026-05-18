import os
import importlib.util
from typing import Dict, Type, Any

class AgentRegistry:
    _instance = None
    _agents: Dict[str, Type] = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._discover_builtin_agents()
            cls._instance._discover_custom_agents()
        return cls._instance
    def _discover_builtin_agents(self):
        from agents.perception_agent import PerceptionAgent
        from agents.reasoning_agent import ReasoningAgent
        from agents.action_agent import ActionAgent
        from agents.reporting_agent import ReportingAgent
        from agents.learning_agent import LearningAgent
        from agents.reflection_agent import ReflectionAgent
        from agents.meta_agent import MetaAgent
        from agents.training_agent import TrainingAgent
        self._agents['PerceptionAgent'] = PerceptionAgent
        self._agents['ReasoningAgent'] = ReasoningAgent
        self._agents['ActionAgent'] = ActionAgent
        self._agents['ReportingAgent'] = ReportingAgent
        self._agents['LearningAgent'] = LearningAgent
        self._agents['ReflectionAgent'] = ReflectionAgent
        self._agents['MetaAgent'] = MetaAgent
        self._agents['TrainingAgent'] = TrainingAgent
    def _discover_custom_agents(self):
        custom_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'custom_agents')
        if not os.path.isdir(custom_dir):
            return
        for filename in os.listdir(custom_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                filepath = os.path.join(custom_dir, filename)
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for attr_name in dir(module):
                    if attr_name.endswith('Agent') and not attr_name.startswith('_'):
                        agent_cls = getattr(module, attr_name)
                        self._agents[attr_name] = agent_cls
                        print(f"[Registry] 已加载自定义智能体: {attr_name}")
    def get_agent_class(self, name: str):
        return self._agents.get(name)
    def list_agents(self):
        return list(self._agents.keys())
registry = AgentRegistry()