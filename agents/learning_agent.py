# agents/learning_agent.py
"""学习智能体 - 自动创建训练数据目录"""
import json
import time
import os
from core.data_logger import DataLogger
from config.settings import config

class LearningAgent:
    def __init__(self, logger: DataLogger):
        self.logger = logger
        self.buffer = []

        # 安全获取 TRAINING_DATA_DIR，确保目录存在
        training_dir = getattr(config, 'TRAINING_DATA_DIR', None)
        if training_dir is None:
            training_dir = os.path.join(
                getattr(config, 'OUTPUT_DIR', 'output'), 'training_data'
            )
        os.makedirs(training_dir, exist_ok=True)
        self.training_dir = training_dir

    def record(self, vision_data: dict, plan: dict, success: bool):
        sample = {
            "time": time.time(),
            "plan": plan,
            "success": success,
            "num_objects": vision_data.get("num_objects", 0),
        }
        self.buffer.append(sample)
        if len(self.buffer) >= 10:
            self.flush()

    def flush(self):
        path = os.path.join(self.training_dir, f"batch_{int(time.time())}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.buffer, f, ensure_ascii=False)
        self.buffer.clear()
