# core/data_logger.py
"""数据日志记录器 - 完整兼容所有 Agent"""
import os
import json
import time
from config.settings import config

class DataLogger:
    def __init__(self):
        log_dir = getattr(config, 'LOG_DIR', None)
        if log_dir is None:
            log_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'output'), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"run_{int(time.time())}.log")
        self.metrics_file = os.path.join(log_dir, f"metrics_{int(time.time())}.json")
        self.entries = []
        self.metrics = {}
        print(f"[DataLogger] 日志文件: {self.log_file}")

    def log(self, event_type: str, data: dict = None):
        """记录事件"""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data or {},
        }
        self.entries.append(entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def log_event(self, event_type: str, data: dict = None):
        """log_event 别名 - 兼容 action_agent / integration_agent"""
        self.log(event_type, data)

    def log_metric(self, key: str, value):
        self.metrics[key] = value

    def save_metrics(self):
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": self.metrics,
                "entry_count": len(self.entries),
                "timestamp": time.time(),
            }, f, indent=2, ensure_ascii=False)
        print(f"[DataLogger] 指标已保存: {self.metrics_file}")

    def get_recent_entries(self, n: int = 10) -> list:
        return self.entries[-n:]

    def get_recent_events(self, n: int = 50) -> list:
        """get_recent_events 别名 - 兼容 reflection_agent"""
        return self.get_recent_entries(n)

    def get_metrics(self) -> dict:
        return self.metrics.copy()
