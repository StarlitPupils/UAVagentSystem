# evaluation/metrics_collector.py
"""指标收集器 - 自动确保目录存在，保存到会话目录"""
import time
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from config.settings import config

@dataclass
class TaskMetrics:
    task_id: str
    command: str
    start_time: float
    end_time: float
    success: bool
    perception_objects: int
    reasoning_plan: Dict
    action_type: str
    llm_called: bool
    llm_success: bool
    llm_latency: float
    fallback_used: bool
    total_latency: float
    collaboration_rounds: int = 0
    agent_idle_ratios: Dict[str, float] = field(default_factory=dict)
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_reasoning_length: int = 0
    llm_ttft: float = 0.0
    llm_error_type: str = ""

class MetricsCollector:
    def __init__(self):
        # 确保会话存在
        out_dir = getattr(config, 'OUTPUT_DIR', None)
        if not out_dir:
            config.setup_session()
            out_dir = config.OUTPUT_DIR

        self.metrics_dir = os.path.join(out_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.current_metrics_file = os.path.join(
            self.metrics_dir,
            f"session_{int(time.time())}.jsonl"
        )

    def record_task(self, metrics: TaskMetrics):
        with open(self.current_metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(metrics), ensure_ascii=False) + '\n')

    def get_session_file(self) -> str:
        return self.current_metrics_file
