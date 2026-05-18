import json
import os
from typing import Dict, List, Optional
from collections import deque
from config.settings import config

class CaseBase:
    def __init__(self, max_cases: int = 50):
        self.max_cases = max_cases
        self.cases: deque = deque(maxlen=max_cases)
        # 使用配置中的 OUTPUT_DIR 确保位于当前会话目录
        self.case_file = os.path.join(config.OUTPUT_DIR, "case_base.json")
        self._load()
    def add_case(self, command: str, vision_summary: str, plan: Dict, success: bool):
        if not success:
            return
        case = {"command": command, "vision_summary": vision_summary, "plan": plan}
        self.cases.append(case)
        self._save()
    def find_similar(self, command: str, num_objects: int) -> Optional[Dict]:
        cmd_lower = command.lower()
        best_match = None
        best_score = 0
        for case in self.cases:
            score = 0
            if case["command"].lower() in cmd_lower or cmd_lower in case["command"].lower():
                score += 3
            if case["vision_summary"].count("目标") == num_objects:
                score += 1
            if score > best_score:
                best_score = score
                best_match = case["plan"]
        return best_match if best_score >= 2 else None
    def _save(self):
        os.makedirs(os.path.dirname(self.case_file), exist_ok=True)
        with open(self.case_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.cases), f, ensure_ascii=False, indent=2)
    def _load(self):
        if os.path.exists(self.case_file):
            with open(self.case_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cases = deque(data, maxlen=self.max_cases)
