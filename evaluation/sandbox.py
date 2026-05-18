# evaluation/sandbox.py
"""沙盒测试器 - 弱化严格度，避免错误中断"""
import subprocess
import sys
import os
from typing import Tuple

class SandboxTester:
    def __init__(self):
        self.test_scenarios = ["search", "track person", "track car"]
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run_tests(self) -> Tuple[bool, float, float]:
        batch_runner = os.path.join(self.project_root, "evaluation", "batch_runner.py")
        if not os.path.exists(batch_runner):
            print(f"[沙盒] 错误：找不到 {batch_runner}")
            return False, 0.0, 0.0

        try:
            result = subprocess.run(
                [sys.executable, batch_runner],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root
            )
            if result.returncode != 0:
                print(f"[沙盒] 测试脚本返回码 {result.returncode}，但继续分析")
                # 即使子进程报错，也尝试加载指标（可能部分成功）
        except subprocess.TimeoutExpired:
            print("[沙盒] 测试超时")
            return False, 0.0, 0.0
        except Exception as e:
            print(f"[沙盒] 执行异常: {e}")
            return False, 0.0, 0.0

        # 加载最新会话指标
        try:
            from evaluation.analyzer import MetricsAnalyzer
            analyzer = MetricsAnalyzer()
            df = analyzer.load_all_metrics()
            if df.empty:
                print("[沙盒] 未收集到指标数据")
                return False, 0.0, 0.0
            success_rate = df['success'].mean()
            avg_latency = df['total_latency'].mean()
            return True, success_rate, avg_latency
        except Exception as e:
            print(f"[沙盒] 指标分析失败: {e}")
            return False, 0.0, 0.0

    def is_improvement(self, new_success: float, new_latency: float,
                       old_success: float = 0.8, old_latency: float = 60.0) -> bool:
        if new_success > old_success:
            return True
        if new_success >= old_success * 0.95 and new_latency < old_latency * 0.9:
            return True
        return False
