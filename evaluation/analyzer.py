# evaluation/analyzer.py
"""指标分析器 - 自动查找最新会话的指标"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import glob
from typing import Dict
import pandas as pd
from config.settings import config

class MetricsAnalyzer:
    def __init__(self):
        # 自动定位最新会话目录
        out_dir = getattr(config, 'OUTPUT_DIR', None)
        if not out_dir:
            # 查找 output 下最新的 session_* 目录
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
            candidates = glob.glob(os.path.join(base, 'session_*'))
            if candidates:
                out_dir = max(candidates, key=os.path.getmtime)
            else:
                out_dir = os.path.join(base, 'default')

        self.metrics_dir = os.path.join(out_dir, "metrics")
        self.reports_dir = os.path.join(out_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_all_metrics(self) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.metrics_dir, "*.jsonl"))
        records = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
        return pd.DataFrame(records)

    def compute_summary(self, df: pd.DataFrame) -> Dict:
        total_tasks = len(df)
        if total_tasks == 0:
            return {"total_tasks": 0, "message": "无任务数据"}

        success_rate = df['success'].mean() if 'success' in df.columns else 0.0
        avg_latency = df['total_latency'].mean() if 'total_latency' in df.columns else 0.0

        llm_success_rate = 0.0
        if 'llm_called' in df.columns and 'llm_success' in df.columns:
            llm_called = df[df['llm_called']]
            if len(llm_called) > 0:
                llm_success_rate = llm_called['llm_success'].mean()

        fallback_rate = df['fallback_used'].mean() if 'fallback_used' in df.columns else 0.0
        avg_objects = df['perception_objects'].mean() if 'perception_objects' in df.columns else 0.0

        return {
            "total_tasks": total_tasks,
            "success_rate": round(success_rate, 3),
            "avg_total_latency": round(avg_latency, 2),
            "llm_success_rate": round(llm_success_rate, 3),
            "fallback_rate": round(fallback_rate, 3),
            "avg_detected_objects": round(avg_objects, 1),
        }

    def generate_report(self):
        df = self.load_all_metrics()
        if df.empty:
            print("没有找到指标数据，请先运行一些任务。")
            return {"total_tasks": 0, "message": "无数据"}

        summary = self.compute_summary(df)
        report_path = os.path.join(self.reports_dir, "performance_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"性能汇总报告已保存至: {report_path}")
        print("\n=== 性能摘要 ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary


if __name__ == "__main__":
    analyzer = MetricsAnalyzer()
    analyzer.generate_report()
