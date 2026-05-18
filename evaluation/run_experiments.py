import json
import os
import subprocess
import sys
import time
import pandas as pd
from typing import Dict, List, Any

# 实验配置组合
EXPERIMENTS = [
    {"name": "Transformer_YOLOv8n_CacheOff", "tracker": "transformer", "model": "yolov8n.pt", "cache": False, "casebase": False},
    {"name": "Transformer_YOLOv8n_CacheOn", "tracker": "transformer", "model": "yolov8n.pt", "cache": True, "casebase": True},
    {"name": "Transformer_YOLOv8x_CacheOff", "tracker": "transformer", "model": "yolov8x.pt", "cache": False, "casebase": False},
    {"name": "Transformer_YOLOv8x_CacheOn", "tracker": "transformer", "model": "yolov8x.pt", "cache": True, "casebase": True},
    # 若BoT-SORT可用，可添加对应实验
]

def update_config(tracker, model_path):
    # 修改 config/settings.py 中的 TRACKER_TYPE 和 YOLO_MODEL_PATH
    settings_path = "config/settings.py"
    with open(settings_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'TRACKER_TYPE: str = ".*"', f'TRACKER_TYPE: str = "{tracker}"', content)
    content = re.sub(r'YOLO_MODEL_PATH: str = .*', f'YOLO_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "{model}")', content)
    with open(settings_path, 'w', encoding='utf-8') as f:
        f.write(content)

def set_cache_flags(cache_on, casebase_on):
    # 修改 core/llm_client.py 中 use_cache 默认值，及 reasoning_agent 中案例库开关（此处简化，通过环境变量或代码开关）
    # 实际可通过全局变量控制，为简化，我们直接修改 LLMClient 的 use_cache 参数
    # 由于我们设计时 generate 方法有 use_cache 参数，但调用时未显式传递，我们可修改默认值
    llm_client_path = "core/llm_client.py"
    with open(llm_client_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 替换默认 use_cache 值
    content = re.sub(r'use_cache: bool = True', f'use_cache: bool = {str(cache_on)}', content)
    with open(llm_client_path, 'w', encoding='utf-8') as f:
        f.write(content)
    # 案例库开关可在 reasoning_agent 的 parse_command 中控制是否使用 case_base

def run_experiment(exp: Dict):
    print(f"\n{'='*50}")
    print(f"运行实验: {exp['name']}")
    print(f"{'='*50}")
    # 更新配置
    update_config(exp['tracker'], exp['model'])
    set_cache_flags(exp['cache'], exp['casebase'])
    # 运行批量测试
    subprocess.run([sys.executable, "evaluation/batch_runner.py"], check=True)
    # 收集指标
    from evaluation.analyzer import MetricsAnalyzer
    analyzer = MetricsAnalyzer()
    df = analyzer.load_all_metrics()
    summary = analyzer.compute_summary(df) if not df.empty else {}
    summary['experiment'] = exp['name']
    summary['tracker'] = exp['tracker']
    summary['model'] = exp['model']
    summary['cache'] = exp['cache']
    summary['casebase'] = exp['casebase']
    # 保存到汇总文件
    results_file = "output/experiments_summary.json"
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(summary)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"实验 {exp['name']} 完成，成功率: {summary.get('success_rate', 'N/A')}")

if __name__ == "__main__":
    import re
    for exp in EXPERIMENTS:
        run_experiment(exp)
    print("\n所有实验完成，汇总结果保存在 output/experiments_summary.json")
