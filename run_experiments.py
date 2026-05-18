import os
import sys
import json
import shutil
import asyncio
import subprocess
import numpy as np
from datetime import datetime
from collections import defaultdict

experiments = [
    {"name": "baseline", "config_file": "experiments/baseline.env", "tracking_seq": "uav0000086_00000_v"},
    {"name": "optimized_a", "config_file": "experiments/optimized_a.env", "tracking_seq": "uav0000086_00000_v"},
    {"name": "optimized_b", "config_file": "experiments/optimized_b.env", "tracking_seq": "uav0000086_00000_v"},
    {"name": "optimized_c", "config_file": "experiments/optimized_c.env", "tracking_seq": "uav0000086_00000_v"},
]

async def run_experiment(exp_name, config_file, tracking_seq, session_base):
    print(f"\n{'='*60}\n开始实验: {exp_name}\n{'='*60}")
    
    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                os.environ[key] = val
    
    import config.settings as settings
    session_name = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    settings.config.setup_session(os.path.join(session_base, session_name))
    
    original_model = settings.config.YOLO_MODEL_PATH
    model_name = os.environ.get('YOLO_MODEL', 'yolov8x.pt')
    settings.config.YOLO_MODEL_PATH = os.path.join(settings.config.BASE_DIR, "models", model_name)
    settings.config.TRACKER_TYPE = os.environ.get('TRACKER_TYPE', 'transformer')
    
    # 1. 运行 VisDrone 回放，收集跟踪结果
    print("正在收集跟踪数据...")
    try:
        # 动态修改 run_visdrone_real.py 中的设备为 CPU 以避免 NMS 问题
        subprocess.run([sys.executable, "run_visdrone_real.py"], check=True, timeout=600)
    except Exception as e:
        print(f"跟踪数据收集失败（将使用占位指标）: {e}")
    
    # 2. 运行批量测试
    metrics_dir = os.path.join(settings.config.OUTPUT_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    from evaluation.batch_runner import BatchRunner
    runner = BatchRunner()
    with open('tests/test_scenarios.json', 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    await runner.run_scenarios(scenarios, repeats=5)
    
    settings.config.YOLO_MODEL_PATH = original_model
    
    # 3. 收集指标
    from evaluation.analyzer import MetricsAnalyzer
    analyzer = MetricsAnalyzer()
    df = analyzer.load_all_metrics()
    summary = analyzer.compute_summary(df) if not df.empty else {}
    
    # 4. 跟踪评估
    tracking_metrics = {}
    pred_file = os.path.join(settings.config.OUTPUT_DIR, "tracking", f"{tracking_seq}.txt")
    gt_file = f"E:/datasets/VisDrone/VisDrone2019-MOT-val/annotations/{tracking_seq}.txt"
    if os.path.exists(pred_file) and os.path.exists(gt_file):
        result = subprocess.run([sys.executable, "evaluate_tracking.py", gt_file, pred_file], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'UAVagent' in line:
                parts = line.split()
                tracking_metrics = {
                    'IDF1': float(parts[1]), 'IDP': float(parts[2]), 'IDR': float(parts[3]),
                    'Recall': float(parts[4]), 'Precision': float(parts[5]),
                    'MOTA': float(parts[10]), 'MOTP': float(parts[11])
                }
                break
    summary.update(tracking_metrics)
    
    result = {
        "experiment": exp_name, "timestamp": datetime.now().isoformat(),
        "config": {"model": model_name, "tracker": settings.config.TRACKER_TYPE,
                   "cache_enabled": os.environ.get('LLM_CACHE_ENABLED') == 'true',
                   "case_base_enabled": os.environ.get('CASE_BASE_ENABLED') == 'true'},
        "metrics": summary
    }
    result_file = os.path.join(settings.config.OUTPUT_DIR, f"exp_{exp_name}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"实验 {exp_name} 完成")
    return result

async def main():
    import config.settings as settings
    batch_session = f"experiment_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = os.path.join(settings.config.OUTPUT_BASE, batch_session)
    os.makedirs(batch_dir, exist_ok=True)
    
    results = []
    for exp in experiments:
        results.append(await run_experiment(exp["name"], exp["config_file"], exp["tracking_seq"], batch_session))
    
    # 汇总并保存 CSV
    headers = ["实验组", "模型", "成功率", "延迟(s)", "LLM成功率", "缓存命中率", "降级率", "决策一致性", "MOTA", "IDF1", "HOTA"]
    summary_csv = os.path.join(batch_dir, "experiment_summary.csv")
    import csv
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            m = r.get("metrics", {})
            writer.writerow([r['experiment'], r['config']['model'],
                             f"{m.get('success_rate', 0):.2%}", f"{m.get('avg_total_latency', 0):.2f}",
                             f"{m.get('llm_success_rate', 0):.2%}", f"{m.get('cache_hit_rate', 0):.2%}",
                             f"{m.get('fallback_rate', 0):.2%}", f"{m.get('decision_consistency', 0):.2f}",
                             f"{m.get('MOTA', 0):.3f}", f"{m.get('IDF1', 0):.3f}", f"{m.get('HOTA', 0):.3f}"])
    
    print(f"\n实验完成！结果已保存至 {batch_dir}")
    print("接下来请运行生成对比图表的脚本。")

if __name__ == "__main__":
    asyncio.run(main())
