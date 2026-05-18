import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EXP_NAMES = {
    "baseline": "基线 (YOLOv8n)",
    "optimized_a": "优化A (YOLOv8x)",
    "optimized_b": "优化B (+缓存)",
    "optimized_c": "优化C (+案例库)"
}

METRIC_NAMES = {
    "success_rate": "成功率",
    "avg_total_latency": "平均延迟(s)",
    "llm_success_rate": "LLM成功率",
    "fallback_rate": "降级率",
    "decision_consistency": "决策一致性",
    "MOTA": "MOTA",
    "IDF1": "IDF1"
}

COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E", "#C73E1D"]

def find_latest_archive(base_dir: str = "output"):
    patterns = ["experiment_batch_*", "experiment_*", "session_*"]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(base_dir, pat)))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def load_experiment_results(archive_dir: str) -> pd.DataFrame:
    records = []
    csv_file = os.path.join(archive_dir, "experiment_summary.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df.rename(columns={
            "实验组": "experiment",
            "模型": "model",
            "成功率": "success_rate",
            "延迟(s)": "avg_total_latency",
            "LLM成功率": "llm_success_rate",
            "降级率": "fallback_rate",
            "决策一致性": "decision_consistency",
            "MOTA": "MOTA",
            "IDF1": "IDF1"
        }, inplace=True)
        for col in ["success_rate", "llm_success_rate", "fallback_rate"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100.0
        df["display_name"] = df["experiment"].map(EXP_NAMES).fillna(df["experiment"])
        return df
    json_files = glob.glob(os.path.join(archive_dir, "**", "exp_*.json"), recursive=True)
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        exp_name = data.get("experiment") or os.path.basename(f).replace("exp_", "").replace(".json", "")
        config = data.get("config", {})
        metrics = data.get("metrics", {})
        record = {
            "experiment": exp_name,
            "model": config.get("model", ""),
            "cache_enabled": config.get("cache_enabled", False),
            "case_base_enabled": config.get("case_base_enabled", False),
        }
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    record[f"{k}_{sub_k}"] = sub_v
            else:
                record[k] = v
        records.append(record)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["display_name"] = df["experiment"].map(EXP_NAMES).fillna(df["experiment"])
    return df

def plot_radar(df: pd.DataFrame, metrics: list, title: str = "多维性能雷达图",
               save_path: str = "output/figures/radar.png"):
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < 3:
        print("指标数量不足，无法生成雷达图")
        return
    df_plot = df.copy()
    reverse_metrics = ["avg_total_latency", "fallback_rate"]
    for m in reverse_metrics:
        if m in df_plot.columns:
            max_val = df_plot[m].max()
            min_val = df_plot[m].min()
            if max_val > min_val:
                df_plot[m] = 1 - (df_plot[m] - min_val) / (max_val - min_val)
            else:
                df_plot[m] = 0.5
    for m in available_metrics:
        if m in reverse_metrics:
            continue
        min_val = df_plot[m].min()
        max_val = df_plot[m].max()
        if max_val > min_val:
            df_plot[m] = (df_plot[m] - min_val) / (max_val - min_val)
        else:
            df_plot[m] = 0.5

    categories = [METRIC_NAMES.get(m, m) for m in available_metrics]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(df_plot.iterrows()):
        values = [row[m] for m in available_metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row["display_name"], color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")

def generate_all_enhanced_plots(archive_dir: str = None):
    if archive_dir is None:
        archive_dir = find_latest_archive()
        if archive_dir is None:
            print("未找到任何归档文件夹")
            return
        print(f"自动选择最新归档: {archive_dir}")
    df = load_experiment_results(archive_dir)
    if df.empty:
        print("未加载到任何实验数据")
        return
    figures_dir = os.path.join(archive_dir, "figures_enhanced")
    os.makedirs(figures_dir, exist_ok=True)
    metrics = ["success_rate", "avg_total_latency", "llm_success_rate", 
               "fallback_rate", "decision_consistency"]
    plot_radar(df, metrics, "多维性能雷达图", os.path.join(figures_dir, "radar.png"))
    print(f"雷达图已生成至: {figures_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_all_enhanced_plots(sys.argv[1])
    else:
        generate_all_enhanced_plots()
