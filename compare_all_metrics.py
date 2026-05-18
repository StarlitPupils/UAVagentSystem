import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results(exp_dir):
    results = {}
    for exp in ['baseline', 'optimized_a', 'optimized_b', 'optimized_c']:
        # 寻找最新的子目录
        subdirs = [d for d in os.listdir(exp_dir) if d.startswith(exp)]
        if not subdirs:
            continue
        subdir = sorted(subdirs)[-1]
        json_path = os.path.join(exp_dir, subdir, f'exp_{exp}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                results[exp] = json.load(f)['metrics']
    return results

def plot_comparison(results, output_dir):
    labels = ['Baseline', 'Opt-A', 'Opt-B', 'Opt-C']
    exp_names = ['baseline', 'optimized_a', 'optimized_b', 'optimized_c']
    
    metrics_config = {
        'success_rate': ('成功率', 'success_rate'),
        'avg_total_latency': ('平均延迟 (秒)', 'latency'),
        'llm_success_rate': ('LLM成功率', 'llm_success'),
        'cache_hit_rate': ('缓存命中率', 'cache_hit'),
        'decision_consistency': ('决策一致性', 'decision_consistency'),
        'MOTA': ('MOTA', 'mota'),
        'IDF1': ('IDF1', 'idf1'),
        'HOTA': ('HOTA', 'hota'),
    }
    
    n_metrics = len(metrics_config)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (key, (title, filename)) in enumerate(metrics_config.items()):
        values = []
        for exp in exp_names:
            if exp in results:
                val = results[exp].get(key, 0)
                # 将百分比转为小数再绘图
                if key in ['success_rate', 'llm_success_rate', 'cache_hit_rate', 'fallback_rate']:
                    if isinstance(val, str): val = float(val.strip('%')) / 100
                values.append(val)
            else:
                values.append(0)
        
        ax = axes[i]
        bars = ax.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#6A994E'], edgecolor='black')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        # 在柱上标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values)*0.02 if max(values) > 0 else 0.01),
                    f'{val:.3f}' if key in ['MOTA', 'IDF1', 'HOTA'] else f'{val:.2%}' if key in ['success_rate', 'llm_success_rate', 'cache_hit_rate'] else f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合对比图表已保存至 {os.path.join(output_dir, 'all_metrics_comparison.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='实验批次目录路径')
    args = parser.parse_args()
    
    results = load_experiment_results(args.exp_dir)
    plot_comparison(results, args.exp_dir)
