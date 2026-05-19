# E:/UAVagent/generate_final_charts.py
"""生成 MOT 对比图表 + 综合报告"""
import json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载最新结果
base = "E:/UAVagent/output"
sessions = glob.glob(os.path.join(base, "session_*"))
latest = max(sessions, key=os.path.getmtime) if sessions else None

if latest:
    # 找 mot 结果
    mot_file = os.path.join(latest, "real_mot_benchmark.json")
    if os.path.exists(mot_file):
        with open(mot_file) as f:
            data = json.load(f)
        
        metrics = ['MOTA', 'IDF1', 'Recall', 'Precision', 'ID Switches']
        single_vals = [data['single_metrics'].get(m.lower(), 0) for m in metrics]
        ensemble_vals = [data['ensemble_metrics'].get(m.lower(), 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, single_vals, width, label='Single Model', color='#2E86AB')
        rects2 = ax.bar(x + width/2, ensemble_vals, width, label='Ensemble Fusion', color='#A23B72')
        
        ax.set_ylabel('Score')
        ax.set_title(f'UAVagent 1.1 MOT Performance on VisDrone (30 frames)\n'
                     f'MOTA improvement: +{data.get("mota_improvement_ppt", 0)} ppt, '
                     f'IDF1: +{data.get("idf1_improvement_ppt", 0)} ppt')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 数值标注
        for bars in [rects1, rects2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        chart_path = os.path.join(latest, "mot_comparison.png")
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ 图表已保存: {chart_path}")
else:
    print("未找到结果")
