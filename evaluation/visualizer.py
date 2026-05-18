# evaluation/visualizer.py
"""可视化工具 - 自动处理全成功/全失败情况"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from config.settings import config
from evaluation.analyzer import MetricsAnalyzer

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self):
        # 自动定位最新会话
        out_dir = getattr(config, 'OUTPUT_DIR', None)
        if not out_dir:
            import glob
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
            candidates = glob.glob(os.path.join(base, 'session_*'))
            out_dir = max(candidates, key=os.path.getmtime) if candidates else os.path.join(base, 'default')

        self.figures_dir = os.path.join(out_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.analyzer = MetricsAnalyzer()
        self.df = self.analyzer.load_all_metrics()

    def plot_success_rate(self):
        if self.df.empty:
            return
        plt.figure(figsize=(8, 5))
        success_by_cmd = self.df.groupby('command')['success'].mean()
        success_by_cmd.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Task Success Rate by Command')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'success_rate.png'), dpi=150)
        plt.close()
        print("✅ success_rate.png")

    def plot_latency_distribution(self):
        if self.df.empty:
            return
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        self.df['total_latency'].hist(bins=20, color='lightgreen', edgecolor='black')
        plt.title('Total Latency Distribution')
        plt.xlabel('Latency (s)')
        plt.subplot(1, 2, 2)
        if self.df['command'].nunique() > 1:
            self.df.boxplot(column='total_latency', by='command', rot=45)
            plt.title('Latency by Command')
            plt.suptitle('')
        else:
            self.df['total_latency'].plot(kind='box')
            plt.title('Latency Boxplot')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'latency_analysis.png'), dpi=150)
        plt.close()
        print("✅ latency_analysis.png")

    def plot_llm_performance(self):
        if self.df.empty or not self.df['llm_called'].any():
            return
        llm_df = self.df[self.df['llm_called']]

        plt.figure(figsize=(10, 5))

        # ----- Pie: LLM success -----
        plt.subplot(1, 2, 1)
        success_counts = llm_df['llm_success'].value_counts().to_dict()
        # 确保两个类别都存在
        labels = []
        sizes = []
        colors = []
        if success_counts.get(True, 0) > 0:
            labels.append(f'Success ({success_counts[True]})')
            sizes.append(success_counts[True])
            colors.append('lightgreen')
        if success_counts.get(False, 0) > 0:
            labels.append(f'Fail ({success_counts[False]})')
            sizes.append(success_counts[False])
            colors.append('salmon')
        if len(sizes) == 1:
            # 全是同一类：补充一个微小切片让 pie 显示正常
            labels.append('(none)')
            sizes.append(0.01)
            colors.append('lightgray')

        if sizes:
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('LLM Call Success Rate')

        # ----- Bar: Fallback -----
        plt.subplot(1, 2, 2)
        fallback_counts = llm_df['fallback_used'].value_counts().to_dict()
        fb_labels = ['No Fallback', 'Fallback']
        fb_vals = [fallback_counts.get(False, 0), fallback_counts.get(True, 0)]
        plt.bar(fb_labels, fb_vals, color=['skyblue', 'orange'])
        plt.title('Fallback Triggered')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'llm_performance.png'), dpi=150)
        plt.close()
        print("✅ llm_performance.png")

    def plot_token_consumption(self):
        if self.df.empty or 'llm_prompt_tokens' not in self.df.columns:
            return
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        cols = ['llm_prompt_tokens', 'llm_completion_tokens']
        self.df[cols].mean().plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
        plt.title('Avg Token Consumption')
        plt.xticks(rotation=0)
        plt.subplot(1, 2, 2)
        plt.scatter(self.df['llm_prompt_tokens'], self.df['llm_completion_tokens'], alpha=0.5, c='#A23B72')
        plt.xlabel('Prompt Tokens')
        plt.ylabel('Completion Tokens')
        plt.title('Token Scatter')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'token_analysis.png'), dpi=150)
        plt.close()
        print("✅ token_analysis.png")

    def generate_all(self):
        if self.df.empty:
            print("No data to visualize. Run batch_runner first.")
            return
        self.plot_success_rate()
        self.plot_latency_distribution()
        self.plot_llm_performance()
        self.plot_token_consumption()
        print(f"\nAll figures saved to: {self.figures_dir}")

if __name__ == "__main__":
    viz = Visualizer()
    viz.generate_all()
