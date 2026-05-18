# E:/UAVagent1.1/run_full_evaluation.py
"""一键评估：批量测试 + 生成图表"""
import sys
sys.path.insert(0, "E:/UAVagent1.1")
import asyncio
import os

async def main():
    print("=" * 60)
    print("UAVagent 2.0 完整评估")
    print("=" * 60)

    # 1. 批量测试
    print("\n[1/3] 批量测试运行中...")
    from evaluation.batch_runner import BatchRunner
    import json
    with open("E:/UAVagent1.1/tests/test_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    runner = BatchRunner()
    await runner.run_scenarios(scenarios, repeats=5)
    print("批量测试完成。")

    # 2. 生成汇总
    print("\n[2/3] 生成性能汇总...")
    from evaluation.analyzer import MetricsAnalyzer
    analyzer = MetricsAnalyzer()
    summary = analyzer.generate_report()
    
    # 3. 可视化
    print("\n[3/3] 生成图表...")
    from evaluation.visualizer import Visualizer
    viz = Visualizer()
    viz.generate_all()
    
    print("\n✅ 评估完成！图表位于 output/session_xxx/figures/")

if __name__ == "__main__":
    asyncio.run(main())
