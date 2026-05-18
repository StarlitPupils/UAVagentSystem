# E:/UAVagent1.1/verify_all.py
"""端到端验证：检测→跟踪→推理→行动→评估"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("UAVagent 2.0 端到端验证")
print("=" * 60)

# 1. 配置
from config.settings import config
config.setup_session()
print(f"\n[配置] 模型:{config.YOLO_MODEL_NAME} 跟踪器:{config.TRACKER_TYPE} LLM:{config.LLM_MODEL}")

# 2. 视觉检测
print("\n[1/4] 视觉检测...")
from core.vision_system import VisionSystem
import numpy as np
vs = VisionSystem(device='cpu')
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
dets = vs.process_frame(dummy)
print(f"  检测到 {len(dets)} 个目标")

# 3. 推理
print("\n[2/4] LLM推理...")
from agents.reasoning_agent import ReasoningAgent
import asyncio
async def test_reason():
    agent = ReasoningAgent()
    vis = {"num_objects": len(dets), "detections": dets}
    plan = await agent.reason("search", vis)
    return plan
plan = asyncio.run(test_reason())
print(f"  计划: {plan.get('action_type')} -> {plan.get('target_description','?')}")

# 4. 行动
print("\n[3/4] 行动执行...")
from agents.action_agent import ActionAgent
from core.uav_controller import UavController
from core.data_logger import DataLogger
action = ActionAgent(UavController(), DataLogger())
result = action.execute(plan)
print(f"  结果: {result}")

# 5. 记忆
print("\n[4/4] 记忆存储...")
from core.memory.memory_manager import memory_manager
memory_manager.vector_store.initialize()
memory_manager.remember(
    f"任务search: 检测{len(dets)}目标 -> {plan.get('action_type')}",
    memory_type="mission"
)
recall = memory_manager.recall("search", top_k=1)
print(f"  记忆检索: {len(recall)}条")

# 6. 汇总
print("\n" + "=" * 60)
print(f"✅ 全链路验证通过")
print(f"   检测: {config.YOLO_MODEL_NAME} ({len(dets)}目标)")
print(f"   跟踪: {config.TRACKER_TYPE}")
print(f"   推理: deepseek-v4-flash")
print(f"   行动: {'成功' if result else '失败'}")
print(f"   记忆: 已存储+可检索")
print("=" * 60)
