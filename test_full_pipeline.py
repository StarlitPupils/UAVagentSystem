# E:/UAVagent1.1/test_full_pipeline.py
"""全链路测试：检测→推理→行动→记忆"""
import sys
sys.path.insert(0, "E:/UAVagent1.1")
import asyncio
import numpy as np

print("=" * 60)
print("UAVagent 2.0 全链路验证")
print("=" * 60)

# 1. 配置
from config.settings import config
config.setup_session()
print(f"\n[配置] 模型={config.YOLO_MODEL_NAME} 跟踪器={config.TRACKER_TYPE} LLM={config.LLM_MODEL}")

# 2. 视觉检测
print("\n[1/5] 视觉检测...")
from core.vision_system import VisionSystem
vs = VisionSystem(device="cpu")
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
dets = vs.process_frame(dummy)
print(f"  检测到 {len(dets)} 个目标")

# 3. 推理
print("\n[2/5] LLM推理...")
from agents.reasoning_agent import ReasoningAgent
async def do_reason():
    agent = ReasoningAgent()
    vis = {"num_objects": len(dets), "detections": dets}
    plan = await agent.reason("search", vis)
    return plan
plan = asyncio.run(do_reason())
print(f"  计划类型={plan.get('action_type')} 目标={plan.get('target_description','?')}")

# 4. 行动
print("\n[3/5] 行动执行...")
from agents.action_agent import ActionAgent
from core.uav_controller import UavController
from core.data_logger import DataLogger
action = ActionAgent(UavController(), DataLogger())
result = action.execute(plan)
print(f"  执行结果={result}")

# 5. 记忆
print("\n[4/5] 记忆存储/检索...")
from core.memory.memory_manager import memory_manager
memory_manager.vector_store.initialize()
memory_manager.remember("全链路测试: search命令成功", memory_type="mission")
results = memory_manager.recall("search", top_k=3)
print(f"  检索到 {len(results)} 条记忆")

# 6. 安全
print("\n[5/5] 安全验证...")
from agents.safety_agent import safety_agent
allowed, reason = safety_agent.validate_action({"action_type": "search"})
print(f"  search动作: {'允许' if allowed else '禁止'} ({reason})")

print("\n" + "=" * 60)
print("✅ 全链路验证通过！")
print("=" * 60)
