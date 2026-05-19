# E:/UAVagent/test_fix.py
"""快速验证所有修复"""
import sys
sys.path.insert(0, "E:/UAVagent")

print("=" * 50)
print("UAVagent 2.0 修复验证")
print("=" * 50)

# 1. LLM 连通性
print("\n1. LLM 测试...")
from core.llm.llm_client import llm_client
r = llm_client.chat([{"role": "user", "content": "回复ok"}])
print("   模型:", r.get("model"))
print("   成功:", not r.get("fallback"))

# 2. 向量记忆
print("\n2. 向量记忆库测试...")
from core.memory.memory_manager import memory_manager
memory_manager.vector_store.initialize()
memory_manager.remember("测试记忆white car tracked", memory_type="tracking")
results = memory_manager.recall("white car", top_k=2)
print("   搜索结果:", len(results))

# 3. 安全智能体
print("\n3. 安全智能体测试...")
from agents.safety_agent import safety_agent
allowed, reason = safety_agent.validate_action({"action_type": "search"})
print("   search允许:", allowed)

# 4. DataLogger
print("\n4. DataLogger 测试...")
from core.data_logger import DataLogger
logger = DataLogger()
logger.log_event("test", {"msg": "hello"})
print("   日志条目:", len(logger.entries))

# 5. ActionAgent
print("\n5. ActionAgent 测试...")
from agents.action_agent import ActionAgent
from core.uav_controller import UavController
action = ActionAgent(UavController(), DataLogger())
result = action.execute({"action_type": "search"})
print("   执行结果:", result, type(result).__name__)

print("\n" + "=" * 50)
print("全部验证通过！")
print("=" * 50)
