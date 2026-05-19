# E:/UAVagent/test_vision.py
"""视觉检测链路测试"""
import sys
sys.path.insert(0, "E:/UAVagent")

from core.vision_system import VisionSystem
import numpy as np

print("=" * 50)
print("YOLO 检测链路测试")
print("=" * 50)

# 创建虚拟图像
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 初始化视觉系统
vs = VisionSystem(device="cpu")
dets = vs.process_frame(dummy)

print(f"\n检测到 {len(dets)} 个目标")
for i, d in enumerate(dets[:5]):
    print(f"  [{i}] id={d.get('id')} cls={d.get('class')} conf={d.get('confidence'):.2f} bbox={d.get('bbox')}")

# 热切换模型测试
print("\n热切换模型测试...")
from config.settings import config
for name in ["yolo11n", "yolov8n"]:
    if config.switch_model(name):
        vs.reload_model()
        dets2 = vs.detect_only(dummy)
        print(f"  {name}: {len(dets2)} 个检测")

print("\n✅ 视觉检测链路正常")
