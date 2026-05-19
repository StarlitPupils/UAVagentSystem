# verify_env.py
"""UAVagent 2.0 环境验证"""
import sys
print(f"Python: {sys.version}")
print(f"虚拟环境: {sys.prefix}")
print(f"项目目录: E:\\UAVagent")
print("-" * 50)

checks = []

# 1. PyTorch + CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    checks.append(("PyTorch CUDA", cuda_ok, f"v{torch.__version__}, GPU: {torch.cuda.get_device_name(0) if cuda_ok else 'N/A'}"))
except Exception as e:
    checks.append(("PyTorch", False, str(e)))

# 2. Ultralytics (YOLO)
try:
    import ultralytics
    checks.append(("Ultralytics", True, f"v{ultralytics.__version__}"))
except Exception as e:
    checks.append(("Ultralytics", False, str(e)))

# 3. OpenAI
try:
    import openai
    checks.append(("OpenAI SDK", True, f"v{openai.__version__}"))
except Exception as e:
    checks.append(("OpenAI SDK", False, str(e)))

# 4. ChromaDB
try:
    import chromadb
    checks.append(("ChromaDB", True, f"v{chromadb.__version__}"))
except Exception as e:
    checks.append(("ChromaDB", False, str(e)))

# 5. FastAPI
try:
    import fastapi
    checks.append(("FastAPI", True, f"v{fastapi.__version__}"))
except Exception as e:
    checks.append(("FastAPI", False, str(e)))

# 6. YOLO模型文件
import os
models_dir = r"E:\UAVagent\models"
yolo11x = os.path.exists(os.path.join(models_dir, "yolo11x.pt"))
yolov8x = os.path.exists(os.path.join(models_dir, "yolov8x.pt"))
checks.append(("yolo11x.pt", yolo11x, "已下载" if yolo11x else "缺失"))
checks.append(("yolov8x.pt", yolov8x, "已下载" if yolov8x else "缺失"))

# 7. LLM配置
from dotenv import load_dotenv
load_dotenv(r"E:\UAVagent\.env")
api_key = os.getenv("DEEPSEEK_API_KEY", "")
checks.append(("DeepSeek API Key", bool(api_key), "已配置" if api_key else "❌ 未配置"))

# 输出结果
print("\n📊 环境检查结果:")
print("-" * 50)
all_ok = True
for name, ok, detail in checks:
    status = "✅" if ok else "❌"
    print(f"  {status} {name}: {detail}")
    if not ok:
        all_ok = False

print("-" * 50)
if all_ok:
    print("🎉 所有检查通过！可以运行 python main.py")
else:
    print("⚠️  部分检查未通过，请查看上方详情")
