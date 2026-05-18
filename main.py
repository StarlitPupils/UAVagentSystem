# main.py - UAVagent 2.0 主入口
"""UAVagent 2.0 - 自进化多智能体协同无人机检测与跟踪系统"""
import sys
import io
import asyncio
import threading
import time
import cv2
import os

# UTF-8 编码
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from agents.perception_agent import PerceptionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.action_agent import ActionAgent
from agents.integration_agent import IntegrationAgent
from agents.reporting_agent import ReportingAgent
from agents.learning_agent import LearningAgent
from agents.reflection_agent import ReflectionAgent
from agents.meta_agent import MetaAgent
from agents.training_agent import TrainingAgent
from agents.safety_agent import safety_agent  # 2.0新增
from core.vision_system import VisionSystem
from core.uav_controller import UavController
from core.data_logger import DataLogger
from core.visdrone_loader import VisDroneLoader
from core.llm.llm_client import llm_client  # 2.0新增
from core.memory.memory_manager import memory_manager  # 2.0新增
from core.tracking.tracker_registry import tracker_registry  # 2.0新增
from config.settings import config

stop_voice_thread = threading.Event()
voice_thread = None


def voice_listener(integration):
    while not stop_voice_thread.is_set():
        time.sleep(10)
        if stop_voice_thread.is_set():
            break
        fake_cmd = "search"
        print(f"\n[语音] 检测到命令: {fake_cmd}")
        try:
            asyncio.run(integration.execute_mission(fake_cmd))
        except Exception as e:
            print(f"[语音] 执行异常: {e}")


async def main():
    config.setup_session()
    global voice_thread

    print("=" * 60)
    print("🚁 UAVagent 2.0 — 自进化多智能体协同系统")
    print(f"   LLM: {config.LLM_MODEL} | 检测: {config.YOLO_MODEL_NAME}")
    print(f"   跟踪: {config.TRACKER_TYPE} | 记忆: {'向量库' if config.VECTOR_MEMORY_ENABLED else 'JSON'}")
    print(f"   安全: {'启用' if config.SAFETY_ENABLED else '禁用'}")
    print("=" * 60)

    # 初始化核心模块
    vision = VisionSystem()
    uav = UavController()
    logger = DataLogger()

    # 初始化向量记忆库
    if config.VECTOR_MEMORY_ENABLED:
        memory_manager.vector_store.initialize()
        print(f"[记忆] 向量记忆库就绪")

    # 初始化智能体
    perception = PerceptionAgent(vision, uav)
    reasoning = ReasoningAgent()
    action = ActionAgent(uav, logger)
    reporting = ReportingAgent(logger)
    learning = LearningAgent(logger)
    reflection = ReflectionAgent(logger)
    meta = MetaAgent()
    training = TrainingAgent(logger)
    integration = IntegrationAgent(perception, reasoning, action, reporting, learning, logger)

    try:
        while True:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd in ['exit', 'quit']:
                break
            elif cmd == 'help':
                print("""
    🚁 UAVagent 2.0 命令列表:
      takeoff / land          — 起飞/降落
      track <描述>            — 跟踪目标 (例: track white car)
      search                  — 搜索目标
      report                  — 生成报告
      reflect                 — 触发反思进化
      status                  — 系统状态
      switch_model <name>     — 切换检测模型 (yolov8n/yolov8x/yolo11n/yolo11x)
      switch_tracker <type>   — 切换跟踪器 (strongsort/bytetrack/transformer)
      benchmark               — 运行检测器基准测试
      memory <query>          — 搜索记忆库
      voice start/stop        — 语音模式
      run_dataset <path> <seq> [auto] — 数据集回放
      train <visdrone_root> <yolo_dir> — 训练模型
      api                     — 启动API服务器 (http://localhost:8000)
      dashboard               — 启动API+Dashboard
                """)
            elif cmd == 'takeoff':
                allowed, reason = safety_agent.validate_action({"action_type": "takeoff"})
                if allowed:
                    uav.takeoff()
                else:
                    print(f"❌ 安全拦截: {reason}")
            elif cmd == 'land':
                uav.land()
            elif cmd == 'status':
                print(f"📊 系统状态:")
                print(f"   检测模型: {config.YOLO_MODEL_NAME}")
                print(f"   跟踪器: {config.TRACKER_TYPE}")
                print(f"   LLM: {config.LLM_MODEL}")
                print(f"   可用跟踪器: {tracker_registry.list_available()}")
                print(f"   可用模型: {list(config.MODEL_REGISTRY.keys())}")
                stats = llm_client.get_stats()
                print(f"   LLM调用: {stats['call_count']}次 | 缓存: {stats['cache_size']}条")
            elif cmd.startswith('switch_model '):
                model_name = cmd.split()[1]
                if config.switch_model(model_name):
                    print(f"✅ 已切换检测模型: {model_name}")
                    vision.reload_model()
                else:
                    print(f"❌ 未知模型: {model_name}，可选: {list(config.MODEL_REGISTRY.keys())}")
            elif cmd.startswith('switch_tracker '):
                tracker_type = cmd.split()[1]
                if config.switch_tracker(tracker_type):
                    print(f"✅ 已切换跟踪器: {tracker_type}")
                else:
                    print(f"❌ 未知跟踪器: {tracker_type}，可选: {tracker_registry.list_available()}")
            elif cmd == 'benchmark':
                from core.edge.exporter import ModelExporter
                for name, path in config.MODEL_REGISTRY.items():
                    if os.path.exists(path):
                        result = ModelExporter.benchmark(path, config.DETECTION_DEVICE)
                        print(f"  {name}: {result.get('fps', 'N/A')} FPS | {result.get('avg_latency_ms', 'N/A')}ms")
            elif cmd.startswith('memory '):
                query = cmd[7:]
                results = memory_manager.recall(query, top_k=5)
                print(f"🔍 记忆搜索: '{query}'")
                for r in results:
                    print(f"  [{r['metadata'].get('type','?')}] {r['content'][:150]}... (相关度:{r['score']:.2f})")
            elif cmd == 'report':
                reporting.generate_report()
            elif cmd == 'reflect':
                sug = await reflection.analyze_logs()
                print(sug)
                if input("生成补丁？(y/n): ").lower() == 'y':
                    patch = await meta.generate_patch(sug)
                    print(patch)
            elif cmd.startswith('track') or cmd.startswith('search'):
                # 安全验证
                allowed, reason = safety_agent.validate_action({"action_type": cmd.split()[0]})
                if allowed:
                    await integration.execute_mission(cmd)
                else:
                    print(f"❌ 安全拦截: {reason}")
            elif cmd.startswith('generate_agent'):
                parts = cmd.split(' ', 2)
                if len(parts) >= 3:
                    result = await meta.generate_custom_agent(parts[1], parts[2])
                    print(result)
                else:
                    print("用法: generate_agent <AgentName> <需求描述>")
            elif cmd.startswith('run_dataset'):
                parts = cmd.split()
                if len(parts) < 3:
                    print("用法: run_dataset <数据集根目录> <序列名> [auto]")
                else:
                    dataset_root = parts[1]
                    seq_name = parts[2]
                    auto_mode = len(parts) >= 4 and parts[3] == 'auto'
                    try:
                        loader = VisDroneLoader(dataset_root, seq_name)
                        print(f"开始回放序列 {seq_name}，共 {loader.get_frame_count()} 帧")
                        cv2.namedWindow('Dataset Playback', cv2.WINDOW_NORMAL)
                        frame_count = 0
                        while True:
                            frame = loader.get_next_frame()
                            if frame is None:
                                break
                            perception.last_image = frame
                            vision.process_frame(frame)
                            cv2.imshow('Dataset Playback', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            key = cv2.waitKey(30) & 0xFF
                            if key == ord('q'):
                                break
                            elif key == ord('s') and not auto_mode:
                                await integration.execute_mission("search")
                            frame_count += 1
                            if auto_mode and frame_count % 30 == 0:
                                print(f"\n[自动] 第 {frame_count} 帧，触发 search 任务")
                                await integration.execute_mission("search")
                        cv2.destroyAllWindows()
                    except Exception as e:
                        print(f"数据集回放失败: {e}")
            elif cmd.startswith('train '):
                parts = cmd.split()
                if len(parts) < 3:
                    print("用法: train <VisDrone根目录> <YOLO数据集目录>")
                else:
                    visdrone_root = parts[1]
                    yolo_dir = parts[2]
                    metrics = training.run_full_pipeline(visdrone_root, yolo_dir)
                    print(f"训练完成，mAP50: {metrics['mAP50']:.4f}")
            elif cmd == 'train_optimize':
                visdrone_root = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
                yolo_dir = "E:/datasets/VisDrone_YOLO"
                os.makedirs(yolo_dir, exist_ok=True)
                yaml_path = training.prepare_dataset(visdrone_root, yolo_dir)
                best_params = training.optimize_hyperparams(yaml_path, base_model=config.YOLO_MODEL_PATH)
                print(f"最佳超参数: {best_params}")
            elif cmd == 'api':
                print("🚀 启动 API 服务器...")
                import subprocess
                subprocess.Popen([sys.executable, "api/server.py"])
                print(f"API 地址: http://localhost:{config.API_PORT}")
                print(f"Dashboard: http://localhost:{config.API_PORT}/dashboard")
            elif cmd == 'dashboard':
                print("🚀 启动 API + Dashboard...")
                import subprocess
                subprocess.Popen([sys.executable, "api/server.py"])
                time.sleep(2)
                print(f"Dashboard: http://localhost:{config.API_PORT}/dashboard")
            elif cmd == 'voice start':
                if voice_thread is None or not voice_thread.is_alive():
                    stop_voice_thread.clear()
                    voice_thread = threading.Thread(target=voice_listener, args=(integration,), daemon=True)
                    voice_thread.start()
                    print("语音模式已启动")
                else:
                    print("语音模式已在运行中")
            elif cmd == 'voice stop':
                stop_voice_thread.set()
                if voice_thread:
                    voice_thread.join(timeout=2)
                print("语音模式已停止")
            else:
                print("未知命令，输入 'help' 查看帮助")
    finally:
        stop_voice_thread.set()
        if voice_thread:
            voice_thread.join(timeout=2)
        print("\n🛬 UAVagent 2.0 系统退出。")

if __name__ == "__main__":
    asyncio.run(main())
