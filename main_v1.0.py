import sys
import io
# 强制标准输入/输出使用 UTF-8 编码，避免 Windows 控制台解码错误
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import asyncio
import sys
import threading
import time
import cv2
import os
from agents.perception_agent import PerceptionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.action_agent import ActionAgent
from agents.integration_agent import IntegrationAgent
from agents.reporting_agent import ReportingAgent
from agents.learning_agent import LearningAgent
from agents.reflection_agent import ReflectionAgent
from agents.meta_agent import MetaAgent
from agents.training_agent import TrainingAgent
from core.vision_system import VisionSystem
from core.uav_controller import UavController
from core.data_logger import DataLogger
from config.settings import config
from core.visdrone_loader import VisDroneLoader

stop_voice_thread = threading.Event()
voice_thread = None

def voice_listener(integration):
    while not stop_voice_thread.is_set():
        time.sleep(10)
        if stop_voice_thread.is_set(): break
        fake_cmd = "search"
        print(f"\n[语音] 检测到命令: {fake_cmd}")
        try:
            asyncio.run(integration.execute_mission(fake_cmd))
        except Exception as e:
            print(f"[语音] 执行异常: {e}")

async def main():
    config.setup_session()
    global voice_thread
    print("="*60)
    print("🚁 无人机检测与跟踪系统 (多智能体协同) - 增强版")
    print("创新点: 多模态感知 | 意图推理 | 自进化闭环 | 认知协同 | 自动训练")
    print("="*60)
    vision = VisionSystem()
    uav = UavController()
    logger = DataLogger()
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
            try: cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt): break
            if cmd in ['exit','quit']: break
            elif cmd == 'help': print("takeoff, land, track <desc>, search, report, reflect, voice start, voice stop, run_dataset <path> <seq> [auto], train <visdrone_root> <yolo_dir>, train_optimize, exit")
            elif cmd == 'takeoff': uav.takeoff()
            elif cmd == 'land': uav.land()
            elif cmd == 'report': reporting.generate_report()
            elif cmd == 'reflect':
                sug = await reflection.analyze_logs()
                print(sug)
                if input("生成补丁？(y/n): ").lower()=='y':
                    patch = await meta.generate_patch(sug)
                    print(patch)
            elif cmd.startswith('track') or cmd.startswith('search'):
                await integration.execute_mission(cmd)
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
                    dataset_root = parts[1]; seq_name = parts[2]; auto_mode = len(parts) >= 4 and parts[3] == 'auto'
                    try:
                        loader = VisDroneLoader(dataset_root, seq_name)
                        print(f"开始回放序列 {seq_name}，共 {loader.get_frame_count()} 帧")
                        cv2.namedWindow('Dataset Playback', cv2.WINDOW_NORMAL)
                        frame_count = 0
                        while True:
                            frame = loader.get_next_frame()
                            if frame is None: break
                            perception.last_image = frame
                            vision.process_frame(frame)
                            cv2.imshow('Dataset Playback', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            key = cv2.waitKey(30) & 0xFF
                            if key == ord('q'): break
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
                if len(parts) < 3: print("用法: train <VisDrone根目录> <YOLO数据集目录>")
                else:
                    visdrone_root = parts[1]; yolo_dir = parts[2]
                    metrics = training.run_full_pipeline(visdrone_root, yolo_dir)
                    print(f"训练完成，mAP50: {metrics['mAP50']:.4f}")
            elif cmd == 'train_optimize':
                visdrone_root = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
                yolo_dir = "E:/datasets/VisDrone_YOLO"
                os.makedirs(yolo_dir, exist_ok=True)
                yaml_path = training.prepare_dataset(visdrone_root, yolo_dir)
                best_params = training.optimize_hyperparams(yaml_path, base_model=config.YOLO_MODEL_PATH)
                print(f"最佳超参数: {best_params}")
            elif cmd == 'voice start':
                if voice_thread is None or not voice_thread.is_alive():
                    stop_voice_thread.clear()
                    voice_thread = threading.Thread(target=voice_listener, args=(integration,), daemon=True)
                    voice_thread.start()
                    print("语音模式已启动")
                else: print("语音模式已在运行中")
            elif cmd == 'voice stop':
                stop_voice_thread.set()
                if voice_thread: voice_thread.join(timeout=2)
                print("语音模式已停止")
            else: print("未知命令")
    finally:
        stop_voice_thread.set()
        if voice_thread: voice_thread.join(timeout=2)
        print("系统退出。")

if __name__ == "__main__":
    asyncio.run(main())
