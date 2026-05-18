import asyncio
import cv2
import time
import os
from agents.perception_agent import PerceptionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.action_agent import ActionAgent
from agents.integration_agent import IntegrationAgent
from agents.reporting_agent import ReportingAgent
from agents.learning_agent import LearningAgent
from core.vision_system import VisionSystem
from core.uav_controller import UavController
from core.data_logger import DataLogger
from core.visdrone_loader import VisDroneLoader
from config.settings import config

async def run_real_validation():
    # 确保会话已设置（main.py 中会调用，这里也调用以防单独运行）
    if not hasattr(config, 'OUTPUT_DIR') or not config.OUTPUT_DIR:
        config.setup_session()
    
    vision = VisionSystem(device='cpu')
    uav = UavController()
    logger = DataLogger()
    perception = PerceptionAgent(vision, uav)
    reasoning = ReasoningAgent()
    action = ActionAgent(uav, logger)
    reporting = ReportingAgent(logger)
    learning = LearningAgent(logger)
    integration = IntegrationAgent(perception, reasoning, action, reporting, learning, logger)
    
    loader = VisDroneLoader("E:/datasets/VisDrone/VisDrone2019-MOT-val", "uav0000086_00000_v")
    total_frames = loader.get_frame_count()
    print(f"开始回放序列，共 {total_frames} 帧 (CPU 真实推理)")
    cv2.namedWindow('Dataset Playback', cv2.WINDOW_NORMAL)
    
    tracking_results = []
    frame_count = 0
    while True:
        frame = loader.get_next_frame()
        if frame is None: break
        vision.process_frame(frame)
        detections = vision.latest_detections
        for det in detections:
            x, y, w, h = det['bbox']
            tracking_results.append({"frame": frame_count+1, "id": det['id'], "bbox": [x-w/2, y-h/2, w, h], "conf": det['confidence']})
        for det in detections:
            x, y, w, h = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0,255,0), 2)
        cv2.imshow('Dataset Playback', frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): await integration.execute_mission("search")
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"\n[自动] 第 {frame_count} 帧，触发 search 任务")
            await integration.execute_mission("search")
        if frame_count % 10 == 0: print(f"进度: {frame_count}/{total_frames}")
    cv2.destroyAllWindows()
    
    tracking_dir = os.path.join(config.OUTPUT_DIR, "tracking")
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_file = os.path.join(tracking_dir, "uav0000086_00000_v.txt")
    with open(tracking_file, "w") as f:
        for res in tracking_results:
            f.write(f"{res['frame']},{res['id']},{res['bbox'][0]:.2f},{res['bbox'][1]:.2f},{res['bbox'][2]:.2f},{res['bbox'][3]:.2f},{res['conf']:.4f},-1,-1,-1\n")
    print(f"跟踪结果已保存至 {tracking_file}，共 {len(tracking_results)} 条")

if __name__ == "__main__":
    asyncio.run(run_real_validation())
