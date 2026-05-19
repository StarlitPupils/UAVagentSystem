# config/settings.py - UAVagent 1.2 配置中心 (v1.2升级)
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_BASE = BASE_DIR / "output"


class Config:
    # ========== 版本信息 ==========
    VERSION: str = "1.2.0"

    # ========== 会话管理 ==========
    OUTPUT_DIR: str = ""
    SESSION_ID: str = ""

    # ========== 检测模型 (v1.2: 新增YOLOv10/RT-DETR) ==========
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "yolo11x.pt")
    YOLO_MODEL_NAME: str = "yolo11x"

    MODEL_REGISTRY: dict = {
        # 原有模型
        "yolov8n": str(BASE_DIR / "models" / "yolov8n.pt"),
        "yolov8x": str(BASE_DIR / "models" / "yolov8x.pt"),
        "yolo11n": str(BASE_DIR / "models" / "yolo11n.pt"),
        "yolo11x": str(BASE_DIR / "models" / "yolo11x.pt"),
        # v1.2 新增
        "yolov10n": str(BASE_DIR / "models" / "yolov10n.pt"),
        "yolov10x": str(BASE_DIR / "models" / "yolov10x.pt"),
        "rtdetr-l": str(BASE_DIR / "models" / "rtdetr-l.pt"),
        "rtdetr-x": str(BASE_DIR / "models" / "rtdetr-x.pt"),
    }
    DETECTION_CONFIDENCE: float = 0.25
    DETECTION_DEVICE: str = "cuda"  # v1.2: 默认CUDA
    DETECTION_IMG_SIZE: int = 640
    DETECTION_FP16: bool = False  # v1.2: FP16推理加速

    # ========== 多模型融合权重 (v1.2: 动态权重) ==========
    MODEL_WEIGHTS: list = [1.0, 0.8, 0.6, 0.5, 0.4]  # 支持5个模型
    ENSEMBLE_IOU_THR: float = 0.50
    ENSEMBLE_CONF_TYPE: str = "weighted_avg"
    ENSEMBLE_MIN_MODELS: int = 2  # v1.2: 最少确认模型数

    # ========== 图像预处理 (v1.2 新增) ==========
    PREPROCESSING_ENABLED: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID: tuple = (8, 8)
    HIST_EQUALIZATION: bool = False  # 全局直方图均衡
    ADAPTIVE_THRESHOLD: bool = True  # v1.2: 自适应检测阈值

    # ========== 跟踪器 (v1.2: 新增EKF/深度ReID) ==========
    TRACKER_TYPE: str = "enhanced"
    TRACKER_REGISTRY: dict = {
        "enhanced": {
            "max_age": 30, "min_hits": 5, "iou_threshold": 0.25,
            "feature_weight": 0.35, "use_ekf": True,  # v1.2: EKF
            "interpolate_gaps": True, "max_gap_frames": 10,  # v1.2: 轨迹插值
        },
        "strongsort": {"max_age": 30, "min_hits": 5, "iou_threshold": 0.25, "feature_weight": 0.3},
        "bytetrack": {"max_age": 25, "min_hits": 4, "iou_threshold": 0.3, "feature_weight": 0.2},
        "deepsort": {"max_age": 30, "min_hits": 5, "iou_threshold": 0.25, "feature_weight": 0.4},
        "botsort": {"max_age": 25, "min_hits": 4, "iou_threshold": 0.3, "feature_weight": 0.2},
        "transformer": {"max_age": 25, "min_hits": 4, "iou_threshold": 0.3, "feature_weight": 0.2},
    }

    # ========== ReID 特征提取 (v1.2 新增) ==========
    REID_ENABLED: bool = True
    REID_MODEL: str = "osnet_x1_0"  # osnet_x1_0 / osnet_ain_x1_0 / resnet50
    REID_FEATURE_DIM: int = 512

    # ========== LLM 配置 (v1.2: 新增Ollama本地模型+路由) ==========
    LLM_PROVIDER: str = "deepseek"
    LLM_MODEL: str = "deepseek-v4-flash"
    LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 3

    # v1.2: Ollama本地模型
    OLLAMA_ENABLED: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"  # 推荐本地模型
    OLLAMA_VISION_MODEL: str = "llava:13b"  # 视觉模型

    # v1.2: LLM路由 (按任务复杂度选择模型)
    LLM_ROUTING_ENABLED: bool = True
    LLM_ROUTING_SIMPLE_MODEL: str = "ollama"  # 简单任务用本地模型
    LLM_ROUTING_COMPLEX_MODEL: str = "deepseek"  # 复杂任务用云端

    # ========== 缓存与记忆 ==========
    LLM_CACHE_ENABLED: bool = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    CASE_BASE_ENABLED: bool = os.getenv("CASE_BASE_ENABLED", "true").lower() == "true"
    VECTOR_MEMORY_ENABLED: bool = os.getenv("VECTOR_MEMORY_ENABLED", "true").lower() == "true"
    CHROMA_DB_PATH: str = str(BASE_DIR / "output" / "chroma_db")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ========== ReAct 多轮推理 (v1.2 新增) ==========
    REACT_MAX_ROUNDS: int = 3
    REACT_TIMEOUT: int = 120

    # ========== 训练数据与报告 ==========
    TRAINING_DATA_DIR: str = str(BASE_DIR / "output" / "training_data")
    REPORT_DIR: str = str(BASE_DIR / "output" / "reports")
    LOG_DIR: str = str(BASE_DIR / "output" / "logs")

    # ========== API 服务 ==========
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: list = ["*"]

    # ========== 安全 ==========
    SAFETY_ENABLED: bool = True
    MAX_ALTITUDE: float = 120.0
    MIN_BATTERY_RETURN: float = 25.0
    GEO_FENCE_RADIUS: float = 500.0

    # ========== 仿真 ==========
    SIMULATION_MODE: bool = os.getenv("SIMULATION_MODE", "true").lower() == "true"

    # ========== 实验 ==========
    EXPERIMENT_REPEATS: int = 5
    DATASET_ROOT: str = "E:/datasets/VisDrone/VisDrone2019-MOT-val"

    # ========== TensorRT/边缘部署 (v1.2 新增) ==========
    TENSORRT_ENABLED: bool = False
    TENSORRT_WORKSPACE: int = 4  # GB
    TENSORRT_FP16: bool = True

    # ========== 多机协同 (v1.2 新增) ==========
    MULTI_DRONE_ENABLED: bool = False
    MAVLINK_PORT: str = "COM3"
    DRONE_IDS: list = ["drone_0", "drone_1", "drone_2"]

    @classmethod
    def setup_session(cls, session_dir: str = None):
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = str(OUTPUT_BASE / f"session_{timestamp}")
        cls.OUTPUT_DIR = session_dir
        cls.SESSION_ID = Path(session_dir).name
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(os.path.join(session_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "tracking"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "training_data"), exist_ok=True)
        return session_dir

    @classmethod
    def switch_model(cls, model_name: str):
        if model_name in cls.MODEL_REGISTRY:
            cls.YOLO_MODEL_PATH = cls.MODEL_REGISTRY[model_name]
            cls.YOLO_MODEL_NAME = model_name
            return True
        return False

    @classmethod
    def switch_tracker(cls, tracker_type: str):
        if tracker_type in cls.TRACKER_REGISTRY:
            cls.TRACKER_TYPE = tracker_type
            return True
        return False

    @classmethod
    def get_available_models(cls) -> list:
        """返回已下载的模型列表"""
        available = []
        for name, path in cls.MODEL_REGISTRY.items():
            if os.path.exists(path):
                available.append(name)
        return available


config = Config()
