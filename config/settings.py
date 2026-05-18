# config/settings.py - UAVagent 2.0 配置中心
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_BASE = BASE_DIR / "output"


class Config:
    # ========== 会话管理 ==========
    OUTPUT_DIR: str = ""
    SESSION_ID: str = ""

    # ========== 检测模型 ==========
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "yolov8x.pt")
    YOLO_MODEL_NAME: str = "yolov8x"

    MODEL_REGISTRY: dict = {
        "yolov8n": str(BASE_DIR / "models" / "yolov8n.pt"),
        "yolov8x": str(BASE_DIR / "models" / "yolov8x.pt"),
        "yolo11n": str(BASE_DIR / "models" / "yolo11n.pt"),
        "yolo11x": str(BASE_DIR / "models" / "yolo11x.pt"),
    }
    DETECTION_CONFIDENCE: float = 0.25
    DETECTION_DEVICE: str = "cpu"
    # ========== 多模型融合权重 ==========
    MODEL_WEIGHTS: list = [1.0, 0.8, 0.6]  # 模型权重（对应model_registry顺序）
    ENSEMBLE_IOU_THR: float = 0.55  # WBF 融合 IoU 阈值
    ENSEMBLE_CONF_TYPE: str = "weighted_avg"  # avg / max / weighted_avg


    # ========== 跟踪器 ==========
    TRACKER_TYPE: str = "strongsort"
    TRACKER_REGISTRY: dict = {
        "strongsort": {"max_age": 30, "min_hits": 5, "iou_threshold": 0.25, "feature_weight": 0.3},
        "bytetrack": {"max_age": 25, "min_hits": 4, "iou_threshold": 0.3, "feature_weight": 0.2},
        "deepsort": {"max_age": 30, "min_hits": 5, "iou_threshold": 0.25, "feature_weight": 0.4},
        "botsort": {"max_age": 25, "min_hits": 4, "iou_threshold": 0.3, "feature_weight": 0.2},
        "transformer": {"max_age": 25, "min_hits": 4, "iou_threshold": 0.3, "feature_weight": 0.2},
    }

    # ========== LLM 配置 ==========
    LLM_PROVIDER: str = "deepseek"
    LLM_MODEL: str = "deepseek-v4-flash"
    LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 3

    # ========== 缓存与记忆 ==========
    LLM_CACHE_ENABLED: bool = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    CASE_BASE_ENABLED: bool = os.getenv("CASE_BASE_ENABLED", "true").lower() == "true"
    VECTOR_MEMORY_ENABLED: bool = os.getenv("VECTOR_MEMORY_ENABLED", "true").lower() == "true"
    CHROMA_DB_PATH: str = str(BASE_DIR / "output" / "chroma_db")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

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


config = Config()


