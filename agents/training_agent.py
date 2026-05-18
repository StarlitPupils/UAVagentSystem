import os
import json
import time
import torch
from typing import Dict, Any, Optional
from ultralytics import YOLO
from config.settings import config
from core.data_logger import DataLogger

class TrainingAgent:
    def __init__(self, logger: Optional[DataLogger] = None):
        self.logger = logger or DataLogger()
        self.training_history = []
        self.best_model_path = None
        self.train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Training] 训练设备: {self.train_device.upper()}")
    def prepare_dataset(self, visdrone_root: str, output_dir: str) -> str:
        print("[Training] 准备数据集...")
        yaml_content = f"path: {visdrone_root}\ntrain: images/train\nval: images/val\nnc: 10\nnames: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']"
        yaml_path = os.path.join(output_dir, "visdrone.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"[Training] 数据集配置文件已生成: {yaml_path}")
        return yaml_path
    def train(self, data_yaml: str, model_name: str = "yolov8x.pt", epochs: int = 50, imgsz: int = 640, batch: int = 8, device: str = None, **kwargs) -> Dict[str, Any]:
        if device is None:
            device = self.train_device
        print(f"[Training] 开始训练模型 {model_name}，epochs={epochs}，batch={batch}，device={device}")
        model = YOLO(model_name)
        results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=config.OUTPUT_DIR, name=f"train_{int(time.time())}", exist_ok=True, **kwargs)
        train_time = 0.0
        if hasattr(model.trainer, 'train_time'):
            train_time = model.trainer.train_time
        elif hasattr(results, 'speed') and 'train' in results.speed:
            train_time = results.speed['train']
        metrics = {
            "model": model_name, "epochs": epochs, "batch": batch, "imgsz": imgsz, "device": device,
            "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
            "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
            "recall": float(results.results_dict.get('metrics/recall(B)', 0)),
            "train_time": train_time,
            "best_model_path": str(results.save_dir / 'weights' / 'best.pt')
        }
        self.training_history.append(metrics)
        self.best_model_path = metrics["best_model_path"]
        self._save_metrics(metrics)
        print(f"[Training] 训练完成，mAP50: {metrics['mAP50']:.4f}，最佳模型: {self.best_model_path}")
        return metrics
    def optimize_hyperparams(self, data_yaml: str, base_model: str = "yolov8x.pt", trials: int = 5) -> Dict[str, Any]:
        print("[Training] 超参数优化（网格搜索）...")
        best_metrics = None
        best_params = {}
        lr_options = [0.01, 0.001]
        batch_options = [8, 16] if self.train_device == 'cuda' else [4, 8]
        for lr in lr_options:
            for batch in batch_options:
                print(f"[Training] 尝试 lr={lr}, batch={batch}")
                metrics = self.train(data_yaml, model_name=base_model, epochs=20, batch=batch, lr0=lr, device=self.train_device)
                if best_metrics is None or metrics['mAP50'] > best_metrics['mAP50']:
                    best_metrics = metrics
                    best_params = {'lr0': lr, 'batch': batch}
        print(f"[Training] 最佳超参数: {best_params}，mAP50: {best_metrics['mAP50']:.4f}")
        return best_params
    def deploy_model(self, source_path: str, target_path: str = None):
        if target_path is None:
            target_path = config.YOLO_MODEL_PATH
        import shutil
        shutil.copy2(source_path, target_path)
        print(f"[Training] 新模型已部署到 {target_path}")
        self.logger.log_event("model_deployed", {"source": source_path, "target": target_path})
    def _save_metrics(self, metrics: Dict):
        metrics_file = os.path.join(config.REPORT_DIR, "training_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        all_metrics.append(metrics)
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    def run_full_pipeline(self, visdrone_root: str, yolo_dataset_dir: str):
        yaml_path = self.prepare_dataset(visdrone_root, yolo_dataset_dir)
        metrics = self.train(yaml_path, model_name=config.YOLO_MODEL_PATH, epochs=50, device=self.train_device)
        self.deploy_model(metrics['best_model_path'])
        return metrics