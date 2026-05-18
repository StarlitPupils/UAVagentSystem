# core/edge/exporter.py
"""模型导出工具 - PyTorch → ONNX / TensorRT"""
import os
from pathlib import Path
from config.settings import config

class ModelExporter:
    @staticmethod
    def to_onnx(model_path: str = None, output_path: str = None, simplify: bool = True):
        """导出YOLO模型为ONNX格式"""
        if model_path is None:
            model_path = config.YOLO_MODEL_PATH
        if output_path is None:
            output_path = str(Path(model_path).with_suffix('.onnx'))
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.export(format='onnx', simplify=simplify)
            print(f"[ONNX] 导出完成: {output_path}")
            return output_path
        except Exception as e:
            print(f"[ONNX] 导出失败: {e}")
            return None

    @staticmethod
    def to_tensorrt(model_path: str = None, output_path: str = None, fp16: bool = True):
        """导出YOLO模型为TensorRT格式（需要GPU）"""
        if model_path is None:
            model_path = config.YOLO_MODEL_PATH
        if output_path is None:
            output_path = str(Path(model_path).with_suffix('.engine'))
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.export(format='engine', half=fp16)
            print(f"[TensorRT] 导出完成: {output_path}")
            return output_path
        except Exception as e:
            print(f"[TensorRT] 导出失败: {e}")
            print("[TensorRT] 提示: 需要NVIDIA GPU + TensorRT库")
            return None

    @staticmethod
    def benchmark(model_path: str, device: str = "cpu", iterations: int = 100):
        """模型推理基准测试"""
        import time
        import numpy as np
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(device)
            # 预热
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            for _ in range(10):
                model(dummy_input, verbose=False)
            # 计时
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                model(dummy_input, verbose=False)
                times.append(time.perf_counter() - t0)
            avg_latency = np.mean(times) * 1000
            fps = 1000 / avg_latency
            return {"model": os.path.basename(model_path), "device": device,
                    "avg_latency_ms": round(avg_latency, 2), "fps": round(fps, 1),
                    "iterations": iterations}
        except Exception as e:
            return {"error": str(e)}
