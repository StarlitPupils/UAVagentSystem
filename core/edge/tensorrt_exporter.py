# core/edge/tensorrt_exporter.py (UAVagent 1.3)
"""TensorRT FP16 模型导出与推理加速 - 预期加速 3-5x"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import time


class TensorRTExporter:
    """YOLO -> TensorRT 导出器"""

    @staticmethod
    def export_to_engine(model_path: str, output_path: str = None,
                         fp16: bool = True, workspace: int = 4,
                         imgsz: int = 640, batch: int = 1) -> Optional[str]:
        if output_path is None:
            output_path = str(Path(model_path).with_suffix(".engine"))
        try:
            from ultralytics import YOLO
            print(f"[TensorRT] 导出 {os.path.basename(model_path)} -> TensorRT")
            print(f"  FP16: {fp16} | Workspace: {workspace}GB | imgsz: {imgsz}")
            model = YOLO(model_path)
            model.export(
                format="engine",
                half=fp16,
                workspace=workspace,
                imgsz=imgsz,
                batch=batch,
                device=0,
            )
            print(f"[TensorRT] OK 导出完成: {output_path}")
            return output_path
        except ImportError:
            print("[TensorRT] ERROR 请安装: pip install tensorrt onnx onnxruntime-gpu")
            return None
        except Exception as e:
            print(f"[TensorRT] ERROR 导出失败: {e}")
            return None

    @staticmethod
    def export_all_models(models_dir: str = None, fp16: bool = True) -> Dict[str, str]:
        if models_dir is None:
            models_dir = str(Path(__file__).resolve().parent.parent.parent / "models")
        results = {}
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt") and not f.endswith("_cpu.pt"):
                pt_path = os.path.join(models_dir, f)
                engine_path = os.path.join(models_dir, f.replace(".pt", ".engine"))
                if os.path.exists(engine_path):
                    print(f"[TensorRT] {f} -> 已存在 engine, 跳过")
                    results[f] = engine_path
                    continue
                engine = TensorRTExporter.export_to_engine(pt_path, engine_path, fp16=fp16)
                if engine:
                    results[f] = engine
        return results


class TensorRTInference:
    """TensorRT 推理包装器"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.model = None
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.engine_path)
            print(f"[TensorRT] 已加载: {os.path.basename(self.engine_path)}")
        except Exception as e:
            print(f"[TensorRT] 加载失败: {e}")
            self.model = None

    def infer(self, image: np.ndarray, conf: float = 0.25, **kwargs) -> List[Dict]:
        if self.model is None:
            return []
        results = self.model(image, conf=conf, verbose=False, **kwargs)
        detections = []
        if results and results[0].boxes is not None:
            for i in range(len(results[0].boxes)):
                x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
                detections.append({
                    "bbox": [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                    "confidence": float(results[0].boxes.conf[i]),
                    "class": int(results[0].boxes.cls[i]),
                    "id": None,
                    "num_models": 1,
                })
        return detections


def benchmark_tensorrt(model_path: str, engine_path: str = None,
                       num_runs: int = 100, imgsz: int = 640):
    """对比 PyTorch vs TensorRT 推理速度"""
    import torch
    from ultralytics import YOLO

    print("=" * 60)
    print("TensorRT vs PyTorch 推理速度对比")
    print("=" * 60)

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # PyTorch
    print("\n[1] PyTorch 推理...")
    model_pt = YOLO(model_path)
    if torch.cuda.is_available():
        model_pt.to("cuda")
    for _ in range(10):
        model_pt(dummy, verbose=False)
    times_pt = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        model_pt(dummy, verbose=False)
        times_pt.append(time.perf_counter() - t0)
    avg_pt = np.mean(times_pt) * 1000
    fps_pt = 1000 / avg_pt
    print(f"  PyTorch: {avg_pt:.1f}ms ({fps_pt:.1f} FPS)")

    # TensorRT
    avg_trt = 0
    if engine_path and os.path.exists(engine_path):
        print("\n[2] TensorRT 推理...")
        model_trt = YOLO(engine_path)
        for _ in range(10):
            model_trt(dummy, verbose=False)
        times_trt = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model_trt(dummy, verbose=False)
            times_trt.append(time.perf_counter() - t0)
        avg_trt = np.mean(times_trt) * 1000
        fps_trt = 1000 / avg_trt
        speedup = avg_pt / avg_trt
        print(f"  TensorRT: {avg_trt:.1f}ms ({fps_trt:.1f} FPS)")
        print(f"\n  Speedup: {speedup:.1f}x")

    return {"pytorch_ms": round(avg_pt, 1), "tensorrt_ms": round(avg_trt, 1)}


if __name__ == "__main__":
    model_path = r"E:\UAVagent\models\yolo11x_visdrone.pt"
    if os.path.exists(model_path):
        TensorRTExporter.export_to_engine(model_path, fp16=True)
        benchmark_tensorrt(model_path, model_path.replace(".pt", ".engine"))