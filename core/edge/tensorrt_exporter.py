# core/edge/tensorrt_exporter.py (UAVagent 1.4 P2.2 - 修复版)
"""TensorRT FP16/INT8 模型导出与推理加速"""
import os, time, json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict


class TensorRTExporter:
    """YOLO -> TensorRT 导出器 (FP16 + INT8)"""

    @staticmethod
    def export_to_engine(model_path: str, output_path: str = None,
                         fp16: bool = True, int8: bool = False,
                         workspace: int = 4, imgsz: int = 640,
                         batch: int = 1,
                         calibration_images: List[str] = None) -> Optional[str]:
        """导出模型为 TensorRT Engine"""
        if output_path is None:
            stem = Path(model_path).stem
            suffix = "_int8" if int8 else ("_fp16" if fp16 else "")
            output_path = str(Path(model_path).parent / f"{stem}{suffix}.engine")
        
        try:
            from ultralytics import YOLO
            
            mode_str = "INT8" if int8 else ("FP16" if fp16 else "FP32")
            print(f"[TensorRT] 导出 {os.path.basename(model_path)} -> TensorRT {mode_str}")
            print(f"  Workspace: {workspace}GB | imgsz: {imgsz} | Batch: {batch}")
            
            model = YOLO(model_path)
            
            export_kwargs = {
                "format": "engine",
                "half": fp16 and not int8,
                "int8": int8,
                "workspace": workspace,
                "imgsz": imgsz,
                "batch": batch,
                "device": 0,
            }
            
            if int8 and calibration_images:
                export_kwargs["data"] = calibration_images[0] if len(calibration_images) == 1 else calibration_images
                print(f"  校准图像: {len(calibration_images)} 张")
            
            model.export(**export_kwargs)
            
            print(f"[TensorRT] OK 导出完成: {output_path}")
            return output_path
        
        except ImportError:
            print("[TensorRT] ERROR 请安装: pip install tensorrt onnx onnxruntime-gpu")
            return None
        except Exception as e:
            print(f"[TensorRT] ERROR 导出失败: {e}")
            return None

    @staticmethod
    def export_all_models(models_dir: str = None, fp16: bool = True,
                          int8: bool = False,
                          calibration_dir: str = None) -> Dict[str, str]:
        """批量导出所有模型"""
        if models_dir is None:
            models_dir = str(Path(__file__).resolve().parent.parent.parent / "models")
        
        calibration_images = None
        if int8 and calibration_dir:
            calibration_images = TensorRTExporter._collect_calibration_images(
                calibration_dir, max_images=100
            )
        
        results = {}
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt") and not f.endswith("_cpu.pt"):
                pt_path = os.path.join(models_dir, f)
                stem = Path(f).stem
                suffix = "_int8" if int8 else ("_fp16" if fp16 else "")
                engine_path = os.path.join(models_dir, f"{stem}{suffix}.engine")
                
                if os.path.exists(engine_path):
                    print(f"[TensorRT] {f} -> engine 已存在, 跳过")
                    results[f] = engine_path
                    continue
                
                engine = TensorRTExporter.export_to_engine(
                    pt_path, engine_path, fp16=fp16, int8=int8,
                    calibration_images=calibration_images,
                )
                if engine:
                    results[f] = engine
        
        return results
    
    @staticmethod
    def _collect_calibration_images(data_dir: str, max_images: int = 100) -> List[str]:
        """收集校准图像"""
        images = []
        if os.path.isfile(data_dir):
            return [data_dir]
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(root, f))
                    if len(images) >= max_images:
                        return images
        return images


class TensorRTInference:
    """TensorRT 推理包装器"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.model = None
        self.precision = "unknown"
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.engine_path)
            fname = os.path.basename(self.engine_path).lower()
            if "int8" in fname:
                self.precision = "INT8"
            elif "fp16" in fname:
                self.precision = "FP16"
            else:
                self.precision = "FP32"
            print(f"[TensorRT] 已加载: {os.path.basename(self.engine_path)} ({self.precision})")
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


def benchmark_all_precisions(model_path: str, engine_dir: str = None,
                              num_runs: int = 100, imgsz: int = 640):
    """对比 PyTorch FP32 / TensorRT FP16 / TensorRT INT8 推理速度"""
    import torch
    from ultralytics import YOLO

    print("=" * 70)
    print("TensorRT 全精度速度对比: PyTorch FP32 vs TRT FP16 vs TRT INT8")
    print("=" * 70)

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    results = {}

    # 1. PyTorch FP32
    print("\n[1] PyTorch FP32...")
    model_pt = YOLO(model_path)
    if torch.cuda.is_available():
        model_pt.to("cuda")
    for _ in range(10):
        model_pt(dummy, verbose=False)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(num_runs):
        model_pt(dummy, verbose=False)
    torch.cuda.synchronize()
    avg_pt = (time.perf_counter() - t0) / num_runs * 1000
    results["PyTorch FP32"] = {"avg_ms": round(avg_pt, 1), "fps": round(1000/avg_pt, 1)}
    print(f"  {avg_pt:.1f}ms ({1000/avg_pt:.0f} FPS)")

    # 2. TensorRT FP16
    stem = Path(model_path).stem
    engine_fp16 = None
    if engine_dir:
        engine_fp16 = os.path.join(engine_dir, f"{stem}_fp16.engine")
    if not engine_fp16 or not os.path.exists(engine_fp16):
        engine_fp16 = str(Path(model_path).parent / f"{stem}_fp16.engine")
    if not os.path.exists(engine_fp16):
        engine_fp16 = str(Path(model_path).parent / f"{stem}.engine")
    
    if os.path.exists(engine_fp16):
        print("\n[2] TensorRT FP16...")
        model_fp16 = YOLO(engine_fp16)
        for _ in range(10):
            model_fp16(dummy, verbose=False)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(num_runs):
            model_fp16(dummy, verbose=False)
        torch.cuda.synchronize()
        avg_fp16 = (time.perf_counter() - t0) / num_runs * 1000
        results["TensorRT FP16"] = {"avg_ms": round(avg_fp16, 1), "fps": round(1000/avg_fp16, 1)}
        speedup = avg_pt / avg_fp16
        print(f"  {avg_fp16:.1f}ms ({1000/avg_fp16:.0f} FPS) | Speedup: {speedup:.1f}x")
    else:
        print(f"\n[2] TensorRT FP16: engine 不存在 ({engine_fp16}), 跳过")

    # 3. TensorRT INT8
    engine_int8 = str(Path(model_path).parent / f"{stem}_int8.engine")
    if not os.path.exists(engine_int8) and engine_dir:
        engine_int8 = os.path.join(engine_dir, f"{stem}_int8.engine")
    
    if os.path.exists(engine_int8):
        print("\n[3] TensorRT INT8...")
        model_int8 = YOLO(engine_int8)
        for _ in range(10):
            model_int8(dummy, verbose=False)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(num_runs):
            model_int8(dummy, verbose=False)
        torch.cuda.synchronize()
        avg_int8 = (time.perf_counter() - t0) / num_runs * 1000
        results["TensorRT INT8"] = {"avg_ms": round(avg_int8, 1), "fps": round(1000/avg_int8, 1)}
        speedup = avg_pt / avg_int8
        print(f"  {avg_int8:.1f}ms ({1000/avg_int8:.0f} FPS) | Speedup: {speedup:.1f}x")
    else:
        print(f"\n[3] TensorRT INT8: engine 不存在, 跳过")

    # 总结
    print(f"\n{'='*70}")
    print("速度对比总结")
    print(f"{'='*70}")
    print(f"{'精度':<20} {'延迟':>10} {'FPS':>10} {'加速比':>10}")
    print("-" * 50)
    
    base_ms = results.get("PyTorch FP32", {}).get("avg_ms", 1)
    for name, data in results.items():
        ms = data["avg_ms"]
        fps = data["fps"]
        sp = base_ms / ms
        print(f"{name:<20} {ms:>8.1f}ms {fps:>8.1f} {sp:>8.1f}x")

    return results


if __name__ == "__main__":
    model_path = r"E:\UAVagent\models\yolo11x_visdrone.pt"
    if os.path.exists(model_path):
        engine_path = str(Path(model_path).parent / f"{Path(model_path).stem}.engine")
        if not os.path.exists(engine_path):
            TensorRTExporter.export_to_engine(model_path, fp16=True)
        
        calib_dir = r"E:\datasets\VisDrone\VisDrone2019-MOT-val\sequences\uav0000086_00000_v"
        if os.path.isdir(calib_dir):
            print(f"\n[INT8] 使用校准数据: {calib_dir}")
            TensorRTExporter.export_to_engine(model_path, fp16=False, int8=True,
                                              calibration_images=[calib_dir])
        
        benchmark_all_precisions(model_path)