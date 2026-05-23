# E:/UAVagent/bench_tensorrt_speed.py
"""TensorRT vs PyTorch 纯推理速度基准"""
import time, numpy as np, torch, os

MODEL_PT = r"E:\UAVagent\models\yolo11x_visdrone.pt"
MODEL_ENGINE = r"E:\UAVagent\models\yolo11x_visdrone.engine"

def benchmark_pytorch():
    from ultralytics import YOLO
    model = YOLO(MODEL_PT)
    model.to('cuda')
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 预热
    for _ in range(20): model(dummy, verbose=False)
    torch.cuda.synchronize()
    
    # 计时
    t0 = time.perf_counter()
    for _ in range(100): model(dummy, verbose=False)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    print(f"PyTorch:   {t:.1f} ms ({1000/t:.0f} FPS)")
    return t

def benchmark_tensorrt():
    from ultralytics import YOLO
    model = YOLO(MODEL_ENGINE)
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 预热
    for _ in range(20): model(dummy, verbose=False)
    torch.cuda.synchronize()
    
    # 计时
    t0 = time.perf_counter()
    for _ in range(100): model(dummy, verbose=False)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    print(f"TensorRT:  {t:.1f} ms ({1000/t:.0f} FPS)")
    return t

if __name__ == "__main__":
    print("="*50)
    print("TensorRT vs PyTorch 推理速度对比")
    print("="*50)
    
    t_pt = benchmark_pytorch()
    if os.path.exists(MODEL_ENGINE):
        t_trt = benchmark_tensorrt()
        speedup = t_pt / t_trt
        print(f"\n  加速比: {speedup:.1f}x")
        if speedup > 2:
            print("  TensorRT 加速生效！")
        else:
            print("  TensorRT 加速不明显，可能原因：")
            print("  - YOLO 内部的 TensorRT 路径需要额外优化")
            print("  - FP16 精度与 PyTorch FP32 速度接近（已很快）")
    else:
        print(f"\n  Engine 文件不存在: {MODEL_ENGINE}")