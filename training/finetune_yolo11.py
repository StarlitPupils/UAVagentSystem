# E:/UAVagent/training/finetune_yolo11.py (UAVagent 1.4 P3.2)
"""YOLO11x VisDrone 微调训练 — 优化版 (更长的训练 + 更低学习率)"""
import os, sys, time, json, argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_environment():
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {'OK' if cuda_available else 'CPU only'}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("请安装: pip install ultralytics>=8.3.0")
        return False
    return True


def train_yolo11(data_yaml: str, output_dir: str = None, **kwargs):
    """YOLO11x 微调训练 (v1.4 优化版)
    
    P3.2 优化:
    - lr0=0.0005 (1.3 是 0.001) → 更稳定的收敛
    - epochs=50+EarlyStopping → 充分训练
    - warmup_epochs=5 → 更平滑的启动
    - weight_decay=0.0005 → 适度正则化
    """
    from ultralytics import YOLO
    
    params = {
        "model": r"E:\UAVagent\models\yolo11x.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "workers": 4,
        "lr0": 0.0005,            # P3.2: 更低学习率
        "lrf": 0.005,             # P3.2: 最终学习率 = lr0 * lrf
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,       # P3.2: 更长预热
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 30,           # P3.2: 更多耐心
        "save": True,
        "save_period": 5,         # P3.2: 更频繁保存
        "val": True,
        "plots": True,
        "project": output_dir or "E:/UAVagent/output/training",
        "name": f"yolo11x_visdrone_v14_{time.strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": True,
    }
    
    params.update(kwargs)
    
    print("=" * 60)
    print("YOLO11x VisDrone 微调训练 (v1.4 优化版)")
    print("=" * 60)
    print(f"数据: {data_yaml}")
    print(f"基础模型: {params['model']}")
    print(f"Epochs: {params['epochs']} | Batch: {params['batch']}")
    print(f"lr0: {params['lr0']} | lrf: {params['lrf']} (final_lr={params['lr0']*params['lrf']:.6f})")
    print(f"Patience: {params['patience']} | Warmup: {params['warmup_epochs']}")
    print(f"输出: {os.path.join(params['project'], params['name'])}")
    
    model_path = params.pop('model')
    model = YOLO(model_path)
    
    print(f"\nTraining...")
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=params['epochs'], imgsz=params['imgsz'], batch=params['batch'],
        device=params['device'], workers=params['workers'],
        lr0=params['lr0'], lrf=params['lrf'],
        momentum=params['momentum'], weight_decay=params['weight_decay'],
        warmup_epochs=params['warmup_epochs'], warmup_momentum=params['warmup_momentum'],
        warmup_bias_lr=params['warmup_bias_lr'],
        box=params['box'], cls=params['cls'], dfl=params['dfl'],
        hsv_h=params['hsv_h'], hsv_s=params['hsv_s'], hsv_v=params['hsv_v'],
        degrees=params['degrees'], translate=params['translate'],
        scale=params['scale'], shear=params['shear'], perspective=params['perspective'],
        flipud=params['flipud'], fliplr=params['fliplr'],
        mosaic=params['mosaic'], mixup=params['mixup'], copy_paste=params['copy_paste'],
        patience=params['patience'], save=params['save'],
        save_period=params['save_period'], val=params['val'], plots=params['plots'],
        project=params['project'], name=params['name'], exist_ok=params['exist_ok'],
    )
    
    train_time = time.time() - start_time
    best_pt = os.path.join(params['project'], params['name'], 'weights', 'best.pt')
    last_pt = os.path.join(params['project'], params['name'], 'weights', 'last.pt')
    
    metrics = {
        "version": "1.4",
        "model": "yolo11x",
        "data": data_yaml,
        "lr0": params['lr0'],
        "epochs": params['epochs'],
        "batch": params['batch'],
        "train_time_hours": round(train_time / 3600, 2),
        "best_model": best_pt,
        "last_model": last_pt,
    }
    
    if hasattr(results, 'results_dict'):
        metrics.update({
            "mAP50": round(float(results.results_dict.get('metrics/mAP50(B)', 0)), 4),
            "mAP50_95": round(float(results.results_dict.get('metrics/mAP50-95(B)', 0)), 4),
            "precision": round(float(results.results_dict.get('metrics/precision(B)', 0)), 4),
            "recall": round(float(results.results_dict.get('metrics/recall(B)', 0)), 4),
        })
    
    metrics_path = os.path.join(params['project'], params['name'], 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training done! Time: {train_time/3600:.1f}h")
    print(f"   Best: {best_pt}")
    print(f"   mAP50: {metrics.get('mAP50', 'N/A')}")
    print(f"   mAP50-95: {metrics.get('mAP50_95', 'N/A')}")
    print(f"{'='*60}")
    
    return metrics, best_pt


def main():
    parser = argparse.ArgumentParser(description="YOLO11x VisDrone 微调训练 (v1.4)")
    parser.add_argument("--data", type=str, default="E:/datasets/VisDrone_YOLO/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--deploy", action="store_true", help="训练后自动部署到models/")
    args = parser.parse_args()
    
    if not check_environment():
        return
    
    kwargs = {"epochs": args.epochs, "batch": args.batch, "lr0": args.lr0}
    if args.device:
        kwargs["device"] = args.device
    
    metrics, best_model = train_yolo11(args.data, args.output, **kwargs)
    
    if args.deploy and best_model and os.path.exists(best_model):
        import shutil
        target = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo11x_visdrone_v14.pt")
        shutil.copy2(best_model, target)
        print(f"Model deployed: {target}")


if __name__ == "__main__":
    main()