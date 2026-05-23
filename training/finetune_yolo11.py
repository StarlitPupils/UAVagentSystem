# E:/UAVagent/training/finetune_yolo11.py (UAVagent 1.3)
"""YOLO11x 在 VisDrone 数据集上微调训练 (P0: 预期Recall +10~15pp)"""
import os
import sys
import argparse
import time
import json
from pathlib import Path
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_environment():
    """检查训练环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {'✅ 可用' if cuda_available else '⚠️ 不可用 (将使用CPU训练，速度较慢)'}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Ultralytics
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ 请安装: pip install ultralytics>=8.3.0")
        return False
    
    return True


def train_yolo11(data_yaml: str, output_dir: str = None, **kwargs):
    """
    YOLO11x 微调训练
    
    Args:
        data_yaml: 数据集配置文件路径
        output_dir: 输出目录
        **kwargs: 训练参数覆盖
    """
    from ultralytics import YOLO
    
    # 默认参数
    params = {
        "model": r"E:\UAVagent\models\yolo11x.pt",           # 基础模型
        "epochs": 100,                    # 训练轮数
        "imgsz": 640,                     # 图像尺寸
        "batch": 8,                       # batch size (根据显存调整)
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "workers": 4,                     # 数据加载线程
        "lr0": 0.001,                     # 初始学习率
        "lrf": 0.01,                      # 最终学习率系数
        "momentum": 0.937,                # SGD momentum
        "weight_decay": 0.0005,           # 权重衰减
        "warmup_epochs": 3,              # 预热轮数
        "warmup_momentum": 0.8,          # 预热momentum
        "warmup_bias_lr": 0.1,           # 预热bias学习率
        "box": 7.5,                       # box损失权重
        "cls": 0.5,                       # 分类损失权重
        "dfl": 1.5,                       # DFL损失权重
        "hsv_h": 0.015,                   # HSV-Hue增强
        "hsv_s": 0.7,                     # HSV-Saturation增强
        "hsv_v": 0.4,                     # HSV-Value增强
        "degrees": 0.0,                   # 旋转增强
        "translate": 0.1,                 # 平移增强
        "scale": 0.5,                     # 缩放增强
        "shear": 0.0,                     # 剪切增强
        "perspective": 0.0,               # 透视增强
        "flipud": 0.0,                    # 上下翻转
        "fliplr": 0.5,                    # 左右翻转
        "mosaic": 1.0,                    # Mosaic增强
        "mixup": 0.0,                     # MixUp增强 (VisDrone小目标多，关闭)
        "copy_paste": 0.0,               # Copy-Paste增强
        "patience": 20,                   # 早停耐心
        "save": True,                     # 保存模型
        "save_period": 10,               # 每10 epoch保存
        "val": True,                      # 验证
        "plots": True,                    # 生成图表
        "project": output_dir or "E:/UAVagent/output/training",
        "name": f"yolo11x_visdrone_{time.strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": True,
    }
    
    # 用户覆盖参数
    params.update(kwargs)
    
    print("=" * 60)
    print("YOLO11x VisDrone 微调训练")
    print("=" * 60)
    print(f"数据配置: {data_yaml}")
    print(f"基础模型: {params['model']}")
    print(f"训练轮数: {params['epochs']}")
    print(f"Batch: {params['batch']}")
    print(f"设备: {params['device']}")
    print(f"图像尺寸: {params['imgsz']}")
    print(f"输出目录: {os.path.join(params['project'], params['name'])}")
    
    # 加载模型
    model_path = params.pop('model')
    print(f"\n加载基础模型: {model_path}")
    model = YOLO(model_path)
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=params['epochs'],
        imgsz=params['imgsz'],
        batch=params['batch'],
        device=params['device'],
        workers=params['workers'],
        lr0=params['lr0'],
        lrf=params['lrf'],
        momentum=params['momentum'],
        weight_decay=params['weight_decay'],
        warmup_epochs=params['warmup_epochs'],
        warmup_momentum=params['warmup_momentum'],
        warmup_bias_lr=params['warmup_bias_lr'],
        box=params['box'],
        cls=params['cls'],
        dfl=params['dfl'],
        hsv_h=params['hsv_h'],
        hsv_s=params['hsv_s'],
        hsv_v=params['hsv_v'],
        degrees=params['degrees'],
        translate=params['translate'],
        scale=params['scale'],
        shear=params['shear'],
        perspective=params['perspective'],
        flipud=params['flipud'],
        fliplr=params['fliplr'],
        mosaic=params['mosaic'],
        mixup=params['mixup'],
        copy_paste=params['copy_paste'],
        patience=params['patience'],
        save=params['save'],
        save_period=params['save_period'],
        val=params['val'],
        plots=params['plots'],
        project=params['project'],
        name=params['name'],
        exist_ok=params['exist_ok'],
    )
    
    train_time = time.time() - start_time
    
    # 收集指标
    best_pt = os.path.join(params['project'], params['name'], 'weights', 'best.pt')
    last_pt = os.path.join(params['project'], params['name'], 'weights', 'last.pt')
    
    metrics = {
        "model": "yolo11x",
        "data": data_yaml,
        "epochs": params['epochs'],
        "batch": params['batch'],
        "imgsz": params['imgsz'],
        "train_time_hours": round(train_time / 3600, 2),
        "best_model": best_pt,
        "last_model": last_pt,
    }
    
    # 提取最终指标
    if hasattr(results, 'results_dict'):
        metrics.update({
            "mAP50": round(float(results.results_dict.get('metrics/mAP50(B)', 0)), 4),
            "mAP50_95": round(float(results.results_dict.get('metrics/mAP50-95(B)', 0)), 4),
            "precision": round(float(results.results_dict.get('metrics/precision(B)', 0)), 4),
            "recall": round(float(results.results_dict.get('metrics/recall(B)', 0)), 4),
        })
    
    # 保存训练指标
    metrics_path = os.path.join(params['project'], params['name'], 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 训练完成! 用时: {train_time/3600:.1f}小时")
    print(f"   最佳模型: {best_pt}")
    print(f"   mAP50: {metrics.get('mAP50', 'N/A')}")
    print(f"   mAP50-95: {metrics.get('mAP50_95', 'N/A')}")
    print(f"   指标文件: {metrics_path}")
    print(f"{'='*60}")
    
    return metrics, best_pt


def deploy_model(model_path: str, target_name: str = "yolo11x_visdrone.pt"):
    """将训练好的模型部署到 models/ 目录"""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    target_path = models_dir / target_name
    
    import shutil
    shutil.copy2(model_path, target_path)
    print(f"✅ 模型已部署: {target_path}")
    
    # 更新 MODEL_REGISTRY (在 settings.py 中添加新模型)
    print(f"\n📝 请手动将以下内容添加到 config/settings.py 的 MODEL_REGISTRY:")
    print(f'   "yolo11x_visdrone": str(BASE_DIR / "models" / "{target_name}"),')
    
    return str(target_path)


def main():
    parser = argparse.ArgumentParser(description="YOLO11x VisDrone 微调训练")
    parser.add_argument("--data", type=str, default="E:/datasets/VisDrone_YOLO/data.yaml",
                       help="data.yaml 路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--deploy", action="store_true", help="训练后自动部署到models/")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--resume", type=str, default=None, help="从checkpoint恢复")
    args = parser.parse_args()
    
    if not check_environment():
        return
    
    # 训练
    kwargs = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
    }
    if args.device:
        kwargs["device"] = args.device
    
    if args.resume:
        kwargs["resume"] = args.resume
    
    metrics, best_model = train_yolo11(args.data, args.output, **kwargs)
    
    # 部署
    if args.deploy and best_model and os.path.exists(best_model):
        deploy_model(best_model)


if __name__ == "__main__":
    main()
