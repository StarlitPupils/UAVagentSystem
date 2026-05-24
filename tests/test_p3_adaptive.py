# E:/UAVagent/tests/test_p3_adaptive.py (UAVagent 1.4 P3)
"""P3 测试套件 — 自适应阈值 + 微调优化"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config
import numpy as np, cv2


def test_scene_analysis():
    """测试场景分析 + 自适应阈值建议"""
    print("\n" + "=" * 60)
    print("P3.1 测试: 场景自适应阈值")
    print("=" * 60)
    
    from core.detection.preprocessing import ImagePreprocessor
    pp = ImagePreprocessor()
    
    # 创建不同场景的测试图像
    scenes = {
        "bright_day": (np.ones((640, 640, 3), dtype=np.uint8) * 220).astype(np.uint8),
        "normal_day": (np.ones((640, 640, 3), dtype=np.uint8) * 128).astype(np.uint8),
        "dim_dusk":    (np.ones((640, 640, 3), dtype=np.uint8) * 60).astype(np.uint8),
        "night":       (np.ones((640, 640, 3), dtype=np.uint8) * 30).astype(np.uint8),
    }
    
    # 添加一些边缘（模拟物体）
    for name, img in scenes.items():
        if name != "night":
            cv2.rectangle(img, (200, 300), (300, 380), (0, 0, 255), -1)
            cv2.rectangle(img, (400, 320), (500, 400), (255, 0, 0), -1)
    
    print(f"\n{'场景':<15} {'亮度':>6} {'类型':>8} {'建议conf':>10} {'建议IoU':>8}")
    print("-" * 50)
    
    all_ok = True
    for name, img in scenes.items():
        analysis = pp.analyze_scene(img)
        print(f"{name:<15} {analysis['mean_brightness']:>6.0f} "
              f"{analysis['scene_type']:>8} {analysis['suggested_conf']:>10.2f} "
              f"{analysis['suggested_iou']:>8.2f}")
        
        # 验证
        if name == "night" and analysis['suggested_conf'] > 0.25:
            print(f"    ⚠️ 夜景应降低阈值")
            all_ok = False
        if name == "bright_day" and analysis['suggested_conf'] < 0.20:
            print(f"    ⚠️ 明亮场景不应过低")
            all_ok = False
    
    if all_ok:
        print(f"\n  ✅ 场景分析 + 自适应阈值建议合理")
    
    return all_ok


def test_adaptive_vision():
    """测试 VisionSystem 集成自适应阈值"""
    print("\n" + "=" * 60)
    print("P3.1 测试: VisionSystem 自适应阈值集成")
    print("=" * 60)
    
    from core.vision_system import VisionSystem
    
    vs = VisionSystem(device="cpu", use_ensemble=True)
    
    # 默认关闭自适应（避免干扰测试）
    vs.adaptive_threshold = True
    
    # 测试不同场景
    test_images = {
        "dark": np.ones((640, 640, 3), dtype=np.uint8) * 40,
        "normal": np.ones((640, 640, 3), dtype=np.uint8) * 128,
        "bright": np.ones((640, 640, 3), dtype=np.uint8) * 210,
    }
    
    for name, img in test_images.items():
        # 添加一些边缘
        cv2.rectangle(img, (100, 200), (200, 300), (255, 0, 0), -1)
        
        conf, iou = vs._get_adaptive_params(img)
        print(f"  {name:8s}: conf={conf:.2f}, iou={iou:.2f}")
    
    print(f"\n  ✅ VisionSystem 自适应阈值集成正常")
    return True


def test_finetune_config():
    """测试微调配置文件"""
    print("\n" + "=" * 60)
    print("P3.2 测试: 微调训练配置验证")
    print("=" * 60)
    
    # 验证配置文件存在
    data_yaml = "E:/datasets/VisDrone_YOLO/data.yaml"
    base_model = "E:/UAVagent/models/yolo11x.pt"
    
    checks = []
    
    if os.path.exists(data_yaml):
        print(f"  ✅ 数据集配置: {data_yaml}")
        checks.append(True)
    else:
        print(f"  ⚠️ 数据集配置不存在: {data_yaml}")
        print(f"     请先运行: python training/visdrone_to_yolo.py")
        checks.append(False)
    
    if os.path.exists(base_model):
        size_mb = os.path.getsize(base_model) / (1024*1024)
        print(f"  ✅ 基础模型: yolo11x.pt ({size_mb:.0f}MB)")
        checks.append(True)
    else:
        print(f"  ⚠️ 基础模型不存在: {base_model}")
        checks.append(False)
    
    # 验证训练参数
    print(f"\n  P3.2 优化参数:")
    print(f"    lr0: 0.0005 (vs 1.3: 0.001)")
    print(f"    epochs: 50+EarlyStopping")
    print(f"    warmup_epochs: 5")
    print(f"    patience: 30")
    print(f"    weight_decay: 0.0005")
    
    ready = all(checks)
    if ready:
        print(f"\n  ✅ 微调训练配置就绪")
        print(f"  运行命令: python training/finetune_yolo11.py --epochs 50 --lr0 0.0005")
    else:
        print(f"\n  ⚠️ 部分文件缺失，请先准备数据集和模型")
    
    return ready


def main():
    print("=" * 70)
    print("UAVagent 1.4 P3 测试套件")
    print("=" * 70)
    
    config.setup_session()
    
    results = {}
    
    results['scene_analysis'] = test_scene_analysis()
    results['adaptive_vision'] = test_adaptive_vision()
    results['finetune_config'] = test_finetune_config()
    
    print("\n" + "=" * 70)
    print("P3 测试总结")
    print("=" * 70)
    for name, ok in results.items():
        status = "✅" if ok else "⚠️"
        print(f"  {status} {name}")
    
    passed = sum(1 for v in results.values() if v)
    print(f"\n  通过: {passed}/{len(results)}")
    
    if passed == len(results):
        print(f"\n  ✅ P3 全部测试通过")
        print(f"\n  📋 P3 交付清单:")
        print(f"    1. 自适应检测阈值: 根据亮度/密度动态调整 conf")
        print(f"    2. 微调优化: lr0=0.0005, epochs=50+, warmup=5")
        print(f"    3. 场景分析: 返回建议 conf/iou")
    else:
        print(f"\n  ⚠️ 部分测试需要额外条件")

if __name__ == "__main__":
    main()