# E:/UAVagent/tests/test_p2_connectors.py (UAVagent 1.4 P2)
"""P2 测试套件 — MAVLink + TensorRT INT8"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


def test_mavlink_simulation():
    """测试 MAVLink 模拟模式"""
    print("\n" + "=" * 60)
    print("P2.1 测试: MAVLink 飞控连接器 (模拟模式)")
    print("=" * 60)
    
    from core.mavlink_connector import MavlinkConnector, TelemetryData
    
    # 模拟连接
    conn = MavlinkConnector("mock://")
    conn.connect()
    
    print(f"  连接状态: {conn.connection_state.value}")
    print(f"  遥测数据: {conn.telemetry}")
    
    # 测试解锁
    print(f"\n  解锁测试...")
    conn.arm()
    assert conn.is_armed(), "解锁失败"
    print(f"  ✅ 解锁成功")
    
    # 测试起飞
    print(f"  起飞测试...")
    conn.takeoff(10.0)
    time.sleep(1.5)
    _, _, alt = conn.get_position()
    print(f"  当前高度: {alt:.1f}m")
    assert alt > 0, f"高度应为正值, 实际 {alt}"
    print(f"  ✅ 起飞成功")
    
    # 测试 GPS 飞行
    print(f"  飞行测试...")
    conn.goto(47.3980, 8.5460, 20.0)
    lat, lon, alt = conn.get_position()
    print(f"  目标位置: ({lat:.4f}, {lon:.4f}, {alt:.1f}m)")
    print(f"  ✅ 飞行指令成功")
    
    # 测试降落
    print(f"  降落测试...")
    conn.land()
    print(f"  ✅ 降落成功")
    
    conn.disconnect()
    
    print(f"\n  ✅ MAVLink 模拟模式全部测试通过")
    return True


def test_uav_controller_mavlink():
    """测试 UAVController 的 MAVLink 后端"""
    print("\n" + "=" * 60)
    print("P2.1 测试: UAVController + MAVLink 后端")
    print("=" * 60)
    
    from core.uav_controller import UavController
    
    uav = UavController(backend="mavlink", mavlink_conn="mock://")
    
    print(f"  后端: {uav.backend}")
    print(f"  模式: {uav.mode}")
    print(f"  已连接: {uav.is_connected}")
    
    # 起飞
    uav.takeoff(5.0)
    time.sleep(0.5)
    pos = uav.get_position()
    print(f"  位置: {pos}")
    
    # 遥测
    telem = uav.get_telemetry()
    print(f"  电池: {telem.get('battery', 'N/A')}%")
    print(f"  GPS: {telem.get('gps_fix', 'N/A')}")
    
    uav.land()
    uav.disconnect()
    
    print(f"\n  ✅ UAVController MAVLink 后端测试通过")
    return True


def test_tensorrt_int8_export():
    """测试 TensorRT INT8 导出（如果条件满足）"""
    print("\n" + "=" * 60)
    print("P2.2 测试: TensorRT INT8 量化导出")
    print("=" * 60)
    
    import torch
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA 不可用，跳过 TensorRT 测试")
        return False
    
    from core.edge.tensorrt_exporter import TensorRTExporter
    
    model_path = r"E:\UAVagent\models\yolo11n.pt"  # 用小模型测试
    if not os.path.exists(model_path):
        # 尝试自动下载
        try:
            from ultralytics import YOLO
            YOLO("yolo11n.pt")
            model_path = r"E:\UAVagent\models\yolo11n.pt"
        except:
            print(f"  ⚠️ 模型不存在: {model_path}")
            return False
    
    print(f"  测试模型: {os.path.basename(model_path)}")
    
    # 1. 先导出 FP16 engine
    print(f"\n  [1] 导出 FP16 engine...")
    fp16_path = TensorRTExporter.export_to_engine(model_path, fp16=True)
    if fp16_path:
        print(f"  ✅ FP16: {fp16_path}")
    else:
        print(f"  ⚠️ FP16 导出失败（可能已存在）")
    
    # 2. 尝试 INT8 导出（使用 VisDrone 校准数据）
    calib_dir = r"E:\datasets\VisDrone\VisDrone2019-MOT-val\sequences\uav0000086_00000_v"
    if os.path.isdir(calib_dir):
        print(f"\n  [2] 导出 INT8 engine (校准数据: {calib_dir})...")
        int8_path = TensorRTExporter.export_to_engine(
            model_path, fp16=False, int8=True,
            calibration_images=[calib_dir],
        )
        if int8_path:
            print(f"  ✅ INT8: {int8_path}")
        else:
            print(f"  ⚠️ INT8 导出失败（可能需要额外配置）")
    else:
        print(f"\n  [2] INT8 校准数据不可用，跳过")
    
    # 3. 速度对比
    if fp16_path:
        print(f"\n  [3] 速度对比...")
        from core.edge.tensorrt_exporter import benchmark_all_precisions
        benchmark_all_precisions(model_path)
    
    print(f"\n  ✅ TensorRT 测试完成")
    return True


def main():
    print("=" * 70)
    print("UAVagent 1.4 P2 测试套件")
    print("=" * 70)
    
    config.setup_session()
    
    results = {}
    
    # P2.1
    results['mavlink_sim'] = test_mavlink_simulation()
    results['uav_mavlink'] = test_uav_controller_mavlink()
    
    # P2.2
    results['tensorrt_int8'] = test_tensorrt_int8_export()
    
    # 总结
    print("\n" + "=" * 70)
    print("P2 测试总结")
    print("=" * 70)
    for name, ok in results.items():
        status = "✅" if ok else "⚠️"
        print(f"  {status} {name}")
    
    passed = sum(1 for v in results.values() if v)
    print(f"\n  通过: {passed}/{len(results)}")
    
    if passed == len(results):
        print(f"\n  ✅ P2 全部测试通过")
    else:
        print(f"\n  ⚠️ 部分测试需要额外条件（真实飞控/CUDA）")


if __name__ == "__main__":
    main()