# core/uav_controller.py
"""无人机控制器 - 所有方法显式返回 bool"""
import os

try:
    import cosysairsim as airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None

from config.settings import config


class UavController:
    def __init__(self):
        self.client = None
        self.connected = False
        self.mode = "simulation"

        sim_mode = getattr(config, 'SIMULATION_MODE', True)
        if sim_mode and AIRSIM_AVAILABLE:
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                self.connected = True
                self.mode = "airsim"
                print("[UAV] 已连接 AirSim")
            except Exception as e:
                print(f"[UAV] AirSim 连接失败: {e}，降级模拟")
                self.mode = "simulation"
        else:
            print("[UAV] 模拟模式")

    @property
    def is_connected(self) -> bool:
        return self.connected or self.mode == "simulation"

    def takeoff(self) -> bool:
        print("[UAV] 起飞")
        if self.mode == "airsim" and self.client:
            try:
                self.client.takeoffAsync().join()
            except Exception as e:
                print(f"[UAV] 起飞失败: {e}")
                return False
        return True

    def land(self) -> bool:
        print("[UAV] 降落")
        if self.mode == "airsim" and self.client:
            try:
                self.client.landAsync().join()
            except Exception as e:
                print(f"[UAV] 降落失败: {e}")
                return False
        return True

    def hover(self) -> bool:
        print("[UAV] 悬停")
        if self.mode == "airsim" and self.client:
            try:
                self.client.hoverAsync().join()
            except Exception as e:
                print(f"[UAV] 悬停失败: {e}")
                return False
        return True

    def move_to(self, x=0, y=0, z=-5, velocity=5) -> bool:
        print(f"[UAV] 移动到 ({x},{y},{z})")
        if self.mode == "airsim" and self.client:
            self.client.moveToPositionAsync(x, y, z, velocity).join()
        return True

    def get_position(self) -> dict:
        if self.mode == "airsim" and self.client:
            pose = self.client.simGetVehiclePose()
            return {"x": pose.position.x_val, "y": pose.position.y_val, "z": pose.position.z_val}
        return {"x": 0, "y": 0, "z": 0}
