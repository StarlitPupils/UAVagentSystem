# core/uav_controller.py (UAVagent 1.4 P2.1)
"""无人机控制器 — 多后端支持：模拟 / AirSim / MAVLink"""
import os, time
from typing import Optional, Dict

try:
    import cosysairsim as airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False

from config.settings import config


class UavController:
    """无人机控制器 — 统一接口
    
    支持三种后端:
    - "simulation" : 纯软件模拟 (默认)
    - "airsim"     : AirSim 模拟器
    - "mavlink"    : 真实 PX4/ArduPilot 飞控
    
    使用示例:
        uav = UavController(backend="mavlink", 
                           mavlink_conn="udp:127.0.0.1:14550")
        uav.connect()
        uav.takeoff(10.0)
        uav.hover()
        uav.land()
        uav.disconnect()
    """
    
    def __init__(self, backend: str = None, 
                 mavlink_conn: str = None):
        if backend is None:
            backend = getattr(config, 'UAV_BACKEND', 'simulation')
        self.backend = backend
        
        self.client = None       # AirSim client
        self.connected = False
        self._mavlink = None     # MAVLink connector
        
        if backend == "mavlink":
            self._init_mavlink(mavlink_conn)
        elif backend == "airsim" and AIRSIM_AVAILABLE:
            self._init_airsim()
        else:
            self.mode = "simulation"
            self.connected = True
            print("[UAV] 模拟模式")
    
    def _init_mavlink(self, connection_string: str = None):
        """初始化 MAVLink 后端"""
        try:
            from core.mavlink_connector import MavlinkConnector
            
            if connection_string is None:
                connection_string = getattr(config, 'MAVLINK_CONNECTION_STRING',
                                           'udp:127.0.0.1:14550')
            
            self._mavlink = MavlinkConnector(connection_string)
            self.mode = "mavlink"
            print(f"[UAV] MAVLink 模式 ({connection_string})")
        except ImportError as e:
            print(f"[UAV] MAVLink 初始化失败: {e}, 降级到模拟模式")
            self.mode = "simulation"
            self.connected = True
    
    def _init_airsim(self):
        """初始化 AirSim 后端"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.connected = True
            self.mode = "airsim"
            print("[UAV] AirSim 已连接")
        except Exception as e:
            print(f"[UAV] AirSim 连接失败: {e}, 降级模拟")
            self.mode = "simulation"
            self.connected = True
    
    # ==================== 连接管理 ====================
    
    def connect(self) -> bool:
        """建立连接"""
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.connect()
        return self.is_connected
    
    def disconnect(self):
        """断开连接"""
        if self.mode == "mavlink" and self._mavlink:
            self._mavlink.disconnect()
        self.connected = False
    
    @property
    def is_connected(self) -> bool:
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.is_connected
        return self.connected or self.mode == "simulation"
    
    # ==================== 飞行控制 ====================
    
    def takeoff(self, altitude: float = 10.0) -> bool:
        """起飞"""
        print(f"[UAV] 起飞 -> {altitude}m")
        
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.takeoff(altitude)
        elif self.mode == "airsim" and self.client:
            try:
                self.client.takeoffAsync().join()
                return True
            except Exception as e:
                print(f"[UAV] AirSim 起飞失败: {e}")
                return False
        return True  # 模拟模式
    
    def land(self) -> bool:
        """降落"""
        print("[UAV] 降落")
        
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.land()
        elif self.mode == "airsim" and self.client:
            try:
                self.client.landAsync().join()
                return True
            except:
                return False
        return True
    
    def hover(self) -> bool:
        """悬停"""
        print("[UAV] 悬停")
        
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.hover()
        elif self.mode == "airsim" and self.client:
            try:
                self.client.hoverAsync().join()
                return True
            except:
                return False
        return True
    
    def return_home(self) -> bool:
        """返航"""
        print("[UAV] 返航")
        
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.return_to_launch()
        return self.land()
    
    def arm(self) -> bool:
        """解锁"""
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.arm()
        return True
    
    def disarm(self) -> bool:
        """锁定"""
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.disarm()
        return True
    
    # ==================== 位置控制 ====================
    
    def goto(self, lat: float, lon: float, alt: float, 
             speed: float = 5.0) -> bool:
        """飞到指定GPS坐标"""
        print(f"[UAV] 飞行 -> ({lat:.4f}, {lon:.4f}, {alt}m)")
        
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.goto(lat, lon, alt, speed)
        return True
    
    def move_to(self, x: float = 0, y: float = 0, z: float = -5,
                velocity: float = 5) -> bool:
        """相对移动 (AirSim/模拟)"""
        print(f"[UAV] 移动 -> ({x}, {y}, {z})")
        
        if self.mode == "airsim" and self.client:
            try:
                self.client.moveToPositionAsync(x, y, z, velocity).join()
                return True
            except:
                return False
        return True
    
    def get_position(self) -> dict:
        """获取当前位置"""
        if self.mode == "mavlink" and self._mavlink:
            t = self._mavlink.telemetry
            return {"lat": t.lat, "lon": t.lon, "alt": t.alt_rel}
        elif self.mode == "airsim" and self.client:
            try:
                pose = self.client.simGetVehiclePose()
                return {
                    "x": pose.position.x_val,
                    "y": pose.position.y_val,
                    "z": pose.position.z_val,
                }
            except:
                pass
        return {"x": 0, "y": 0, "z": 0}
    
    # ==================== 状态查询 ====================
    
    def get_telemetry(self) -> dict:
        """获取遥测数据"""
        if self.mode == "mavlink" and self._mavlink:
            t = self._mavlink.telemetry
            return {
                "armed": t.armed,
                "mode": t.mode,
                "lat": t.lat,
                "lon": t.lon,
                "alt_rel": t.alt_rel,
                "battery": t.battery_remaining,
                "gps_fix": t.gps_fix,
            }
        return {"mode": "simulation", "armed": True, "battery": 100}
    
    def get_battery(self) -> float:
        """获取电池百分比"""
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.get_battery()
        return 100.0
    
    def get_gps_fix(self) -> int:
        """获取GPS定位状态"""
        if self.mode == "mavlink" and self._mavlink:
            return self._mavlink.get_gps_fix()
        return 3  # 模拟模式返回3D定位
    
    def get_mavlink_connector(self):
        """获取 MAVLink 连接器实例"""
        return self._mavlink