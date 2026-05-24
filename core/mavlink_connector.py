# core/mavlink_connector.py (UAVagent 1.4 P2.1)
"""MAVLink 飞控连接器 — 支持 PX4/ArduPilot 真实飞控通信
协议: MAVLink v2 over UDP/Serial
支持: 心跳、姿态、GPS、电池、飞行模式、位置指令
"""
import time, threading, math
from typing import Optional, Dict, Callable
from enum import Enum
from dataclasses import dataclass, field


class MavlinkConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"


@dataclass
class TelemetryData:
    """无人机遥测数据"""
    armed: bool = False
    mode: str = "STABILIZE"
    
    # GPS
    lat: float = 0.0
    lon: float = 0.0
    alt_msl: float = 0.0       # 海拔高度(m)
    alt_rel: float = 0.0       # 相对起飞点高度(m)
    gps_fix: int = 0           # 0=无, 1=2D, 2=3D
    gps_satellites: int = 0
    
    # 姿态
    roll: float = 0.0          # rad
    pitch: float = 0.0         # rad
    yaw: float = 0.0           # rad (0=北, pi/2=东)
    
    # 速度 (NED坐标系, m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # 电池
    battery_voltage: float = 0.0
    battery_current: float = 0.0
    battery_remaining: float = 100.0  # 0-100%
    
    # 系统状态
    cpu_load: float = 0.0
    drop_rate: float = 0.0


class MavlinkConnector:
    """MAVLink 飞控连接器
    
    支持两种连接方式:
    - UDP: mavlink_conn = MavlinkConnector("udp:127.0.0.1:14550")
    - Serial: mavlink_conn = MavlinkConnector("serial:COM3:57600")
    - TCP: mavlink_conn = MavlinkConnector("tcp:127.0.0.1:5760")
    
    使用示例:
        conn = MavlinkConnector("udp:127.0.0.1:14550")
        conn.connect()
        conn.wait_for_heartbeat(timeout=10)
        
        telem = conn.get_telemetry()
        print(f"GPS: {telem.lat}, {telem.lon}, alt={telem.alt_rel}m")
        
        conn.arm()
        conn.takeoff(10.0)  # 起飞到10米
        conn.goto(47.3977, 8.5456, 20.0)  # 飞到指定GPS坐标
        conn.land()
        conn.disconnect()
    """
    
    # MAVLink 消息ID常量
    MAVLINK_MSG_ID_HEARTBEAT = 0
    MAVLINK_MSG_ID_GLOBAL_POSITION_INT = 33
    MAVLINK_MSG_ID_ATTITUDE = 30
    MAVLINK_MSG_ID_BATTERY_STATUS = 147
    MAVLINK_MSG_ID_SYS_STATUS = 1
    MAVLINK_MSG_ID_COMMAND_ACK = 77
    
    # MAVLink 命令ID
    MAV_CMD_NAV_TAKEOFF = 22
    MAV_CMD_NAV_LAND = 21
    MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
    MAV_CMD_DO_SET_MODE = 176
    MAV_CMD_COMPONENT_ARM_DISARM = 400
    
    MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
    MAV_MODE_FLAG_SAFETY_ARMED = 128
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550",
                 system_id: int = 255, component_id: int = 1,
                 source_system: int = 1):
        self.connection_string = connection_string
        self.system_id = system_id        # 飞控系统ID
        self.component_id = component_id  # 飞控组件ID
        self.source_system = source_system
        
        self._state = MavlinkConnectionState.DISCONNECTED
        self._telemetry = TelemetryData()
        self._last_heartbeat = 0.0
        self._heartbeat_timeout = 5.0     # 心跳超时秒数
        
        self._master = None
        self._thread = None
        self._stop_event = threading.Event()
        
        # 回调函数
        self.on_telemetry_update: Optional[Callable] = None
        self.on_heartbeat: Optional[Callable] = None
        self.on_connection_lost: Optional[Callable] = None
        
        # 实际pymavlink可用性
        self._pymavlink_available = False
        try:
            import pymavlink.mavlinkv20 as mavlink2
            import pymavlink.dialects.v20.ardupilotmega as mavlink_dialect
            self._mavlink = mavlink_dialect
            self._pymavlink_available = True
        except ImportError:
            print("[MAVLink] pymavlink 未安装，使用模拟模式")
            print("[MAVLink] 安装: pip install pymavlink")
    
    # ==================== 连接管理 ====================
    
    def connect(self, blocking: bool = False, timeout: float = 10.0) -> bool:
        """建立 MAVLink 连接"""
        if self._pymavlink_available and not self._is_simulation():
            return self._connect_real(blocking, timeout)
        else:
            return self._connect_simulated()
    
    def _is_simulation(self) -> bool:
        """判断是否为模拟连接"""
        return "sim" in self.connection_string.lower() or self.connection_string.startswith("mock")
    
    def _connect_real(self, blocking: bool, timeout: float) -> bool:
        """真实 MAVLink 连接"""
        try:
            from pymavlink import mavutil
            
            self._state = MavlinkConnectionState.CONNECTING
            print(f"[MAVLink] 连接 {self.connection_string}...")
            
            self._master = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self.source_system,
            )
            
            if blocking:
                self._state = MavlinkConnectionState.CONNECTED
                self._start_reader_thread()
                print(f"[MAVLink] 已连接")
                return True
            else:
                # 等待心跳
                print("[MAVLink] 等待心跳...")
                msg = self._master.recv_match(
                    type='HEARTBEAT', blocking=True, timeout=timeout
                )
                if msg:
                    self._state = MavlinkConnectionState.CONNECTED
                    self._last_heartbeat = time.time()
                    self._start_reader_thread()
                    print(f"[MAVLink] 心跳收到, 已连接 (type={msg.type}, autopilot={msg.autopilot})")
                    return True
                else:
                    self._state = MavlinkConnectionState.HEARTBEAT_TIMEOUT
                    print(f"[MAVLink] 心跳超时 ({timeout}s)")
                    return False
        
        except Exception as e:
            self._state = MavlinkConnectionState.DISCONNECTED
            print(f"[MAVLink] 连接失败: {e}")
            return False
    
    def _connect_simulated(self) -> bool:
        """模拟连接（开发测试用）"""
        self._state = MavlinkConnectionState.CONNECTED
        self._last_heartbeat = time.time()
        print(f"[MAVLink] 模拟模式已连接 (connection_string={self.connection_string})")
        
        # 初始化模拟遥测数据
        self._telemetry = TelemetryData(
            armed=False,
            mode="GUIDED",
            lat=47.3977,
            lon=8.5456,
            alt_rel=0.0,
            gps_fix=3,
            gps_satellites=12,
            battery_remaining=85.0,
        )
        
        # 启动模拟遥测线程
        self._start_simulated_telemetry_thread()
        return True
    
    def disconnect(self):
        """断开连接"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        
        if self._master:
            try:
                self._master.close()
            except:
                pass
        
        self._state = MavlinkConnectionState.DISCONNECTED
        print("[MAVLink] 已断开")
    
    def wait_for_heartbeat(self, timeout: float = 10.0) -> bool:
        """等待心跳信号"""
        if self._state == MavlinkConnectionState.CONNECTED:
            # 已经是连接状态，检查心跳是否过期
            if time.time() - self._last_heartbeat < self._heartbeat_timeout:
                return True
        
        start = time.time()
        while time.time() - start < timeout:
            if self._state == MavlinkConnectionState.CONNECTED:
                return True
            time.sleep(0.1)
        
        return False
    
    # ==================== 遥测数据 ====================
    
    @property
    def telemetry(self) -> TelemetryData:
        return self._telemetry
    
    def get_telemetry(self) -> TelemetryData:
        """获取最新遥测数据"""
        return self._telemetry
    
    def is_armed(self) -> bool:
        return self._telemetry.armed
    
    def get_position(self) -> tuple:
        """获取当前位置 (lat, lon, alt_rel)"""
        t = self._telemetry
        return (t.lat, t.lon, t.alt_rel)
    
    def get_battery(self) -> float:
        """获取电池剩余百分比"""
        return self._telemetry.battery_remaining
    
    def get_gps_fix(self) -> int:
        """获取GPS定位状态 (0=无, 1=2D, 2=3D)"""
        return self._telemetry.gps_fix
    
    # ==================== 飞行控制 ====================
    
    def arm(self, force: bool = False) -> bool:
        """解锁飞控"""
        if self._telemetry.armed:
            print("[MAVLink] 已解锁")
            return True
        
        print("[MAVLink] 发送解锁指令...")
        
        if not self._pymavlink_available or self._is_simulation():
            self._telemetry.armed = True
            print("[MAVLink] 模拟解锁成功")
            return True
        
        try:
            self._master.mav.command_long_send(
                self.system_id,
                self.component_id,
                self.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1.0 if not force else 21196.0,  # 1=正常解锁, 21196=强制解锁
                0, 0, 0, 0, 0, 0
            )
            
            # 等待确认
            ack = self._master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
            if ack and ack.result == 0:  # MAV_RESULT_ACCEPTED
                self._telemetry.armed = True
                print("[MAVLink] 解锁成功")
                return True
            else:
                print(f"[MAVLink] 解锁失败: {ack.result if ack else 'no response'}")
                return False
        
        except Exception as e:
            print(f"[MAVLink] 解锁异常: {e}")
            return False
    
    def disarm(self) -> bool:
        """锁定飞控"""
        if not self._telemetry.armed:
            print("[MAVLink] 已锁定")
            return True
        
        if not self._pymavlink_available or self._is_simulation():
            self._telemetry.armed = False
            print("[MAVLink] 模拟锁定成功")
            return True
        
        try:
            self._master.mav.command_long_send(
                self.system_id, self.component_id,
                self.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                0.0, 0, 0, 0, 0, 0, 0
            )
            self._telemetry.armed = False
            print("[MAVLink] 锁定成功")
            return True
        except Exception as e:
            print(f"[MAVLink] 锁定异常: {e}")
            return False
    
    def takeoff(self, target_altitude: float = 10.0) -> bool:
        """起飞到指定高度(m)"""
        if self._is_simulation():
            print(f"[MAVLink] 模拟起飞 -> {target_altitude}m")
            self._telemetry.armed = True
            self._telemetry.alt_rel = target_altitude
            return True
        
        if not self._telemetry.armed:
            print("[MAVLink] 未解锁,无法起飞")
            return False
        
        try:
            self._master.mav.command_long_send(
                self.system_id, self.component_id,
                self.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, target_altitude
            )
            print(f"[MAVLink] 起飞指令: {target_altitude}m")
            return True
        except Exception as e:
            print(f"[MAVLink] 起飞异常: {e}")
            return False
    
    def land(self) -> bool:
        """降落"""
        if self._is_simulation():
            print("[MAVLink] 模拟降落")
            self._telemetry.alt_rel = 0.0
            return True
        
        try:
            self._master.mav.command_long_send(
                self.system_id, self.component_id,
                self.MAV_CMD_NAV_LAND, 0,
                0, 0, 0, 0, 0, 0, 0
            )
            print("[MAVLink] 降落指令已发送")
            return True
        except Exception as e:
            print(f"[MAVLink] 降落异常: {e}")
            return False
    
    def return_to_launch(self) -> bool:
        """返航"""
        if self._is_simulation():
            print("[MAVLink] 模拟返航")
            self._telemetry.lat = 47.3977
            self._telemetry.lon = 8.5456
            self._telemetry.alt_rel = 0.0
            return True
        
        try:
            self._master.mav.command_long_send(
                self.system_id, self.component_id,
                self.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,
                0, 0, 0, 0, 0, 0, 0
            )
            print("[MAVLink] 返航指令已发送")
            return True
        except Exception as e:
            print(f"[MAVLink] 返航异常: {e}")
            return False
    
    def goto(self, lat: float, lon: float, alt: float, 
             ground_speed: float = 5.0) -> bool:
        """飞到指定GPS坐标"""
        if self._is_simulation():
            print(f"[MAVLink] 模拟飞行 -> ({lat:.4f}, {lon:.4f}, {alt}m)")
            self._telemetry.lat = lat
            self._telemetry.lon = lon
            self._telemetry.alt_rel = alt
            return True
        
        try:
            self._master.mav.set_position_target_global_int_send(
                0,  # 时间戳
                self.system_id,
                self.component_id,
                6,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
                0b0000111111111000,  # 仅位置
                int(lat * 1e7),
                int(lon * 1e7),
                alt,
                0, 0, 0,  # 速度 NED
                0, 0, 0,  # 加速度
                0, 0      # yaw, yaw_rate
            )
            print(f"[MAVLink] 飞行指令: ({lat:.4f}, {lon:.4f}, {alt}m)")
            return True
        except Exception as e:
            print(f"[MAVLink] GPS飞行异常: {e}")
            return False
    
    def hover(self) -> bool:
        """悬停（当前位置）"""
        t = self._telemetry
        return self.goto(t.lat, t.lon, t.alt_rel)
    
    def set_mode(self, mode: str) -> bool:
        """切换飞行模式"""
        valid_modes = {
            "GUIDED", "AUTO", "LOITER", "RTL", "LAND",
            "STABILIZE", "ALT_HOLD", "POSHOLD",
        }
        if mode.upper() not in valid_modes:
            print(f"[MAVLink] 未知模式: {mode}, 有效模式: {valid_modes}")
            return False
        
        if self._is_simulation():
            self._telemetry.mode = mode.upper()
            print(f"[MAVLink] 模拟切换模式 -> {mode}")
            return True
        
        try:
            mode_id = self._master.mode_mapping().get(mode.upper())
            if mode_id is None:
                print(f"[MAVLink] 模式映射失败: {mode}")
                return False
            
            self._master.mav.command_long_send(
                self.system_id, self.component_id,
                self.MAV_CMD_DO_SET_MODE, 0,
                self.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id, 0, 0, 0, 0, 0
            )
            self._telemetry.mode = mode.upper()
            print(f"[MAVLink] 模式切换: {mode}")
            return True
        except Exception as e:
            print(f"[MAVLink] 模式切换异常: {e}")
            return False
    
    # ==================== 线程管理 ====================
    
    def _start_reader_thread(self):
        """启动 MAVLink 消息读取线程"""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="mavlink-reader"
        )
        self._thread.start()
    
    def _reader_loop(self):
        """MAVLink 消息读取循环"""
        while not self._stop_event.is_set():
            try:
                if self._master is None:
                    break
                
                msg = self._master.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    # 检查心跳超时
                    if time.time() - self._last_heartbeat > self._heartbeat_timeout:
                        if self._state == MavlinkConnectionState.CONNECTED:
                            self._state = MavlinkConnectionState.HEARTBEAT_TIMEOUT
                            print("[MAVLink] 心跳超时，连接丢失")
                            if self.on_connection_lost:
                                self.on_connection_lost()
                    continue
                
                self._process_message(msg)
                
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"[MAVLink] 读取错误: {e}")
                break
    
    def _process_message(self, msg):
        """处理 MAVLink 消息"""
        msg_type = msg.get_type()
        
        if msg_type == "HEARTBEAT":
            self._last_heartbeat = time.time()
            self._telemetry.mode = self._decode_flight_mode(
                msg.custom_mode, msg.base_mode
            )
            if self.on_heartbeat:
                self.on_heartbeat()
        
        elif msg_type == "GLOBAL_POSITION_INT":
            self._telemetry.lat = msg.lat / 1e7
            self._telemetry.lon = msg.lon / 1e7
            self._telemetry.alt_msl = msg.alt / 1000.0
            self._telemetry.alt_rel = msg.relative_alt / 1000.0
        
        elif msg_type == "ATTITUDE":
            self._telemetry.roll = msg.roll
            self._telemetry.pitch = msg.pitch
            self._telemetry.yaw = msg.yaw
        
        elif msg_type == "BATTERY_STATUS":
            if msg.battery_function == 0:  # 动力电池
                self._telemetry.battery_voltage = msg.voltages[0] / 1000.0 if msg.voltages else 0
                self._telemetry.battery_remaining = msg.battery_remaining
        
        elif msg_type == "SYS_STATUS":
            self._telemetry.cpu_load = msg.load / 10.0
            self._telemetry.drop_rate = msg.drop_rate_comm / 100.0
        
        if self.on_telemetry_update:
            self.on_telemetry_update(self._telemetry)
    
    def _start_simulated_telemetry_thread(self):
        """启动模拟遥测线程（开发测试用）"""
        try:
            import pymavlink
            has_pymavlink = True
        except ImportError:
            has_pymavlink = False
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._simulated_telemetry_loop, daemon=True, name="mavlink-sim"
        )
        self._thread.start()
    
    def _simulated_telemetry_loop(self):
        """模拟遥测数据更新"""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            
            if self._telemetry.armed and self._telemetry.alt_rel < 10.0:
                self._telemetry.alt_rel += 0.5
            
            self._last_heartbeat = time.time()
            self._telemetry.cpu_load = min(100, self._telemetry.cpu_load + 2)
    
    def _decode_flight_mode(self, custom_mode: int, base_mode: int) -> str:
        """解码PX4飞行模式"""
        # PX4模式映射 (简化版)
        mode_map = {
            0: "MANUAL", 1: "ALTCTL", 2: "POSCTL",
            3: "AUTO", 4: "ACRO", 5: "OFFBOARD",
            6: "STABILIZED", 7: "RATTITUDE", 8: "SIMPLE",
            9: "GUIDED",
        }
        return mode_map.get(custom_mode & 0xFF, f"MODE_{custom_mode}")
    
    # ==================== 状态查询 ====================
    
    @property
    def is_connected(self) -> bool:
        return self._state == MavlinkConnectionState.CONNECTED
    
    @property
    def connection_state(self) -> MavlinkConnectionState:
        return self._state
    
    def get_connection_info(self) -> dict:
        return {
            "connection_string": self.connection_string,
            "state": self._state.value,
            "is_connected": self.is_connected,
            "last_heartbeat_ago": round(time.time() - self._last_heartbeat, 1),
            "pymavlink_available": self._pymavlink_available,
        }


# ==================== 便捷函数 ====================

def create_mavlink_connector(connection_string: str = None) -> MavlinkConnector:
    """工厂函数：根据配置创建 MAVLink 连接器"""
    from config.settings import config
    
    if connection_string is None:
        connection_string = getattr(config, 'MAVLINK_CONNECTION_STRING', 
                                    'udp:127.0.0.1:14550')
    
    return MavlinkConnector(connection_string)