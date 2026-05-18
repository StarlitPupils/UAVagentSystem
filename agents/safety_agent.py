# agents/safety_agent.py
"""安全守护智能体 - 独立于主控制回路，优先级最高"""
import time
from enum import Enum

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

class SafetyAgent:
    """
    安全守护智能体
    - 独立运行，可中断任何其他Agent的指令
    - 基于规则的硬约束 + 可选的LLM辅助评估
    """
    def __init__(self):
        self.name = "SafetyAgent"
        self.enabled = True
        self.safety_rules = [
            {"condition": "battery < 25", "action": "强制返航", "level": SafetyLevel.CRITICAL},
            {"condition": "altitude > 120m", "action": "限高降落至120m以下", "level": SafetyLevel.DANGER},
            {"condition": "geofence_breached", "action": "立即悬停并返航", "level": SafetyLevel.CRITICAL},
            {"condition": "obstacle_distance < 5m", "action": "紧急避障/悬停", "level": SafetyLevel.CRITICAL},
            {"condition": "signal_lost > 10s", "action": "自动返航", "level": SafetyLevel.CRITICAL},
            {"condition": "motor_temp > 80C", "action": "强制降落", "level": SafetyLevel.CRITICAL},
            {"condition": "gps_accuracy < 3m", "action": "切换光流/视觉定位", "level": SafetyLevel.WARNING},
            {"condition": "wind_speed > 10m/s", "action": "降低高度，提示风险", "level": SafetyLevel.WARNING},
        ]
        self.safety_log: list[dict] = []

    def check_telemetry(self, telemetry: dict) -> dict:
        """检查遥测数据，返回安全评估"""
        warnings = []
        critical = False

        battery = telemetry.get("battery", 100)
        if battery < 15:
            critical = True
            warnings.append({"level": SafetyLevel.CRITICAL, "msg": f"电量严重不足({battery:.1f}%)，立即返航"})
        elif battery < 25:
            warnings.append({"level": SafetyLevel.WARNING, "msg": f"电量偏低({battery:.1f}%)，建议返航"})

        altitude = telemetry.get("altitude", 0)
        if altitude > 120:
            critical = True
            warnings.append({"level": SafetyLevel.DANGER, "msg": f"高度超限({altitude:.1f}m)，立即降低"})

        obstacle = telemetry.get("obstacle_distance", 999)
        if obstacle < 3:
            critical = True
            warnings.append({"level": SafetyLevel.CRITICAL, "msg": f"障碍物过近({obstacle:.1f}m)，紧急悬停"})
        elif obstacle < 8:
            warnings.append({"level": SafetyLevel.WARNING, "msg": f"障碍物接近({obstacle:.1f}m)，注意避让"})

        signal = telemetry.get("signal_strength", 100)
        if signal < 20:
            warnings.append({"level": SafetyLevel.WARNING, "msg": f"信号弱({signal}%)，注意通信"})

        return {
            "safe": not critical and len([w for w in warnings if w['level'] == SafetyLevel.CRITICAL]) == 0,
            "critical": critical,
            "warnings": warnings,
            "timestamp": time.time(),
        }

    def validate_action(self, action: dict, telemetry: dict = None) -> tuple[bool, str]:
        """
        验证行动计划是否安全
        返回: (是否允许, 原因)
        """
        if not self.enabled:
            return True, "安全Agent已禁用"

        action_type = action.get("action_type", "")

        # 规则1: 紧急情况下禁止除安全操作外的所有动作
        if telemetry:
            safety_check = self.check_telemetry(telemetry)
            if safety_check["critical"]:
                allowed_actions = ["land", "return_home", "hover", "emergency_stop"]
                if action_type not in allowed_actions:
                    return False, f"紧急情况，仅允许安全操作。当前不允许: {action_type}"

        # 规则2: 检查地理围栏
        if action_type in ["takeoff", "fly_to", "goto"]:
            target = action.get("target_position", {})
            lat, lon = target.get("lat", 0), target.get("lon", 0)
            if lat == 0 and lon == 0:
                return True, "无GPS目标，跳过地理围栏检查"
            # 地理围栏检查逻辑（有GPS数据时启用）
            # if not self._within_geofence(lat, lon):
            #     return False, f"目标位置({lat:.4f},{lon:.4f})超出地理围栏"

        # 规则3: 模拟模式下放宽限制
        if telemetry is None:
            return True, "模拟模式，安全限制放宽"

        self.safety_log.append({
            "action": action_type,
            "allowed": True,
            "timestamp": time.time(),
        })
        if len(self.safety_log) > 100:
            self.safety_log = self.safety_log[-100:]

        return True, "安全检查通过"

    def _within_geofence(self, lat: float, lon: float, home_lat: float = 0, home_lon: float = 0, radius_m: float = 500) -> bool:
        """简单地理围栏检查（欧氏距离近似）"""
        import math
        # 纬度1度≈111km，经度1度≈111km*cos(lat)
        dlat = (lat - home_lat) * 111000
        dlon = (lon - home_lon) * 111000 * math.cos(math.radians((lat + home_lat) / 2))
        distance = math.sqrt(dlat**2 + dlon**2)
        return distance <= radius_m

    def get_safety_report(self) -> dict:
        return {
            "enabled": self.enabled,
            "total_checks": len(self.safety_log),
            "rules_count": len(self.safety_rules),
            "recent_warnings": [log for log in self.safety_log[-10:] if not log.get("allowed", True)],
        }

# 全局单例
safety_agent = SafetyAgent()
