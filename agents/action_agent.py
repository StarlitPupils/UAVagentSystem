# agents/action_agent.py
"""行动智能体 - 字典驱动，全动作覆盖"""
from core.uav_controller import UavController
from core.data_logger import DataLogger

class ActionAgent:
    def __init__(self, uav: UavController, logger: DataLogger):
        self.uav = uav
        self.logger = logger

        # 动作映射表：标准动作名 -> (执行方法, 是否模拟支持)
        self.action_map = {
            "takeoff":    (self._do_takeoff,    True),
            "land":       (self._do_land,       True),
            "search":     (self._do_search,     True),
            "track":      (self._do_track,      True),
            "hover":      (self._do_hover,      True),
            "report":     (self._do_report,     True),
            "investigate":(self._do_search,     True),  # 别名
            "explore":    (self._do_search,     True),  # 别名
            "follow":     (self._do_track,      True),  # 别名
            "pursue":     (self._do_track,      True),  # 别名
            "return_home":(self._do_return_home,True),
            "emergency_stop": (self._do_emergency_stop, True),
        }

    def _do_takeoff(self):     return self.uav.takeoff()
    def _do_land(self):        return self.uav.land()
    def _do_hover(self):       return self.uav.hover()
    def _do_search(self):      return self.uav.hover()  # search/track 在模拟中即悬停
    def _do_track(self):       return self.uav.hover()
    def _do_report(self):      return True              # 报告无需硬件动作
    def _do_return_home(self): return self.uav.land()
    def _do_emergency_stop(self): return self.uav.hover()

    def execute(self, plan: dict) -> bool:
        action = plan.get("action_type", "hover").strip().lower()
        self.logger.log_event("action_start", {"action": action, "plan": plan})

        # 查找动作处理器
        handler, sim_supported = self.action_map.get(
            action, (None, False)
        )

        if handler is None:
            # 未知动作：模拟模式下尝试智能推断
            print(f"[行动智能体] 未知动作: {action}，尝试智能推断...")
            if any(kw in action for kw in ["search", "find", "detect"]):
                handler = self._do_search
            elif any(kw in action for kw in ["track", "follow"]):
                handler = self._do_track
            else:
                handler = self._do_hover

        # 执行
        try:
            success = handler()
            # 确保返回 bool
            if success is None:
                success = True
        except Exception as e:
            print(f"[行动智能体] 执行失败: {e}")
            success = False

        self.logger.log_event("action_done", {"action": action, "success": success})
        return bool(success)
