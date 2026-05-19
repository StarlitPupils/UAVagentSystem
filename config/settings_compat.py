# 追加到 E:\UAVagent\config\settings.py 的 Config 类中

    # ========== 仿真模式 ==========
    SIMULATION_MODE: bool = os.getenv("SIMULATION_MODE", "true").lower() == "true"
    AIRSIM_ENABLED: bool = False  # 无硬件时默认关闭
