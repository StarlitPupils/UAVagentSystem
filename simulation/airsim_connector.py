import time
import numpy as np
from typing import Optional
from config.settings import config
try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None

class AirSimConnector:
    def __init__(self):
        self.client = None
        self.is_connected = False
        if config.SIMULATION_MODE and AIRSIM_AVAILABLE:
            self._connect()
    def _connect(self):
        try:
            self.client = airsim.MultirotorClient(ip=config.AIRSIM_IP, port=config.AIRSIM_PORT)
            self.client.confirmConnection()
            self.is_connected = True
            print(f"[AirSim] 已连接到 {config.AIRSIM_IP}:{config.AIRSIM_PORT}")
        except Exception as e:
            print(f"[AirSim] 连接失败: {e}")
    def get_image(self, camera_name: str = "0") -> Optional[np.ndarray]:
        if not self.is_connected: return None
        try:
            responses = self.client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            return img1d.reshape(response.height, response.width, 3)
        except:
            return None