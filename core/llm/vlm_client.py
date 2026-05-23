
# core/llm/vlm_client.py (UAVagent 1.3 Phase 3)
"""VLM 视觉推理客户端 — 支持 GPT-4V / Qwen-VL / LLaVA"""
import os, base64, json, time, cv2
import numpy as np
from typing import Optional, Dict, List
from config.settings import config


class VLMClient:
    """视觉语言模型统一客户端"""
    
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or 'deepseek'  # 默认复用 LLM provider
        self.model = model or 'qwen-vl-max'       # 可替换为 gpt-4-vision-preview
        self.api_key = config.LLM_API_KEY or os.getenv('DEEPSEEK_API_KEY', '')
        self.base_url = config.LLM_BASE_URL
        self.timeout = config.LLM_TIMEOUT
        self.enabled = config.VLM_ENABLED if hasattr(config, 'VLM_ENABLED') else True
    
    def encode_image(self, image: np.ndarray) -> str:
        """将图像编码为 base64 data URL"""
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    
    def analyze_scene(self, image: np.ndarray, prompt: str = None,
                      detections: List[Dict] = None) -> Dict:
        """分析场景，返回结构化理解"""
        if not self.enabled:
            return {'fallback': True, 'summary': 'VLM 未启用'}
        
        if prompt is None:
            prompt = (
                "你是一个无人机航拍分析专家。请分析这张图像："
                "1. 场景类型（城市/乡村/道路/水面等）"
                "2. 光照条件（明亮/阴暗/逆光）"
                "3. 是否存在遮挡或模糊区域"
                "4. 主要目标类别和大致数量"
                "5. 潜在的安全风险或异常情况"
            )
        
        # 构建带检测框的提示
        if detections:
            det_info = "检测到的目标：" + json.dumps([
                {
                    'class': d.get('class_name', d.get('class', '?')),
                    'conf': round(d.get('confidence', 0), 2),
                    'pos': [round(d['bbox'][0], 1), round(d['bbox'][1], 1)]
                }
                for d in detections[:20]
            ], ensure_ascii=False)
            prompt += f"\n\n{det_info}"
        
        try:
            response = self._call_api(image, prompt)
            return {'fallback': False, 'analysis': response}
        except Exception as e:
            print(f"[VLM] 调用失败: {e}")
            return {'fallback': True, 'error': str(e)}
    
    def _call_api(self, image: np.ndarray, prompt: str) -> str:
        """调用 VLM API"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=1,
        )
        
        image_data = self.encode_image(image)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        
        return response.choices[0].message.content or ""
    
    def detect_anomalies(self, image: np.ndarray, detections: List[Dict]) -> List[str]:
        """基于 VLM 的异常检测"""
        prompt = (
            "根据图像和检测结果，判断是否存在以下异常："
            "1. 检测框位置不合理（如物体在天空）"
            "2. 检测类别与环境矛盾（如沙漠中出现船）"
            "3. 存在危险场景（如人群聚集、道路事故）"
            "请以 JSON 数组返回异常描述，若无异常返回空数组 []。"
        )
        result = self.analyze_scene(image, prompt, detections)
        if result.get('fallback'):
            return []
        try:
            anomalies = json.loads(result['analysis'])
            return anomalies if isinstance(anomalies, list) else []
        except:
            return []


# 全局单例
vlm_client = VLMClient()
