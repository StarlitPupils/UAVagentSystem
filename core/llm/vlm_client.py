# core/llm/vlm_client.py (UAVagent 1.4 P1 - v3 接入智谱 GLM-4V-Flash)
"""VLM 视觉推理客户端 — 国内免费模型：智谱 GLM-4V-Flash"""
import os, base64, json, time, cv2
import numpy as np
from typing import Optional, Dict, List
from config.settings import config


class VLMClient:
    """视觉语言模型统一客户端 (v1.4.3)
    
    优先级:
    1. 智谱 GLM-4V-Flash (国内免费，OpenAI兼容) ← 推荐
    2. Ollama + LLaVA (本地，免费)
    3. OpenAI GPT-4V (云端，付费)
    """
    
    # Provider 配置表
    PROVIDER_CONFIGS = {
        "zhipu": {
            "name": "智谱 GLM-4V-Flash",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4v-flash",
            "api_key_env": "ZHIPU_API_KEY",
            "free": True,
        },
        "ollama": {
            "name": "Ollama LLaVA",
            "base_url": "http://localhost:11434/v1",
            "model": "llava:7b",
            "api_key_env": None,
            "free": True,
        },
        "openai": {
            "name": "OpenAI GPT-4o",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
            "free": False,
        },
    }
    
    def __init__(self, provider: str = None, model: str = None):
        # 自动检测最佳 provider
        if provider is None:
            provider = self._detect_best_provider()
        
        self.provider = provider
        cfg = self.PROVIDER_CONFIGS.get(provider, self.PROVIDER_CONFIGS["zhipu"])
        
        if model is None:
            model = cfg["model"]
        self.model = model
        
        self.api_key = self._get_api_key(provider, cfg)
        self.base_url = cfg["base_url"]
        self.timeout = getattr(config, 'VLM_TIMEOUT', 60)
        self.enabled = bool(self.api_key)
        self.is_free = cfg.get("free", False)
        
        self.call_count = 0
        self.error_count = 0
        
        if self.enabled:
            print(f"[VLM] {cfg['name']} ({self.model}) {'免费' if self.is_free else '付费'}")
        else:
            self._print_how_to_enable(provider, cfg)
    
    def _detect_best_provider(self) -> str:
        """自动检测可用的 VLM provider"""
        # 1. 检查智谱 API Key
        if os.getenv("ZHIPU_API_KEY"):
            print("[VLM] 检测到智谱 API Key → 使用 GLM-4V-Flash (免费)")
            return "zhipu"
        
        # 2. 检查 Ollama
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                for m in models:
                    if "llava" in m.lower() or "minicpm" in m.lower():
                        print(f"[VLM] 检测到 Ollama 视觉模型: {m} → 使用本地 LLaVA")
                        return "ollama"
        except Exception:
            pass
        
        # 3. 检查 OpenAI
        if os.getenv("OPENAI_API_KEY"):
            print("[VLM] 检测到 OpenAI API Key → 使用 GPT-4o")
            return "openai"
        
        # 4. 默认推荐智谱（但需要配置 Key）
        print("[VLM] 未检测到视觉 API Key")
        return "zhipu"
    
    def _get_api_key(self, provider: str, cfg: dict) -> str:
        env_var = cfg.get("api_key_env")
        if env_var:
            return os.getenv(env_var, "")
        return "ollama"  # Ollama 不需要 key
    
    def _print_how_to_enable(self, provider: str, cfg: dict):
        """打印启用指南"""
        name = cfg.get("name", provider)
        print(f"[VLM] {name} 未配置")
        print(f"[VLM] 💡 获取免费 API Key (30秒完成):")
        
        if provider == "zhipu":
            print(f"[VLM]   1. 访问 https://bigmodel.cn 注册 (手机号)")
            print(f"[VLM]   2. 右上角 → API Keys → 创建新 Key")
            print(f"[VLM]   3. 复制 Key 到 .env: ZHIPU_API_KEY=你的key")
            print(f"[VLM]   ✅ GLM-4V-Flash 完全免费，Token 不限量")
        elif provider == "ollama":
            print(f"[VLM]   1. 安装: winget install Ollama.Ollama")
            print(f"[VLM]   2. 拉取: ollama pull llava:7b")
            print(f"[VLM]   3. 启动: ollama serve")
        elif provider == "openai":
            print(f"[VLM]   1. 访问 https://platform.openai.com")
            print(f"[VLM]   2. 创建 API Key")
            print(f"[VLM]   3. 设置环境变量 OPENAI_API_KEY")
    
    def encode_image(self, image: np.ndarray, quality: int = 85) -> str:
        """将图像编码为 base64 data URL"""
        if image.shape[-1] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image
        
        # 限制最大尺寸
        h, w = bgr.shape[:2]
        max_dim = 2048
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    
    def analyze_scene(self, image: np.ndarray, prompt: str = None,
                      detections: List[Dict] = None) -> Dict:
        """分析场景"""
        if not self.enabled:
            return {'fallback': True, 'reason': 'VLM 未配置 API Key'}
        
        if prompt is None:
            prompt = "你是一个无人机航拍分析专家。请分析这张图像：场景类型、光照条件、主要目标、安全风险。"
        
        if detections:
            det_info = self._format_detections(detections)
            prompt += f"\n\n[检测器提供的目标]\n{det_info}"
        
        t0 = time.perf_counter()
        try:
            analysis = self._call_api(image, prompt)
            latency = (time.perf_counter() - t0) * 1000
            self.call_count += 1
            return {'fallback': False, 'analysis': analysis, 'latency_ms': round(latency, 1),
                    'provider': self.provider, 'model': self.model}
        except Exception as e:
            self.error_count += 1
            print(f"[VLM] 调用失败: {e}")
            return {'fallback': True, 'error': str(e)}
    
    def _format_detections(self, detections, max_dets=20):
        if not detections:
            return ""
        lines = []
        for d in detections[:max_dets]:
            bbox = d.get('bbox', [0,0,0,0])
            lines.append(f"  {d.get('class_name','obj')} at ({bbox[0]:.0f},{bbox[1]:.0f}) "
                        f"conf={d.get('confidence',0):.2f}")
        return '\n'.join(lines)
    
    def _call_api(self, image, prompt):
        """统一调用 API（智谱/Ollama/OpenAI 均兼容 OpenAI 协议）"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        
        image_url = self.encode_image(image)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            max_tokens=1024,
            temperature=0.3,
        )
        
        return response.choices[0].message.content or ""
    
    def detect_anomalies(self, image, detections):
        """异常检测"""
        if not self.enabled:
            return []
        prompt = ("检查以下检测是否存在异常（如车辆在天空、类别与环境矛盾）。"
                  "以 JSON 数组返回异常描述，无异常返回 []。")
        result = self.analyze_scene(image, prompt, detections)
        if result.get('fallback'):
            return []
        try:
            import re
            match = re.search(r'\[.*\]', result['analysis'], re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []
    
    def get_stats(self):
        return {
            "provider": self.provider, "model": self.model, "enabled": self.enabled,
            "free": self.is_free, "call_count": self.call_count,
            "error_count": self.error_count,
        }


vlm_client = VLMClient()