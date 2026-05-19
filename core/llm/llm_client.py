# core/llm/llm_client.py (v1.2)
"""LLM客户端 v2 - DeepSeek + Ollama本地模型 + 路由"""
import os
import json
import time
import hashlib
from datetime import datetime
from config.settings import config


class LLMClient:
    def __init__(self):
        # DeepSeek配置
        self.provider = config.LLM_PROVIDER
        self.model = config.LLM_MODEL
        self.api_key = config.LLM_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = config.LLM_BASE_URL
        self.max_tokens = config.LLM_MAX_TOKENS
        self.temperature = config.LLM_TEMPERATURE
        self.timeout = config.LLM_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES
        
        # Ollama配置
        self.ollama_enabled = config.OLLAMA_ENABLED
        self.ollama_url = config.OLLAMA_BASE_URL
        self.ollama_model = config.OLLAMA_MODEL
        self.ollama_vision_model = config.OLLAMA_VISION_MODEL
        
        # 缓存
        self.response_cache: dict = {}
        
        # 统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        self.ollama_calls = 0
        self.deepseek_calls = 0
        
        # 上一次调用详情
        self.last_call_details: dict = {
            "success": False, "model": "", "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_content": "",
            "ttft": 0.0, "error_type": "", "latency_ms": 0,
            "provider": "",  # v1.2: 记录使用的provider
        }
    
    # ========== 1.0兼容接口 ==========
    
    def get_last_call_details(self) -> dict:
        return self.last_call_details
    
    async def generate(self, prompt: str, timeout_override: int = None) -> str:
        """异步生成 (meta_agent/reflection_agent使用)"""
        timeout = timeout_override or self.timeout
        start_time = time.time()
        
        self.last_call_details = {
            "success": False, "model": self.model, "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_content": "",
            "ttft": 0.0, "error_type": "", "latency_ms": 0,
            "provider": "",
        }
        
        # v1.2: 路由决策
        provider = self._route_provider(prompt)
        
        try:
            if provider == "ollama":
                content = self._call_ollama(prompt, timeout)
                self.last_call_details["provider"] = "ollama"
            else:
                content = self._call_deepseek(prompt, timeout)
                self.last_call_details["provider"] = "deepseek"
            
            elapsed = (time.time() - start_time) * 1000
            self.last_call_details.update({
                "success": True, "latency_ms": elapsed, "ttft": elapsed,
            })
            
            # 缓存
            cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            self.response_cache[cache_key] = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
            }
            
            return content
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.last_call_details.update({
                "success": False, "error_type": type(e).__name__,
                "latency_ms": elapsed,
            })
            print(f"[LLM] generate失败 ({provider}): {e}")
            
            # 降级: 尝试另一provider
            if provider == "deepseek":
                try:
                    content = self._call_ollama(prompt, timeout)
                    self.last_call_details["provider"] = "ollama_fallback"
                    return content
                except Exception:
                    pass
            
            return f"生成失败: {e}"
    
    # ========== 2.0核心接口 ==========
    
    def chat(self, messages: list[dict], use_cache: bool = True,
             system_prompt: str = None) -> dict:
        """调用LLM聊天接口 (v1.2: 支持路由)"""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # 提取用户消息用于路由
        user_msg = ""
        for msg in messages:
            if msg.get('role') == 'user':
                user_msg += msg.get('content', '') + ' '
        
        # 缓存检查
        cache_key = self._cache_key(messages)
        if use_cache and config.LLM_CACHE_ENABLED and cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            return {
                "content": cached.get("content", ""),
                "role": "assistant", "model": "cache",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_ms": 0, "fallback": False,
                "provider": cached.get("provider", "cache"),
            }
        
        # 路由
        provider = self._route_provider(user_msg)
        
        start_time = time.time()
        
        try:
            if provider == "ollama":
                content = self._call_ollama_chat(messages)
                provider_used = "ollama"
            else:
                content = self._call_deepseek_chat(messages)
                provider_used = "deepseek"
            
            elapsed = (time.time() - start_time) * 1000
            self.call_count += 1
            
            result = {
                "content": content, "role": "assistant",
                "model": self.ollama_model if provider == "ollama" else self.model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_ms": elapsed, "fallback": False,
                "provider": provider_used,
            }
            
            # 缓存
            if config.LLM_CACHE_ENABLED:
                self.response_cache[cache_key] = {
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "provider": provider_used,
                }
                if len(self.response_cache) > 500:
                    oldest = sorted(self.response_cache.items(),
                                   key=lambda x: x[1].get('timestamp', ''))[:50]
                    for k, _ in oldest:
                        del self.response_cache[k]
            
            return result
        except Exception as e:
            print(f"[LLM] {provider}调用失败: {e}")
            
            # 降级到另一provider
            try:
                if provider == "deepseek":
                    content = self._call_ollama_chat(messages)
                    provider_used = "ollama_fallback"
                else:
                    content = self._call_deepseek_chat(messages)
                    provider_used = "deepseek_fallback"
                
                elapsed = (time.time() - start_time) * 1000
                return {
                    "content": content, "role": "assistant",
                    "model": "fallback",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "latency_ms": elapsed, "fallback": True,
                    "provider": provider_used,
                }
            except Exception:
                pass
            
            # 完全失败
            fallback = '{"action_type": "search", "target_description": "unknown", "reasoning": "LLM不可用，使用默认计划"}'
            return {
                "content": fallback, "role": "assistant",
                "model": "fallback",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_ms": (time.time() - start_time) * 1000,
                "fallback": True, "provider": "none",
            }
    
    # ========== Provider实现 ==========
    
    def _route_provider(self, prompt: str) -> str:
        """路由决策"""
        try:
            from core.llm.llm_router import llm_router
            return llm_router.route(prompt)
        except ImportError:
            pass
        
        # 简单规则: 短消息用本地，长消息用云端
        if len(prompt) < 100:
            return "ollama" if self.ollama_enabled else "deepseek"
        return "deepseek"
    
    def _call_deepseek(self, prompt: str, timeout: int) -> str:
        """调用DeepSeek API"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=1,
        )
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content or ""
        
        # 更新统计
        self.deepseek_calls += 1
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
        
        return content
    
    def _call_ollama(self, prompt: str, timeout: int) -> str:
        """调用Ollama本地模型"""
        import requests
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        self.ollama_calls += 1
        
        return data.get("response", "")
    
    def _call_deepseek_chat(self, messages: list[dict]) -> str:
        """DeepSeek Chat格式调用"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content or ""
        self.deepseek_calls += 1
        
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
        
        # 更新推理内容
        reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ''
        self.last_call_details["reasoning_content"] = reasoning
        
        return content
    
    def _call_ollama_chat(self, messages: list[dict]) -> str:
        """Ollama Chat格式调用"""
        import requests
        
        # 转换消息格式
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": ollama_messages,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        self.ollama_calls += 1
        
        return data.get("message", {}).get("content", "")
    
    def _cache_key(self, messages: list[dict]) -> str:
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def extract_json(self, response: dict) -> dict:
        """从LLM响应中提取JSON"""
        content = response.get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw_content": content, "parse_error": True}
    
    def get_stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "cache_size": len(self.response_cache),
            "model": self.model,
            "deepseek_calls": self.deepseek_calls,
            "ollama_calls": self.ollama_calls,
            "ollama_available": self.ollama_enabled,
        }
    
    def check_ollama_health(self) -> bool:
        """检查Ollama服务是否运行"""
        try:
            import requests
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# 全局单例
llm_client = LLMClient()
