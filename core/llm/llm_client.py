# core/llm/llm_client.py
"""LLM客户端 - 仅 deepseek-v4-flash，兼容1.0全部接口"""
import os
import json
import time
import hashlib
import asyncio
from datetime import datetime
from config.settings import config

class LLMClient:
    def __init__(self):
        self.api_key = config.LLM_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = config.LLM_BASE_URL
        self.model = config.LLM_MODEL  # 固定 deepseek-v4-flash
        self.max_tokens = config.LLM_MAX_TOKENS
        self.temperature = config.LLM_TEMPERATURE
        self.timeout = config.LLM_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES
        # 缓存
        self.response_cache: dict[str, dict] = {}
        # 统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        # === 1.0兼容：last_call_details ===
        self.last_call_details: dict = {
            "success": False,
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_content": "",
            "ttft": 0.0,
            "error_type": "",
            "latency_ms": 0,
        }

    # ========== 1.0兼容接口 ==========

    def get_last_call_details(self) -> dict:
        """1.0兼容：返回最近一次调用的详情"""
        return self.last_call_details

    async def generate(self, prompt: str, timeout_override: int = None) -> str:
        """1.0兼容：异步生成（meta_agent/reflection_agent使用）"""
        timeout = timeout_override or self.timeout
        start_time = time.time()
        self.last_call_details = {
            "success": False,
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_content": "",
            "ttft": 0.0,
            "error_type": "",
            "latency_ms": 0,
        }
        try:
            from openai import OpenAI, APITimeoutError
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
            elapsed = (time.time() - start_time) * 1000
            # 更新统计
            self.call_count += 1
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            # 更新last_call_details
            self.last_call_details = {
                "success": True,
                "model": response.model or self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_content": getattr(response.choices[0].message, 'reasoning_content', '') or '',
                "ttft": elapsed,
                "error_type": "",
                "latency_ms": elapsed,
            }
            # 缓存
            cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            self.response_cache[cache_key] = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
            return content
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.last_call_details = {
                "success": False,
                "model": self.model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_content": "",
                "ttft": elapsed,
                "error_type": type(e).__name__,
                "latency_ms": elapsed,
            }
            print(f"[LLM] generate失败: {e}")
            return f"生成失败: {e}"

    # ========== 2.0核心接口 ==========

    def _cache_key(self, messages: list[dict]) -> str:
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def chat(self, messages: list[dict], use_cache: bool = True,
             system_prompt: str = None) -> dict:
        """调用LLM聊天接口"""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        cache_key = self._cache_key(messages)
        if use_cache and config.LLM_CACHE_ENABLED and cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            print(f"[LLM] 缓存命中")
            result = {
                "content": cached.get("content", ""),
                "role": "assistant",
                "model": self.model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_ms": 0,
                "fallback": False,
            }
            self.last_call_details = {
                "success": True, "model": self.model,
                "prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_content": "", "ttft": 0, "error_type": "",
                "latency_ms": 0,
            }
            return result

        start_time = time.time()
        for attempt in range(self.max_retries):
            try:
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
                elapsed = (time.time() - start_time) * 1000
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ''

                self.call_count += 1
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

                result = {
                    "content": content,
                    "role": "assistant",
                    "model": response.model or self.model,
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                    "latency_ms": elapsed,
                    "fallback": False,
                }
                self.last_call_details = {
                    "success": True,
                    "model": response.model or self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_content": reasoning,
                    "ttft": elapsed,
                    "error_type": "",
                    "latency_ms": elapsed,
                }
                if config.LLM_CACHE_ENABLED:
                    self.response_cache[cache_key] = {
                        "content": content,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if len(self.response_cache) > 500:
                        oldest = sorted(self.response_cache.items(),
                                       key=lambda x: x[1].get('timestamp', ''))[:50]
                        for k, _ in oldest:
                            del self.response_cache[k]
                print(f"[LLM] 调用成功 ({self.model}) 延迟: {elapsed:.0f}ms")
                return result
            except Exception as e:
                print(f"[LLM] 第{attempt+1}次调用失败: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        elapsed = (time.time() - start_time) * 1000
        self.last_call_details = {
            "success": False, "model": self.model,
            "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_content": "", "ttft": elapsed,
            "error_type": "AllRetriesFailed", "latency_ms": elapsed,
        }
        fallback_content = '{"action_type": "search", "target_description": "unknown", "reasoning": "LLM调用失败，使用默认计划"}'
        return {
            "content": fallback_content,
            "role": "assistant",
            "model": "fallback",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "latency_ms": elapsed,
            "fallback": True,
        }

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
        }

# 全局单例
llm_client = LLMClient()
