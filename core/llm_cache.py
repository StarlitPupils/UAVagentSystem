import hashlib
import json
import time
from typing import Dict, Optional, Tuple
from collections import OrderedDict

class LLMCache:
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[float, str]] = OrderedDict()
    def _generate_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    def get(self, prompt: str, model: str) -> Optional[str]:
        key = self._generate_key(prompt, model)
        if key not in self._cache:
            return None
        timestamp, response = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        print(f"[LLM Cache] 命中缓存，跳过 API 调用", flush=True)
        return response
    def set(self, prompt: str, model: str, response: str):
        key = self._generate_key(prompt, model)
        self._cache[key] = (time.time(), response)
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    def clear(self):
        self._cache.clear()