# core/llm_client.py
"""兼容桥接 - 将旧导入路径 core.llm_client 重定向到 core.llm.llm_client"""
from core.llm.llm_client import LLMClient, llm_client

__all__ = ["LLMClient", "llm_client"]
