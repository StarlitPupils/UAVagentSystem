# core/memory/memory_manager.py
"""记忆管理器 - 统一管理短期/长期/情景记忆"""
from typing import Optional
from .vector_store import vector_store
from .embedding import embedding_service

class MemoryManager:
    def __init__(self):
        self.short_term: list[dict] = []  # 当前任务的上下文
        self.vector_store = vector_store
        self.vector_store.initialize()

    def remember(self, content: str, memory_type: str = "general", metadata: dict = None):
        """存储记忆"""
        # 加入短期记忆
        self.short_term.append({"content": content, "type": memory_type})
        if len(self.short_term) > 50:
            self.short_term = self.short_term[-50:]
        # 持久化到向量库
        return self.vector_store.add_memory(content, metadata, memory_type)

    def recall(self, query: str, top_k: int = 5, memory_type: str = None) -> list[dict]:
        """检索记忆"""
        return self.vector_store.search(query, top_k, memory_type)

    def get_context_for_llm(self, current_command: str, max_tokens: int = 2000) -> str:
        """构建LLM上下文：最近记忆 + 相关历史"""
        results = self.recall(current_command, top_k=5)
        context_parts = []
        # 短期记忆
        if self.short_term:
            recent = self.short_term[-5:]
            context_parts.append("【最近操作】")
            for item in recent:
                context_parts.append(f"- {item['content'][:200]}")
        # 相关历史
        if results:
            context_parts.append("\n【相关历史经验】")
            for r in results:
                context_parts.append(f"- [{r['metadata'].get('type','')}] {r['content'][:300]} (相关度:{r['score']:.2f})")
        context = "\n".join(context_parts)
        if len(context) > max_tokens * 2:
            context = context[:max_tokens * 2]
        return context

    def clear_short_term(self):
        self.short_term = []

memory_manager = MemoryManager()
