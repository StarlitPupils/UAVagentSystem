# core/llm/llm_router.py
"""LLM路由器 - UAVagent 1.2
根据任务复杂度自动选择: 本地Ollama vs 云端DeepSeek
"""
import re
from typing import Optional
from config.settings import config


class LLMRouter:
    """LLM路由决策器"""
    
    # 简单任务关键词 (适合本地模型)
    SIMPLE_PATTERNS = [
        r'\b(search|搜索|find|查找)\b',
        r'\b(report|报告|status|状态)\b',
        r'\b(hover|悬停|land|降落|takeoff|起飞)\b',
        r'\b(simple|简单)\b',
    ]
    
    # 复杂任务关键词 (需要云端模型)
    COMPLEX_PATTERNS = [
        r'\b(analyze|分析|reason|推理|plan|规划)\b',
        r'\b(complex|复杂|difficult|困难)\b',
        r'\b(track.*occluded|跟踪.*遮挡)\b',
        r'\b(multi.*drone|多机|swarm|编队)\b',
        r'\b(emergency|紧急|danger|危险)\b',
    ]
    
    def __init__(self):
        self.enabled = config.LLM_ROUTING_ENABLED
        self.routing_stats = {"simple": 0, "complex": 0, "fallback": 0}
    
    def route(self, prompt: str, messages: list = None) -> str:
        """
        路由决策
        返回: "ollama" | "deepseek" | "auto"
        """
        if not self.enabled:
            return "deepseek"  # 默认云端
        
        # 提取用户消息内容
        user_content = ""
        if messages:
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content += msg.get('content', '') + ' '
        user_content += prompt
        
        user_lower = user_content.lower()
        
        # 检查是否强制使用某模型
        if '#ollama' in user_lower or '#local' in user_lower:
            self.routing_stats["simple"] += 1
            return "ollama"
        if '#deepseek' in user_lower or '#cloud' in user_lower:
            self.routing_stats["complex"] += 1
            return "deepseek"
        
        # 复杂度评分
        complexity_score = 0
        
        # 复杂特征加分
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, user_content, re.IGNORECASE):
                complexity_score += 2
        
        # 简单特征减分
        for pattern in self.SIMPLE_PATTERNS:
            if re.search(pattern, user_content, re.IGNORECASE):
                complexity_score -= 1
        
        # 消息长度因素
        if len(user_content) > 500:
            complexity_score += 1
        if len(user_content) < 50:
            complexity_score -= 1
        
        # 决策
        if complexity_score > 0:
            self.routing_stats["complex"] += 1
            return "deepseek"
        else:
            self.routing_stats["simple"] += 1
            return "ollama"
    
    def get_stats(self) -> dict:
        return self.routing_stats
    
    def reset_stats(self):
        self.routing_stats = {"simple": 0, "complex": 0, "fallback": 0}


# 全局单例
llm_router = LLMRouter()
