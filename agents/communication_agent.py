# agents/communication_agent.py
"""通信智能体 — 负责日志通知、外部API调用、多机通信"""
import json, requests, threading, queue, time

class CommunicationAgent:
    def __init__(self):
        self.name = "CommunicationAgent"
        self.message_queue = queue.Queue()
        self.webhook_url = None
        self.enabled = True

    def set_webhook(self, url: str):
        self.webhook_url = url

    def send_notification(self, title: str, content: str, level: str = "info"):
        """发送通知（webhook / 打印）"""
        msg = {"title": title, "content": content, "level": level, "timestamp": time.time()}
        self.message_queue.put(msg)
        print(f"[Comm] {level.upper()}: {title}")
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=msg, timeout=5)
            except Exception as e:
                print(f"[Comm] webhook失败: {e}")

    def broadcast(self, event_type: str, payload: dict):
        """广播事件（多机协同预留接口）"""
        print(f"[Comm] Broadcast: {event_type} -> {json.dumps(payload, ensure_ascii=False)[:100]}")