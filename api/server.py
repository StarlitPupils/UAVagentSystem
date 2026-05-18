# api/server.py
"""FastAPI 服务器 - UAVagent 2.0 REST API（通过 ReasoningAgent 推理）"""
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from config.settings import config
from core.llm.llm_client import llm_client
from core.memory.memory_manager import memory_manager
from agents.safety_agent import safety_agent


class MissionRequest(BaseModel):
    command: str
    mode: str = "simulation"
    use_memory: bool = True


class ChatRequest(BaseModel):
    message: str
    use_memory: bool = True


class ConfigUpdate(BaseModel):
    model_name: Optional[str] = None
    tracker_type: Optional[str] = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="UAVagent 2.0 API",
        description="自进化多智能体协同无人机检测与跟踪系统",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "service": "UAVagent 2.0",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "llm_model": config.LLM_MODEL,
            "tracker": config.TRACKER_TYPE,
        }

    @app.post("/mission")
    async def execute_mission(req: MissionRequest):
        # 安全检查
        allowed, reason = safety_agent.validate_action({"action_type": req.command})
        if not allowed:
            raise HTTPException(status_code=403, detail=reason)

        # ---- 使用 ReasoningAgent（带系统提示词 + 降级）----
        from agents.reasoning_agent import ReasoningAgent
        reasoning = ReasoningAgent()

        # 构建模拟视觉状态
        visual_state = {
            "num_objects": 0,
            "detections": [],
        }

        try:
            plan = await reasoning.reason(req.command, visual_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推理失败: {e}")

        # 存储记忆
        try:
            memory_manager.remember(
                f"命令:{req.command} -> {plan.get('action_type','?')}:{plan.get('target_description','?')}",
                memory_type="mission"
            )
        except Exception:
            pass

        return {
            "command": req.command,
            "plan": plan,
            "llm_stats": {
                "model": reasoning.last_call_details.get("model", config.LLM_MODEL),
                "latency_ms": reasoning.last_call_details.get("latency_ms", 0),
                "fallback": reasoning.last_call_details.get("fallback", False),
                "success": reasoning.last_call_details.get("success", True),
            },
        }

    @app.post("/chat")
    async def chat(req: ChatRequest):
        context = memory_manager.get_context_for_llm(req.message) if req.use_memory else ""
        messages = [{"role": "user", "content": req.message}]
        if context:
            messages.insert(0, {"role": "system", "content": f"历史上下文:\n{context}"})
        response = llm_client.chat(messages)
        memory_manager.remember(f"对话:{req.message[:200]}", memory_type="chat")
        return {
            "reply": response.get("content"),
            "model": response.get("model"),
            "latency_ms": response.get("latency_ms"),
        }

    @app.get("/memory/search")
    async def search_memory(q: str, top_k: int = 5):
        results = memory_manager.recall(q, top_k=top_k)
        return {"query": q, "results": results}

    @app.get("/stats")
    async def get_stats():
        return {
            "llm": llm_client.get_stats(),
            "safety": safety_agent.get_safety_report(),
            "config": {
                "model": config.YOLO_MODEL_NAME,
                "tracker": config.TRACKER_TYPE,
                "llm_model": config.LLM_MODEL,
            },
        }

    @app.put("/config")
    async def update_config(update: ConfigUpdate):
        changes = {}
        if update.model_name and config.switch_model(update.model_name):
            changes["model"] = update.model_name
        if update.tracker_type and config.switch_tracker(update.tracker_type):
            changes["tracker"] = update.tracker_type
        return {"updated": changes, "message": "配置已更新"}

    @app.get("/dashboard")
    async def dashboard():
        return HTMLResponse(content="""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>UAVagent 2.0 Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#0a0a1a;color:#e0e0e0}
.header{background:linear-gradient(135deg,#1a1a3e,#2d2d5e);padding:20px;text-align:center}
.header h1{font-size:2em;background:linear-gradient(90deg,#4fc3f7,#7c4dff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.container{max-width:1200px;margin:0 auto;padding:20px}
.card{background:#1a1a2e;border-radius:12px;padding:20px;margin:15px 0;border:1px solid #2a2a4e}
.card h3{color:#4fc3f7;margin-bottom:12px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px}
.stat-item{background:#0d0d20;padding:15px;border-radius:8px;text-align:center}
.stat-value{font-size:1.8em;font-weight:bold;color:#7c4dff}
.stat-label{font-size:.85em;color:#888;margin-top:5px}
input,textarea,button{width:100%;padding:12px;margin:8px 0;border-radius:8px;border:1px solid #333;background:#0d0d20;color:#e0e0e0;font-size:1em}
button{background:linear-gradient(135deg,#4fc3f7,#7c4dff);border:none;cursor:pointer;font-weight:bold;color:#fff}
button:hover{opacity:.9}
#output{background:#0d0d20;padding:15px;border-radius:8px;white-space:pre-wrap;font-family:monospace;max-height:400px;overflow-y:auto;margin-top:10px}
.flex-row{display:flex;gap:15px}
.flex-row>*{flex:1}
</style>
</head>
<body>
<div class="header">
<h1>🚁 UAVagent 2.0 Dashboard</h1>
<p style="color:#888">自进化多智能体协同系统 | 状态: <span id="status">运行中</span></p>
</div>
<div class="container">
<div class="card">
<h3>📊 系统状态</h3>
<div class="stats-grid" id="statsGrid">
<div class="stat-item"><div class="stat-value" id="llmModel">-</div><div class="stat-label">LLM模型</div></div>
<div class="stat-item"><div class="stat-value" id="llmCalls">0</div><div class="stat-label">LLM调用次数</div></div>
<div class="stat-item"><div class="stat-value" id="cacheSize">0</div><div class="stat-label">缓存条目</div></div>
<div class="stat-item"><div class="stat-value" id="trackerType">-</div><div class="stat-label">跟踪器</div></div>
</div>
</div>
<div class="flex-row">
<div class="card">
<h3>🎯 任务指令</h3>
<input type="text" id="cmdInput" placeholder="输入命令: search / track car / report ..." />
<button onclick="sendMission()">🚀 执行任务</button>
<div id="output"></div>
</div>
<div class="card">
<h3>💬 AI对话</h3>
<textarea id="chatInput" rows="3" placeholder="与UAVagent对话..."></textarea>
<button onclick="sendChat()">💬 发送</button>
<div id="chatOutput"></div>
</div>
</div>
</div>
<script>
const API = '/';
async function loadStats(){
  try{
    const r = await fetch(API+'stats');
    const data = await r.json();
    document.getElementById('llmModel').textContent = data.config.llm_model;
    document.getElementById('llmCalls').textContent = data.llm.call_count;
    document.getElementById('cacheSize').textContent = data.llm.cache_size;
    document.getElementById('trackerType').textContent = data.config.tracker;
  }catch(e){console.error(e)}
}
async function sendMission(){
  const cmd = document.getElementById('cmdInput').value;
  if(!cmd){ alert('请输入命令'); return; }
  const out = document.getElementById('output');
  out.textContent = '执行中...';
  try{
    const r = await fetch(API+'mission',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({command:cmd, mode:'simulation', use_memory:true})
    });
    const text = await r.text();
    try{ out.textContent = JSON.stringify(JSON.parse(text), null, 2); }
    catch{ out.textContent = text; }
    loadStats();
  }catch(e){ out.textContent = '网络错误: '+e.message; }
}
async function sendChat(){
  const msg = document.getElementById('chatInput').value;
  if(!msg){ alert('请输入消息'); return; }
  const out = document.getElementById('chatOutput');
  out.textContent = '思考中...';
  try{
    const r = await fetch(API+'chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg, use_memory:true})
    });
    const text = await r.text();
    try{ const data = JSON.parse(text); out.textContent = data.reply; }
    catch{ out.textContent = text; }
  }catch(e){ out.textContent = '网络错误: '+e.message; }
}
loadStats();
setInterval(loadStats,5000);
</script>
</body>
</html>""")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "connected", "msg": "connected"}))
        try:
            while True:
                data = await websocket.receive_text()
                req_data = json.loads(data)
                cmd = req_data.get("command", "")
                if cmd:
                    resp = llm_client.chat([{"role": "user", "content": cmd}])
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": resp.get("content"),
                        "latency_ms": resp.get("latency_ms"),
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
        except WebSocketDisconnect:
            print("[WS] client disconnected")

    return app


app = create_app()

if __name__ == "__main__":
    print(f"🚀 UAVagent 2.0 API 启动中...")
    print(f"   LLM: {config.LLM_MODEL}")
    print(f"   跟踪器: {config.TRACKER_TYPE}")
    print(f"   地址: http://{config.API_HOST}:{config.API_PORT}")
    print(f"   Dashboard: http://{config.API_HOST}:{config.API_PORT}/dashboard")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")
