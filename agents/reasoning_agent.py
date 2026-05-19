# agents/reasoning_agent.py (v1.2)
"""推理智能体 v2 - 多轮ReAct + 向量记忆 + LLM路由"""
import json
import asyncio
import time
from config.settings import config


class ReasoningAgent:
    def __init__(self):
        self.name = "ReasoningAgent"
        self._llm = None
        self._memory = None
        self.last_call_details = {"success": False, "model": "", "latency_ms": 0}
        
        # v1.2: ReAct配置
        self.react_enabled = True
        self.max_rounds = config.REACT_MAX_ROUNDS
        self.react_timeout = config.REACT_TIMEOUT
    
    @property
    def llm(self):
        if self._llm is None:
            from core.llm.llm_client import llm_client
            self._llm = llm_client
        return self._llm
    
    @property
    def llm_client(self):
        return self.llm
    
    @property
    def memory(self):
        if self._memory is None:
            from core.memory.memory_manager import memory_manager
            self._memory = memory_manager
        return self._memory
    
    # ========== 1.0 兼容接口 ==========
    async def parse_command(self, command: str, visual_state: dict = None) -> dict:
        return await self.reason(command, visual_state)
    
    # ========== 2.0 核心接口 (ReAct) ==========
    async def reason(self, command: str, visual_state: dict = None) -> dict:
        """多轮ReAct推理"""
        vis = visual_state or {}
        
        # 安全获取字段
        count = vis.get("num_objects", vis.get("count", 0))
        targets = vis.get("detections", vis.get("targets", []))
        
        # 构建目标信息
        targets_info = ""
        if targets:
            simplified = []
            for t in targets[:20]:  # 最多20个目标
                bbox = t.get("bbox", [])
                class_name = t.get("class_name", t.get("class", "object"))
                conf = t.get("confidence", 0)
                tid = t.get("id", "?")
                num_models = t.get("num_models", 1)
                simplified.append({
                    "id": tid, "class": class_name,
                    "conf": round(conf, 2),
                    "pos": [round(b, 1) for b in bbox[:2]] if bbox else [],
                    "verified_by": num_models,
                })
            targets_info = f"目标列表 ({count}个): {json.dumps(simplified, ensure_ascii=False)[:1000]}\n"
        
        # 检索记忆上下文
        context = ""
        try:
            ctx = self.memory.get_context_for_llm(command)
            if ctx:
                context = f"相关历史经验:\n{ctx}\n\n"
        except Exception:
            pass
        
        # v1.2: ReAct多轮推理
        if self.react_enabled and self._is_complex_command(command):
            return await self._react_reason(command, targets_info, context)
        
        # 单轮推理
        return await self._single_reason(command, targets_info, context)
    
    async def _single_reason(self, command: str, targets_info: str, context: str) -> dict:
        """单轮推理 (快速路径)"""
        system_prompt = (
            "你是无人机目标检测与跟踪系统的推理智能体。\n"
            "根据用户命令和当前检测到的目标列表，制定行动计划。\n"
            "输出严格JSON（不要markdown代码块）：\n"
            '{"action_type":"search|track|report", "target_description":"...", '
            '"target_id":null, "confidence":0.5, "reasoning":"..."}'
        )
        
        messages = [{
            "role": "user",
            "content": f"命令: {command}\n检测到 {targets_info}"
        }]
        
        if context:
            messages.insert(0, {"role": "system", "content": context})
        
        try:
            response = self.llm.chat(messages, system_prompt=system_prompt)
            plan = self.llm.extract_json(response)
            
            self.last_call_details = {
                "success": not response.get("fallback", False),
                "model": response.get("model", config.LLM_MODEL),
                "latency_ms": response.get("latency_ms", 0),
                "provider": response.get("provider", ""),
                "react_rounds": 1,
            }
            
            if "parse_error" in plan:
                plan = self.get_fallback_plan(command)
                self.last_call_details["success"] = False
            
            # 存储记忆
            try:
                self.memory.remember(
                    f"命令:{command} -> {plan.get('action_type','?')}",
                    memory_type="reasoning"
                )
            except Exception:
                pass
            
            return plan
        except Exception as e:
            print(f"[Reasoning] LLM调用失败: {e}")
            self.last_call_details = {
                "success": False, "model": "", "latency_ms": 0,
                "error": str(e), "react_rounds": 1,
            }
            return self.get_fallback_plan(command)
    
    async def _react_reason(self, command: str, targets_info: str, context: str) -> dict:
        """多轮ReAct推理 (Thought -> Action -> Observation -> ...)"""
        print(f"[ReAct] 启动多轮推理 (命令: {command[:50]}...)")
        
        observations = [targets_info]
        final_plan = None
        rounds = 0
        
        for round_idx in range(self.max_rounds):
            rounds = round_idx + 1
            
            # 构建ReAct提示词
            obs_text = "\n".join([f"[观测{ri+1}] {obs}" 
                                  for ri, obs in enumerate(observations)])
            
            react_prompt = (
                f"{context}"
                f"用户命令: {command}\n\n"
                f"{obs_text}\n\n"
                f"请按ReAct格式思考并行动:\n"
                f"Thought: [分析当前情况]\n"
                f"Action: [决定下一步: search|track|report|finish]\n"
                f"如果Action=finish，请输出最终JSON计划。\n\n"
                f"输出格式 (JSON): "
                f'{{"thought":"...", "action":"search|track|report|finish", '
                f'"plan":{{"action_type":"...", "target_description":"...", '
                f'"confidence":0.5, "reasoning":"..."}}}}'
            )
            
            try:
                response = self.llm.chat(
                    [{"role": "user", "content": react_prompt}],
                    system_prompt="你是一个使用ReAct框架的无人机推理智能体。先思考(Thought)，再行动(Action)。",
                )
                
                result = self.llm.extract_json(response)
                
                thought = result.get("thought", "")
                action = result.get("action", "").lower()
                
                print(f"[ReAct] 轮次{round_idx+1}: Thought={thought[:60]}... Action={action}")
                
                if action == "finish" or "plan" in result:
                    final_plan = result.get("plan", result)
                    break
                
                # 否则记录观测
                observations.append(f"Thought: {thought}\nAction: {action}")
                
            except Exception as e:
                print(f"[ReAct] 轮次{round_idx+1}失败: {e}")
                break
        
        # 如果没有产生最终计划，用最后一轮的结果
        if final_plan is None:
            final_plan = self.get_fallback_plan(command)
        
        self.last_call_details = {
            "success": True,
            "model": self.llm.model,
            "latency_ms": 0,
            "provider": "react",
            "react_rounds": rounds,
        }
        
        # 存储
        try:
            self.memory.remember(
                f"ReAct({rounds}轮):{command} -> {final_plan.get('action_type','?')}",
                memory_type="reasoning_react"
            )
        except Exception:
            pass
        
        return final_plan
    
    def _is_complex_command(self, command: str) -> bool:
        """判断是否需要多轮推理"""
        complex_keywords = [
            '分析', '推理', '复杂', '多目标', '遮挡', '编队',
            'analyze', 'complex', 'multi', 'occluded', 'swarm',
        ]
        cmd_lower = command.lower()
        
        # 长命令可能是复杂任务
        if len(command) > 100:
            return True
        
        # 包含复杂关键词
        if any(kw in cmd_lower for kw in complex_keywords):
            return True
        
        return False
    
    def reason_sync(self, command: str, visual_state: dict = None) -> dict:
        """同步推理"""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.reason(command, visual_state))
                )
                return future.result(timeout=60)
        except RuntimeError:
            return asyncio.run(self.reason(command, visual_state))
    
    def get_fallback_plan(self, command: str) -> dict:
        """本地规则降级"""
        cmd_lower = command.lower()
        
        if 'search' in cmd_lower or '搜索' in command:
            action_type = "search"
        elif 'report' in cmd_lower or '报告' in command:
            action_type = "report"
        else:
            action_type = "track"
        
        target_map = [
            (['car', 'vehicle', '车辆', '汽车', '轿车'], '车辆'),
            (['person', 'pedestrian', '行人', '人'], '行人'),
            (['truck', '卡车', '货车'], '卡车'),
            (['bus', '公交', '巴士'], '公交车'),
            (['bicycle', 'bike', '自行车'], '自行车'),
            (['uav', 'drone', '无人机'], '无人机'),
        ]
        target_desc = "目标"
        for keywords, desc in target_map:
            if any(kw in cmd_lower or kw in command for kw in keywords):
                target_desc = desc
                break
        
        return {
            "action_type": action_type,
            "target_description": target_desc,
            "target_id": None,
            "confidence": 0.5,
            "reasoning": f"本地规则降级 - 关键词匹配>{target_desc}<",
            "fallback": True,
        }
    
    async def record_successful_plan(self, command, vis, plan, success):
        """记录成功计划"""
        try:
            self.memory.remember(
                f"成功:{command} -> {plan.get('action_type','?')}",
                memory_type="successful_plan",
                metadata={"command": command, "plan": plan}
            )
        except Exception:
            pass
