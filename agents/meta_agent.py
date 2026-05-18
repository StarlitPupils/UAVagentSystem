# agents/meta_agent.py
"""元智能体 - 代码级自进化（仅 deepseek-v4-flash）"""
import os
import ast
import re
import shutil
from core.llm.llm_client import llm_client
from evaluation.sandbox import SandboxTester

class MetaAgent:
    def __init__(self):
        self.llm_client = llm_client
        self.meta_timeout = int(os.getenv("LLM_REFLECTION_TIMEOUT", "120"))
        self.tester = SandboxTester()

    def _extract_code(self, text: str) -> str:
        """提取Python代码块"""
        # 提取 ```python ... ``` 中的代码
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 提取任意 ``` ... ``` 中的内容
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 如果没有代码块，返回原文本（但尝试去除解释性前缀）
        lines = text.split('\n')
        code_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith(('#', '//', '/*', '*')):
                code_lines.append(line)
        return '\n'.join(code_lines)

    async def generate_patch(self, suggestions: str) -> str:
        """根据建议生成代码补丁"""
        prompt = f"根据以下建议生成Python代码补丁。只输出代码，不要任何解释或注释：\n{suggestions}"
        try:
            raw = await self.llm_client.generate(prompt, timeout_override=self.meta_timeout)
            patch = self._extract_code(raw)
            if not patch:
                return "生成的补丁为空"
            # 基础语法检查
            try:
                ast.parse(patch)
            except SyntaxError as e:
                print(f"[元智能体] 生成的补丁存在语法错误: {e}")
                return f"语法错误: {e}"
            return patch
        except Exception as e:
            return f"生成失败: {e}"

    async def apply_and_test(self, patch: str, target_file: str) -> bool:
        """沙盒测试补丁，成功则保留，失败则回滚"""
        if patch.startswith("生成失败") or patch.startswith("语法错误"):
            print(f"[元智能体] 补丁无效，跳过: {patch}")
            return False
        backup = target_file + ".backup_auto"
        if not os.path.exists(backup):
            shutil.copy(target_file, backup)
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(patch)
            success, new_success_rate, new_latency = self.tester.run_tests()
            if not success:
                print("[元智能体] 沙盒测试失败，回滚补丁")
                shutil.copy(backup, target_file)
                return False
            if self.tester.is_improvement(new_success_rate, new_latency):
                print(f"[元智能体] 改进验证通过！成功率={new_success_rate:.2%}，延迟={new_latency:.1f}s")
                return True
            else:
                print(f"[元智能体] 改进不显著，回滚。成功率={new_success_rate:.2%}")
                shutil.copy(backup, target_file)
                return False
        except Exception as e:
            print(f"[元智能体] 测试异常: {e}，回滚")
            if os.path.exists(backup):
                shutil.copy(backup, target_file)
            return False

    async def generate_custom_agent(self, agent_name: str, requirements: str) -> str:
        """生成自定义智能体"""
        prompt = f"生成名为 {agent_name} 的智能体类，要求：{requirements}。输出完整Python代码。"
        raw = await self.llm_client.generate(prompt, timeout_override=self.meta_timeout)
        code = self._extract_code(raw)
        custom_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'custom_agents')
        os.makedirs(custom_dir, exist_ok=True)
        filepath = os.path.join(custom_dir, f'{agent_name.lower()}.py')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        return f"已生成: {filepath}"
