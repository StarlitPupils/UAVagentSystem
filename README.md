🚁 UAVagent 1.2 — 自进化多智能体协同无人机检测与跟踪系统
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3%2B-8A2BE2)](https://github.com/ultralytics/ultralytics)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--v4-536DFE)](https://deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.0-brightgreen)]()
**UAVagent** 是一个基于 **异构多智能体协同框架** 的无人机智能感知与决策系统。它融合了 **5模型集成检测（WBF）**、**EKF扩展卡尔曼跟踪**、**大语言模型（LLM）驱动推理**、**ChromaDB向量记忆库** 以及 **代码级自进化机制**，能够理解自然语言命令，在仿真或真实无人机视角下自主完成目标搜索与跟踪任务。
> **🔥 v1.2 核心升级**：5模型融合（+YOLOv10/RT-DETR）+ EKF跟踪 + CLAHE预处理 + Ollama本地LLM + ReAct多轮推理
---
## ✨ 主要特性
### 🤖 十二大智能体协同
| # | 智能体 | 职责 | 版本 |
|---|--------|------|------|
| 1 | **PerceptionAgent** | 5模型融合检测 + CLAHE预处理 | 🆕 v1.2 |
| 2 | **ReasoningAgent** | ReAct多轮推理 + LLM路由 | 🆕 v1.2 |
| 3 | **ActionAgent** | 无人机操控（字典驱动） | ✅ |
| 4 | **IntegrationAgent** | 全流程编排 + 指标采集 | ✅ |
| 5 | **ReportingAgent** | 日志汇总 + JSON报告 | ✅ |
| 6 | **LearningAgent** | 训练样本收集 | ✅ |
| 7 | **ReflectionAgent** | 性能瓶颈分析 | ✅ |
| 8 | **MetaAgent** | 代码级自进化 | ✅ |
| 9 | **TrainingAgent** | 模型微调 + 超参搜索 | ✅ |
| 10 | **SafetyAgent** | 安全守护 | ✅ |
| 11 | **OrchestratorAgent** | 多机协同 | ✅ |
| 12 | **CommunicationAgent** | 通信管理 | ✅ |
### 🎯 检测与跟踪（v1.2 核心创新）
- **5模型集成检测**：YOLO11x + YOLO8x + YOLO11n + YOLOv10n + RT-DETR-l WBF融合
- **质量门控**：异常模型自动熔断，灰度图场景智能排除RT-DETR
- **CLAHE预处理**：低光照/雾霾自适应增强
- **EKF跟踪器**：8状态扩展卡尔曼 + HSV外观匹配 + 轨迹插值
- **深度ReID**：OSNet备选，自动降级为HSV直方图
### 🧠 LLM 深度集成（v1.2 升级）
- **Ollama本地模型**：离线推理支持（qwen2.5/lama3）
- **LLM路由**：简单任务本地、复杂任务云端，成本优化
- **ReAct多轮推理**：Thought→Action→Observation循环
- **三级降级策略**：LLM缓存 → 案例库检索 → 本地规则
### 🔄 自进化闭环
1. **反思触发**：每完成5次任务自动触发
2. **建议生成**：反思智能体读取日志，LLM分析瓶颈
3. **补丁生成**：元智能体生成Python代码补丁
4. **AST语法检查**：自动验证补丁合法性
5. **沙盒测试**：隔离环境中运行批量测试
6. **自动部署/回滚**：指标提升则保留，否则恢复
### 🌐 Web Dashboard + REST API
- **FastAPI服务器**：一键启动
- **Web Dashboard**：实时状态卡片 + 任务指令 + AI对话面板
- **WebSocket遥测**：实时推送检测结果
- **RESTful API**：`/mission` `/chat` `/memory/search` `/stats` `/config`
---
## 📈 VisDrone 真实数据集验证结果
### 100帧基准测试 (3672 GT目标)
| 指标 | 单模型 (YOLO11x) | 5模型融合 | 提升 |
|------|-------------------|-----------|------|
| **MOTA** | 0.640 | **0.825** | **+18.5 pp** |
| **IDF1** | 0.780 | **0.906** | **+12.5 pp** |
| **Recall** | 0.640 | **0.842** | +20.2 pp |
| **Precision** | 1.000 | 0.979 | – |
| **ID Switches** | 0 | 0 | ✅ 完美 |
| **FN (漏检)** | 1322 | **579** | **‑743** |
### 版本性能演进
| 版本 | MOTA | IDF1 | 核心升级 |
|------|------|------|----------|
| 1.0 | 0.254 | 0.405 | 9 Agent + YOLOv8x |
| 1.1 | 0.784 | 0.880 | 12 Agent + 3模型WBF + ChromaDB |
| **1.2** | **0.825** | **0.906** | **5模型融合 + EKF + CLAHE + Ollama** |
> ✅ **结论**：1.2版本在1.1基础上再提升MOTA +4.1pp，且保持零ID切换。
---
## ⚙️ 快速开始
### 1. 克隆项目
git clone https://github.com/StarlitPupils/UAVagentSystem.git
cd UAVagentSystem
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
2. 安装依赖
pip install -r requirements.txt
3. 下载模型
mkdir models
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolov8x.pt'); YOLO('yolo11n.pt')"
# 可选增强模型
python -c "from ultralytics import YOLO; YOLO('yolov10n.pt'); YOLO('rtdetr-l.pt')"
4. 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
5. 启动系统
# 交互式命令行
python main.py
# 或启动 Web Dashboard
python api/server.py
# 浏览器访问 http://localhost:8000/dashboard
🧪 实验与评估
# 检测精度对比
python benchmark_accuracy.py
# MOT基准测试（合成场景）
python benchmark_mot.py
python benchmark_mot_advanced.py
# 真实VisDrone基准（100帧完整评估）
python benchmark_real_mot.py
python benchmark_real_mot_v2.py
# 自进化闭环验证
python test_evolution.py
# 全链路验证
python verify_all.py
📁 项目结构
UAVagent/
├── agents/                 # 十二个智能体
├── config/                 # 配置中心（settings.py）
├── core/                   # 核心模块
│   ├── detection/          # 5模型集成检测器（WBF）+ 共识过滤 + 预处理
│   ├── tracking/           # EKF跟踪器 + ReID特征提取
│   ├── memory/             # ChromaDB向量记忆库
│   ├── llm/                # LLM客户端 + 路由器
│   └── edge/               # ONNX/TensorRT边缘部署
├── api/                    # FastAPI服务器 + Dashboard
├── evaluation/             # 评估、可视化、沙盒
├── tests/                  # 测试场景
├── docker/                 # Docker容器化
├── .github/workflows/      # CI/CD流水线
├── models/                 # YOLO模型文件
├── output/                 # 运行时输出（自动归档）
├── main.py                 # 主入口
├── requirements.txt        # 依赖清单
├── CHANGELOG.md            # 版本更新日志
└── README.md
🎓 学术创新点
5模型质量门控融合：异常模型自动熔断 + 灰度图智能检测 + 动态权重，在VisDrone上MOTA超越任何单一模型18.5pp
EKF+HSV增强跟踪器：8状态扩展卡尔曼 + 统一64维外观特征 + 轨迹插值，ID Switch=0
代码级自演化：元智能体可修改其他智能体源码，通过"反思→补丁生成→AST检查→沙盒验证→自动部署"闭环实现结构级自我优化
LLM智能路由：Ollama本地模型处理简单任务，DeepSeek云端处理复杂推理，兼顾离线可用性与成本
ReAct多轮推理：Thought→Action→Observation循环，决策准确性显著提升
全栈评估体系：首次在多智能体系统中同时覆盖跟踪精度、系统效率、LLM交互质量、智能体协作一致性四个维度
📄 引用
software{UAVagent2025,
  author = {GouZengrui},
  title = {UAVagent: A Self-Evolving Multi-Agent System for UAV Detection and Tracking},
  year = {2025},
  version = {1.2},
  url = {https://github.com/StarlitPupils/UAVagentSystem}
}
📜 许可证
MIT License - 详见 LICENSE

🙏 致谢
Ultralytics YOLO
DeepSeek API
VisDrone Dataset
ChromaDB
Ollama
所有贡献者与导师
⭐ 如果对你有帮助，欢迎 Star！

📧 联系方式：1526123439@qq.com
