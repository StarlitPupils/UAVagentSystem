# 🚁 UAVagent 1.1 — 自进化多智能体协同无人机检测与跟踪系统
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3%2B-8A2BE2)](https://github.com/ultralytics/ultralytics)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--v4-536DFE)](https://deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
**UAVagent** 是一个基于 **异构多智能体协同框架** 的无人机智能感知与决策系统。它融合了 **多模型集成检测（WBF）**、**卡尔曼+外观增强跟踪**、**大语言模型（LLM）驱动推理**、**向量记忆库（ChromaDB）** 以及 **代码级自进化机制**，能够理解自然语言命令，在仿真或真实无人机视角下自主完成目标搜索与跟踪任务。
> **🔥 v1.1 核心升级**：多模型加权融合检测（WBF）+ 卡尔曼外观跟踪器 + ChromaDB 向量记忆 + Web Dashboard + Docker 边缘部署
本项目源于本科创新研究课题，旨在探索 **LLM Agent 在具身智能场景下的自我优化能力**，并为无人机 AI 系统提供一个模块化、可扩展、便于教学与科研的开源平台。
---
## ✨ 主要特性
### 🤖 十二大智能体协同
系统由十二个专职智能体组成，每个智能体负责特定的子任务，通过消息传递协同工作。所有智能体均支持动态加载和自定义替换。
| # | 智能体 | 职责 | 状态 |
|---|--------|------|------|
| 1 | **PerceptionAgent** | 多模型融合检测（YOLOv11 + WBF） | ✅ v1.1 增强 |
| 2 | **ReasoningAgent** | LLM 推理 + 向量记忆上下文 | ✅ v1.1 增强 |
| 3 | **ActionAgent** | 无人机操控（字典驱动） | ✅ v1.1 增强 |
| 4 | **IntegrationAgent** | 全流程编排 + 指标采集 | ✅ |
| 5 | **ReportingAgent** | 日志汇总 + JSON 报告 | ✅ |
| 6 | **LearningAgent** | 训练样本收集 | ✅ |
| 7 | **ReflectionAgent** | 性能瓶颈分析 | ✅ |
| 8 | **MetaAgent** | 代码级自进化 | ✅ |
| 9 | **TrainingAgent** | 模型微调 + 超参搜索 | ✅ |
| 10 | **SafetyAgent** | 安全守护（地理围栏/碰撞预警） | 🆕 v1.1 |
| 11 | **OrchestratorAgent** | 多机协同任务分配 | 🆕 v1.1 |
| 12 | **CommunicationAgent** | 5G/MAVLink 通信管理 | 🆕 v1.1 |
### 🎯 检测与跟踪（v1.1 核心创新）
- **多模型集成检测**：YOLOv11x + YOLOv8x + YOLOv11n 三模型加权融合（WBF），消除单模型漏检与误检
- **增强跟踪器**：卡尔曼运动预测 + HSV 外观匹配 + IoU 匈牙利关联，ID 保持能力大幅提升
- **共识过滤器**：多模型交叉确认 + 自适应阈值，兼顾 Precision 与 Recall
### 🧠 LLM 深度集成
- **统一 LLM 客户端**：仅使用 `deepseek-v4-flash`，支持缓存、重试、Token 统计
- **向量记忆库**：ChromaDB 语义检索替代简单哈希缓存，支持长期/短期/情景记忆
- **三级降级策略**：LLM 缓存 → 案例库检索 → 本地规则
### 🔄 自进化闭环
1. **反思触发**：每完成 5 次任务自动触发
2. **建议生成**：反思智能体读取日志，LLM 分析瓶颈
3. **补丁生成**：元智能体生成 Python 代码补丁
4. **AST 语法检查**：自动验证补丁合法性
5. **沙盒测试**：隔离环境中运行批量测试
6. **自动部署/回滚**：指标提升则保留，否则恢复
### 🌐 Web Dashboard + REST API
- **FastAPI 服务器**：`python api/server.py` 一键启动
- **Web Dashboard**：实时状态卡片 + 任务指令 + AI 对话面板
- **WebSocket 遥测**：实时推送检测结果
- **RESTful API**：`/mission` `/chat` `/memory/search` `/stats` `/config`
### 📊 全栈评估与可视化
系统自动采集四维度指标并生成发表级对比图表：
| 维度 | 指标 |
|------|------|
| 跟踪性能 | MOTA、IDF1、HOTA、Recall、Precision |
| 系统效率 | 平均任务延迟、吞吐量、缓存命中率 |
| LLM 交互 | 调用成功率、降级率、Token 消耗 |
| 智能体协作 | 协作回合数、决策一致性 |
---
## 📈 VisDrone 真实数据集验证结果
在 **VisDrone2019-MOT-val / uav0000086_00000_v** 序列上，100 帧基准测试：
| 指标 | 单模型 (YOLOv11x) | 多模型融合 | 提升 |
|------|-------------------|-----------|------|
| **MOTA** | 0.743 | **0.784** | **+4.1 pp** |
| **IDF1** | 0.874 | **0.880** | **+0.6 pp** |
| **Recall** | 0.893 | **0.898** | +0.5 pp |
| **Precision** | 0.856 | 0.853 | – |
| **ID Switches** | 0 | 0 | – |
| **FN (漏检)** | 393 | **373** | **‑20** |
> ✅ **结论**：多模型融合在真实无人机航拍序列上，MOTA 系统性超越任何单一模型，且保持完美的 ID 一致性（零切换）。
---
## ⚙️ 快速开始
### 1. 环境准备
Python 3.10 / 3.11 推荐
git clone https://github.com/StarlitPupils/UAVagent1.0.git
cd UAVagent1.0
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
2. 安装依赖
pip install -r requirements.txt
3. 下载 YOLO 模型
mkdir models
# 自动下载（推荐）
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolov8x.pt'); YOLO('yolo11n.pt')"
4. 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
5. 启动系统
# 交互式命令行
python main.py
# 或启动 Web Dashboard
python api/server.py
# 浏览器访问 http://localhost:8000/dashboard
6. 交互命令示例
Text
> help                    # 查看所有命令
> status                  # 系统状态
> search                  # 搜索目标
> track car               # 跟踪车辆
> switch_model yolo11x    # 热切换模型
> switch_tracker bytetrack # 热切换跟踪器
> benchmark               # 运行检测器基准测试
> reflect                 # 触发反思进化
> api                     # 启动 API 服务器
🧪 实验与评估
运行检测精度对比
python benchmark_accuracy.py
运行 MOT 基准测试（合成场景）
python benchmark_mot.py          # 基础 MOT
python benchmark_mot_advanced.py # 遮挡+小目标场景
运行真实数据集基准（VisDrone）
python benchmark_real_mot.py     # 100 帧完整评估
运行批量实验
python run_experiments.py        # 自动执行 4 组配置 × 25 次任务
生成评估图表
python visualizer_enhanced.py    # 雷达图/热力图
python generate_final_charts.py  # MOT 对比柱状图
评估跟踪精度
python run_visdrone_real.py      # 生成跟踪结果
python evaluate_tracking.py      # 计算 MOTA/IDF1
📁 项目结构
Text
UAVagent/
├── agents/                 # 十二个智能体
├── config/                 # 配置中心（settings.py）
├── core/                   # 核心模块
│   ├── detection/          # 多模型集成检测器（WBF）
│   ├── tracking/           # 增强跟踪器（卡尔曼+外观）
│   ├── memory/             # 向量记忆库（ChromaDB）
│   ├── llm/                # LLM 客户端（deepseek-v4-flash）
│   └── edge/               # 边缘部署（ONNX/TensorRT）
├── api/                    # FastAPI 服务器 + Dashboard
├── evaluation/             # 评估、可视化、沙盒
├── experiments/            # 实验配置文件
├── tests/                  # 测试场景
├── docker/                 # Docker 容器化配置
├── models/                 # YOLO 模型文件（需单独下载）
├── output/                 # 运行时输出（自动归档）
├── main.py                 # 主入口
├── requirements.txt        # 依赖清单
├── .env.example            # 环境配置模板
└── README.md
🎓 学术创新点
多模型加权融合检测（WBF）：三个 YOLO 模型通过加权框融合，在 VisDrone 真实数据上 MOTA 提升 4+ 百分点，召回率提升 0.5 个百分点，同时保持零 ID 切换。

卡尔曼+外观增强跟踪器：8 状态卡尔曼预测 + HSV 直方图外观匹配 + 匈牙利 IoU 关联，相比传统方法显著降低轨迹碎片化。

代码级自演化多智能体架构：元智能体可修改其他智能体源码，通过"反思→补丁生成→AST 检查→沙盒验证→自动部署"闭环实现结构级自我优化。

向量记忆库（ChromaDB）：用语义检索替代简单哈希缓存，支持短期/长期/情景三层记忆，RAG 增强 LLM 上下文。

三级降级策略：LLM 缓存 + 案例库 + 本地规则，确保从云端到离线全场景可用。

全栈评估体系：首次在多智能体系统中同时覆盖跟踪精度（MOTA/IDF1/HOTA）、系统效率、LLM 交互质量、智能体协作一致性四个维度。

Sim-to-Real 无缝切换：同一代码库支持 AirSim 仿真、VisDrone 数据集回放、真实无人机部署。

🐳 Docker 部署
cd docker
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:8000/dashboard
🤝 贡献与定制
添加自定义智能体
在 custom_agents/ 下创建 .py 文件，类名以 Agent 结尾，启动时自动注册。

或通过元智能体生成：

Text
> generate_agent MyAgent 需求描述
贡献代码
欢迎 Issue / PR，请保持代码风格一致。

📄 引用
Bibtex
@software{UAVagent2026,
  author = {GouZengrui},
  title = {UAVagent: A Self-Evolving Multi-Agent System for UAV Detection and Tracking},
  year = {2026},
  version = {1.1},
  url = {https://github.com/StarlitPupils/UAVagent1.0}
}
📜 许可证
MIT License - 详见 LICENSE

🙏 致谢
Ultralytics YOLO
DeepSeek API
VisDrone Dataset
ChromaDB
所有贡献者与导师
⭐ 如果对你有帮助，欢迎 Star！

📧 联系方式：1526123439@qq.com
