# 🚁 UAVagent 1.0 — 自进化多智能体协同无人机检测与跟踪系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2%2B-8A2BE2)](https://github.com/ultralytics/ultralytics)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--Reasoner-536DFE)](https://deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**UAVagent** 是一个基于 **异构多智能体协同框架** 的无人机智能感知与决策系统。它融合了 **YOLOv8** 目标检测、**Transformer 自注意力跟踪**、**大语言模型（LLM）驱动推理**以及 **代码级自进化机制**，能够理解自然语言命令，在仿真或真实无人机视角下自主完成目标搜索与跟踪任务。

本项目源于本科创新研究课题，旨在探索 **LLM Agent 在具身智能场景下的自我优化能力**，并为无人机 AI 系统提供一个模块化、可扩展、便于教学与科研的开源平台。

---

## ✨ 主要特性

### 🤖 九大智能体协同（含训练智能体）

系统由九个专职智能体组成，每个智能体负责特定的子任务，通过消息传递协同工作。所有智能体均支持动态加载和自定义替换。

#### 1. 感知智能体 (PerceptionAgent)
- **职责**：从环境中获取视觉信息。
- **数据源**：支持 **AirSim 仿真**（实时图像流）和 **VisDrone 数据集**（离线视频序列）两种模式。
- **多模态**：可同时处理 RGB 图像和模拟热成像图像，并进行决策级融合。
- **降级**：当仿真未连接时，自动返回模拟检测数据，保证系统流程完整性。

#### 2. 推理智能体 (ReasoningAgent)
- **职责**：将自然语言命令与当前视觉状态转化为结构化的行动计划。
- **LLM 集成**：默认使用 **DeepSeek-Reasoner**，支持 OpenAI 格式 API 替换。
- **三级降级策略**：
    1. **LLM 缓存**：相同命令和环境状态下直接返回缓存结果，延迟降至毫秒级。
    2. **案例库检索**：LLM 失败时，检索历史成功案例中相似场景的决策。
    3. **本地规则**：关键词匹配 + 最高置信度目标选择，确保离线可用。
- **输出**：标准化 JSON 计划（`action_type`, `target_id`, `target_description` 等）。

#### 3. 行动智能体 (ActionAgent)
- **职责**：执行推理智能体制定的行动计划。
- **真实模式**：通过 AirSim API 控制无人机（起飞、降落、移动、悬停）。
- **仿真/模拟模式**：当 AirSim 未连接时，模拟执行结果，用于快速验证。

#### 4. 集成智能体 (IntegrationAgent)
- **职责**：任务流程的总指挥，串联感知→推理→行动→学习全流程。
- **指标采集**：记录每次任务的延迟、成功率、LLM 调用详情等，存入 `TaskMetrics`。
- **反思触发**：每完成 5 次任务自动触发一次反思进化流程。

#### 5. 报告智能体 (ReportingAgent)
- **职责**：汇总系统运行日志，生成 JSON 格式的分析报告。
- **输出**：报告保存在 `output/reports/` 目录下。

#### 6. 学习智能体 (LearningAgent)
- **职责**：收集每次任务的环境数据、计划和执行结果，为后续模型微调积累训练样本。
- **数据格式**：JSON 批次文件，保存在 `output/training_data/`。

#### 7. 反思智能体 (ReflectionAgent)
- **职责**：定期分析系统日志，调用 LLM 识别性能瓶颈和可优化点。
- **输出**：生成自然语言优化建议，传递给元智能体。

#### 8. 元智能体 (MetaAgent)
- **职责**：实现 **代码级自进化**。
- **工作流**：
    1. 接收反思智能体的建议。
    2. 调用 LLM 生成 Python 代码补丁。
    3. 对补丁进行 AST 语法检查。
    4. 在沙盒环境中运行批量测试验证补丁有效性。
    5. 验证通过则自动部署，失败则回滚。
- **权限**：可修改其他智能体的源代码（需沙盒验证或人工审批）。

#### 9. 训练智能体 (TrainingAgent)
- **职责**：自动化模型微调与部署。
- **功能**：
    - 将 VisDrone 标注转换为 YOLO 格式。
    - 调用 Ultralytics 训练接口，支持超参数网格搜索。
    - 训练完成后自动替换 `models/` 下的模型文件。
    - 支持 GPU/CPU 自适应。

### 🧠 大模型交互深度优化
- **多提供商支持**：通过 `.env` 文件配置 `LLM_PROVIDER`，可轻松切换 DeepSeek / OpenAI。
- **智能重试**：超时或失败时自动重试，并降级到备选模型。
- **Token 统计**：每次调用记录 prompt/completion tokens，用于成本分析。
- **思维链保留**：对于 DeepSeek-Reasoner，保留 `reasoning_content` 供调试。

### 🎯 检测与跟踪
- **检测器**：**YOLOv8x**（可替换为任意 Ultralytics 支持的模型），支持 CPU/CUDA。
- **跟踪器（可配置切换）**：
    - **Transformer 跟踪器**（默认）：基于多头自注意力的目标关联，ID 保持能力强。
    - **BoT-SORT**（备选）：传统 SOTA 跟踪器，通过修改配置文件即可切换。
- **多模态融合**：RGB 与热成像检测结果通过增强 IoU 融合，提升低光照鲁棒性。

### 🔄 自进化闭环（核心创新）
1. **反思触发**：集成智能体每完成 5 次任务自动触发。
2. **建议生成**：反思智能体读取日志，LLM 分析瓶颈。
3. **补丁生成**：元智能体生成修复代码。
4. **沙盒测试**：在隔离环境中运行 `batch_runner.py` 验证补丁。
5. **自动部署/回滚**：指标提升则保留，否则恢复原状。

### 📊 全栈评估与可视化
系统自动采集以下四维度指标，并生成发表级对比图表：
- **跟踪性能**：MOTA、IDF1、Precision、Recall（基于 `motmetrics`）。
- **系统效率**：平均任务延迟、端到端吞吐量、缓存命中率。
- **LLM 交互**：调用成功率、降级率、平均 Token 消耗、思维链长度。
- **智能体协作**：协作回合数、决策一致性（Jaccard）、降级决策质量。

**图表类型**：分组柱状图、多维雷达图、热力图、延迟箱线图、Token 分析图。

### 📦 工具链
- **VisDrone 加载器**：自动读取序列与标注，支持自动/手动触发任务。
- **AirSim 连接器**：无缝对接 UE 仿真，提供无人机控制与图像获取 API。
- **会话归档**：每次运行自动创建 `session_YYYYMMDD_HHMMSS` 目录，日志、指标、图表完整保存。

---
## 📋 项目成果与创新点

### 一、项目成果简介

本项目成功研制了一套面向无人机平台的自进化异构多智能体协同检测与跟踪系统——UAVagent 1.0。该系统突破了传统无人机依赖预设程序或单一模型进行目标跟踪的局限，首次将大语言模型驱动的认知推理、Transformer自注意力跟踪与智能体代码级自演化机制深度融合，构建了一个具备自主感知、智能决策、持续学习与自我优化能力的机载人工智能系统。在系统架构层面，项目设计并实现了包含感知、推理、行动、集成、报告、学习、反思、元智能体及训练智能体在内的九大专职智能体。通过显式的分工与消息协同，各智能体各司其职：感知智能体融合RGB与模拟热成像数据，推理智能体借助DeepSeek-Reasoner大模型解析自然语言命令并生成行动计划，行动智能体负责执行无人机操控指令，反思与元智能体则构成了业界领先的“经验-反思-进化”闭环，赋予系统根据任务日志自动发现瓶颈、生成代码补丁并安全部署的自进化能力。在视觉算法层面，系统采用了YOLOv8x作为基础检测器，并创新性地集成了一个轻量化Transformer跟踪器。该跟踪器利用多头自注意力机制对帧间目标进行序列化关联，有效提升了复杂背景下目标ID的保持能力。系统同时支持BoT-SORT等传统跟踪器，可通过配置文件灵活切换，为算法对比研究提供了便利。在鲁棒性与实时性方面，项目设计了“LLM缓存—案例库检索—本地规则”三级降级策略。实验数据表明，缓存机制使得重复指令的响应延迟从18.3秒骤降至毫秒级（降低79%），显著提升了系统的实时交互体验。此外，项目还构建了一套覆盖跟踪精度、系统效率、LLM交互质量和智能体协作一致性的全栈评估体系，能够自动生成雷达图、热力图等发表级图表，为系统性能的量化分析提供了坚实的数据支撑。通过在VisDrone无人机数据集上的真实回放验证，优化后的系统取得了MOTA 0.254、IDF1 0.405的跟踪精度，决策一致性达到满分1.00，充分证明了系统在真实场景下的有效性与鲁棒性。目前，该项目已整理为模块化、标准化的开源框架并在GitHub发布，具备完善的文档与示例，不仅可作为无人机自主系统研究的试验平台，也为探索大模型驱动的具身智能提供了高价值的学术参考。

### 二、项目的创新点与特色

本项目的核心创新并非单一算法的修修补补，而是在系统架构层面重新审视了无人机智能感知任务的执行范式。传统无人机视觉系统通常遵循一种固定的流水线——检测然后跟踪然后依据预设逻辑做出响应——这种模式在面对复杂多变的自然语言指令或未曾预见的边缘场景时显得僵硬而脆弱。UAVagent 1.0 通过引入异构多智能体协同与大语言模型认知推理打破了这一局限，并在工程实现上完成了多项具有鲜明特色的机制设计。首要创新在于提出并实现了一套具备代码级自演化能力的多智能体协作框架。在现有的绝大多数LLM Agent研究中，“自我优化”往往局限于提示词微调或参数层面的适应，本质仍是静态程序。本项目中的元智能体被赋予了直接修改其他智能体源代码的权限——反思智能体通过对任务日志的深度分析定位性能瓶颈后将优化建议以自然语言形式提交给元智能体，元智能体随即调用大模型生成具体的Python补丁并在沙盒环境中完成语法检查与回归测试。只有当补丁使系统在成功率与延迟等核心指标上产生可量化的正向增益时才会被自动合并否则立即回滚。这一整套“分析—生成—验证—部署”的自动化流程使得无人机机载智能体具备了根据任务经验持续调整内部逻辑的结构级进化能力，这在无人机具身智能领域尚属前沿尝试。第二个显著特色是将目标跟踪任务中的帧间关联问题重新表述为序列匹配任务并设计了一款轻量化Transformer跟踪器。与依赖卡尔曼滤波和手工设计运动模型的传统方法不同，该跟踪器利用多头自注意力机制直接捕捉不同帧检测框之间的全局依赖关系。具体实现中系统将当前帧检测框与历史轨迹的特征拼接后输入编码层，通过自注意力和交叉注意力计算相似度矩阵，再经由匈牙利算法完成匹配。这一设计使得跟踪器对遮挡和相机运动具有天然鲁棒性且无需复杂的特征工程。实验表明Transformer跟踪器在IDF1指标上较传统方法有显著优势同时保持了与YOLOv8x检测器无缝集成的轻量特性。第三项重要创新在于构建了LLM缓存、案例库检索与本地规则递进的三级降级策略以确保系统在复杂网络环境下的鲁棒性。大模型推理虽然强大但其响应延迟和可用性并非绝对可控，尤其当部署环境涉及移动网络或边缘计算设备时。本系统通过MD5哈希将相同的命令与视觉上下文映射至缓存结果，使得重复查询的响应时间从数十秒压缩至毫秒级别；若缓存未命中且大模型调用失败则系统回退至历史案例库检索出最相似的成功计划作为备选；当所有远程智能均不可用时则触发基于关键词和置信度排序的本地规则保障任务流的连续性。这种层层递进的容错设计使得UAVagent能够适应从云端到离线的全场景部署而无需修改任何核心代码。最后一点特色体现在系统对Sim-to-Real全流程实验支持的开箱即用集成上。项目内置了针对AirSim仿真环境与VisDrone真实数据集的统一数据加载接口以及一套完整的指标采集与可视化流水线。开发者只需修改配置文件即可在仿真飞行、离线数据集回放以及未来真实无人机部署之间无缝切换。自动生成的对比图表涵盖跟踪精度、系统延迟、LLM交互质量与智能体协作一致性四个维度，极大降低了多智能体系统性能评估与论文级结果呈现的门槛。这种将前沿算法研究与工程可复现性紧密结合的设计理念使得项目本身不仅是一个算法验证原型更成为相关领域研究者快速搭建实验基线的理想平台。

## ⚙️ 配置与切换

所有可配置项均通过修改 `config/settings.py` 或 `.env` 实现，无需改动核心代码。

### 切换跟踪器
在 `config/settings.py` 中修改：
```python
TRACKER_TYPE: str = "transformer"   # 或 "botsort"
切换 LLM 提供商
编辑 .env 文件：LLM_PROVIDER=deepseek               # 或 openai
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-reasoner    # 或 deepseek-chat
启用/禁用缓存与案例库
通过环境变量控制（在启动前设置）：

$env:LLM_CACHE_ENABLED = "true"      # 默认开启
$env:CASE_BASE_ENABLED = "true"      # 默认开启
切换检测模型
修改 config/settings.py：
YOLO_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "yolov8x.pt")
``` 
🚀 快速开始
1. 环境准备
Python 3.10 / 3.11 推荐（3.13 需注意包兼容性）

克隆项目并创建虚拟环境：
git clone https://github.com/StarlitPupils/UAVagent.git
cd UAVagent
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
2. 安装依赖
pip install -r requirements.txt
3. 下载 YOLO 模型
mkdir models
# 下载 yolov8x.pt 放入 models/ 目录
# 官方链接：https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt
4. 配置 API 密钥
复制 .env.example 为 .env，填入 DeepSeek API Key（申请地址）。

5. 运行系统
python main.py
交互命令示例：

takeoff — 起飞

search — 搜索目标

track car — 跟踪车辆

report — 生成报告

reflect — 触发反思进化

run_dataset E:/VisDrone/VisDrone2019-MOT-val uav0000086_00000_v auto — 数据集回放

🧪 实验与评估
运行对比实验
python run_experiments.py
自动执行四组配置（Baseline, Opt-A, Opt-B, Opt-C），每组 25 次任务，结果归档至 output/experiment_batch_xxx/。

生成增强图表
python visualizer_enhanced.py
图表输出至归档目录的 figures_enhanced/ 下。

评估跟踪精度
python run_visdrone_real.py      # 生成跟踪结果
python evaluate_tracking.py      # 计算 MOTA/IDF1
📈 核心实验结果（Optimized-C 配置）
指标数值
MOTA0.254
IDF10.405
平均任务延迟4.29 s（首次） / <1 s（缓存命中）
任务成功率100%（模拟模式）
决策一致性1.00
缓存延迟降低79% (vs. 无缓存基线)
📁 项目结构
text
UAVagent/
├── config/                 # 配置（模型、跟踪器、密钥）
├── core/                   # 核心模块（视觉、LLM客户端、缓存等）
├── agents/                 # 九个智能体
├── evaluation/             # 评估、可视化、沙盒
├── simulation/             # AirSim 连接与真值采集
├── utils/                  # 工具（指标计算）
├── models/                 # YOLO 模型文件
├── output/                 # 运行时输出（自动归档）
├── custom_agents/          # 用户自定义智能体
├── tests/                  # 测试场景
├── main.py                 # 主入口
├── run_visdrone_real.py    # 真实数据回放
├── run_experiments.py      # 批量实验
├── evaluate_tracking.py    # 跟踪评估
├── requirements.txt        # 依赖清单
└── README.md
🎓 学术创新点
代码级自演化多智能体架构 — 元智能体可修改其他智能体源码，实现结构级自我优化。

Transformer 自注意力跟踪器 — 将目标关联建模为序列匹配，ID 保持能力优于传统方法。

三级降级策略 — LLM 缓存 + 案例库 + 本地规则，保障系统鲁棒性。

全栈评估体系 — 首次在多智能体系统中同时覆盖跟踪、系统、LLM、协作四维度。

Sim-to-Real 无缝切换 — 同一代码库支持仿真、数据集回放、真实无人机（规划中）。

🤝 贡献与定制
添加自定义智能体
在 custom_agents/ 下创建 .py 文件，类名以 Agent 结尾。

启动时自动注册。

或通过元智能体生成：> generate_agent MyAgent 需求描述。

贡献代码
欢迎 Issue / PR，请保持代码风格一致。

📄 引用
bibtex
@software{UAVagent2026,
  author = {GouZengrui},
  title = {UAVagent: A Self-Evolving Multi-Agent System for UAV Detection and Tracking},
  year = {2026},
  url = {https://github.com/StarlitPupils/UAVagent}
}
📜 许可证
MIT License

🙏 致谢
Ultralytics YOLO

DeepSeek API

VisDrone Dataset

所有贡献者与导师

⭐ 如果对你有帮助，欢迎 Star！
📧 作者现承接计算机视觉项目（工业缺陷检测/视频分析/模型优化部署），联系邮箱：1526123439@qq.com
