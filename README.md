# 🚁 UAVagent 1.3 — 自进化多智能体协同无人机检测与跟踪系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2%2B-8A2BE2)](https://github.com/ultralytics/ultralytics)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.16-green)](https://developer.nvidia.com/tensorrt)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--Reasoner-536DFE)](https://deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**UAVagent 1.3** 在 1.2 基础上引入 **VisDrone 专训微调模型**、
**5 引擎 TensorRT FP16 WBF 融合**、**自适应共识过滤**、
**torchvision ReID** 和 **VLM 视觉推理模块**，
实现 MOTA 0.838（+1.3pp vs 1.2），推理速度 3x 提升（50ms）。

---

## 1.3 核心升级

| 特性 | 描述 |
|------|------|
| VisDrone 专训 YOLO11x | 微调训练 21 epochs，检测数 +125.9%，Recall 0.990 |
| 5 引擎 TensorRT FP16 融合 | RTX 4070S 上推理 3x 加速（140ms → 50ms） |
| 自适应共识过滤 v1.3.1 | 主力模型放宽阈值 primary=0.45，平衡召回/精度 |
| torchvision ResNet50 ReID | 2048 维外观特征，自动降级 HSV |
| VLM 视觉推理模块 | GPT-4V / Qwen-VL 客户端就绪 |
| 定性分析可视化 | 时间序列检测跟踪效果图生成 |
| 跟踪器修复 | min_hits=1 消除历史遗留的确认延迟 |

---

## 性能对比 (VisDrone 100帧)

| 指标 | 1.2 融合 | 1.3 融合 | 提升 |
|------|:------:|:------:|:---:|
| MOTA | 0.825 | **0.838** | +1.3pp |
| IDF1 | 0.906 | **0.922** | +1.6pp |
| Recall | 0.842 | **0.955** | +11.3pp |
| FN | 579 | **164** | -415 |
| 推理速度 | 140ms | **50ms** | 3x |
| ID Switches | 0 | 0 | OK |

---

## 快速开始

### 1. 环境准备

git clone https://github.com/StarlitPupils/UAVagentSystem.git
cd UAVagent1.3
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 2. 下载模型

python download_models.py

### 3. 配置 API 密钥

复制 .env.example 为 .env，填入 DeepSeek API Key。

### 4. 运行

python main.py

---

## 基准测试

python benchmark_v13_full.py
python benchmark_accuracy.py
python run_qualitative_analysis.py
python bench_tensorrt_speed.py
python verify_all.py

---

## 项目结构

UAVagent1.3/
├── agents/                 # 12 个智能体
├── core/                   # 核心模块
│   ├── detection/          # 5 模型 WBF 融合 + SAHI
│   ├── tracking/           # EKF 跟踪 + ReID
│   ├── llm/                # LLM/VLM 客户端
│   ├── memory/             # ChromaDB 向量记忆
│   └── edge/               # TensorRT 导出推理
├── training/               # 微调训练脚本
├── evaluation/             # 评估 + 可视化
├── config/                 # 全局配置
├── api/                    # FastAPI 服务
├── benchmark_v13_full.py   # 1.3 完整基准
└── UAVagent_1.3_知识迁移文档.md

## 引用

@software{UAVagent2026,
  author = {GouZengrui},
  title = {UAVagent: A Self-Evolving Multi-Agent System for UAV Detection and Tracking},
  year = {2026},
  version = {1.3},
  url = {https://github.com/StarlitPupils/UAVagentSystem}
}

## 许可证

MIT License
