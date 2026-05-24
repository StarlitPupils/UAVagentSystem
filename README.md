# 🚁 UAVagent 1.4 — 自进化多智能体协同目标检测与跟踪系统
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3%2B-8A2BE2)](https://github.com/ultralytics/ultralytics)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.16-green)](https://developer.nvidia.com/tensorrt)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--Reasoner-536DFE)](https://deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**UAVagent 1.4** 在 1.3 基础上引入 **v14 微调模型 (mAP +72%)**、**SAHI 大图验证**、
**智谱 GLM-4V-Flash 免费 VLM**、**MAVLink 飞控对接**、**TensorRT INT8 量化框架**，
实现 **VisDrone MOTA 0.858**（+2.0pp vs 1.3），FP 降至 423。
---
## 核心升级
| 特性 | 描述 |
|------|------|
| v14 微调模型 | lr0=0.0005, 50 epochs, mAP50 0.313 (+72% vs 1.3) |
| 共识过滤 v14 | 适配新模型置信度分布，primary=0.55, single=0.70 |
| 智谱 GLM-4V-Flash | 国内免费视觉语言模型，场景分析+异常检测 |
| MAVLink 飞控对接 | UDP/Serial/TCP 连接，支持 PX4/ArduPilot |
| SAHI 大图验证 | 拼接大图检测数 +68%，推理加速 3-8x |
| TensorRT INT8 框架 | FP16 加速 2.2x (6.0ms)，INT8 校准就绪 |
---
## 性能对比 (VisDrone 100帧)
| 指标 | 1.3 融合 | 1.4 融合 | 提升 |
|------|:---:|:---:|:---:|
| **MOTA** | 0.838 | **0.858** | **+2.0pp** |
| **IDF1** | 0.922 | **0.932** | +1.0pp |
| **Recall** | 0.955 | **0.973** | +1.8pp |
| **FP** | 431 | **423** | -8 |
| **FN** | 164 | **99** | -65 |
| **ID Switches** | 0 | 0 | ✅ |
| **推理速度** | <50ms | <50ms | — |
---

## 快速开始
### 环境准备  
git clone https://github.com/StarlitPupils/UAVagentSystem.git  
cd UAVagentSystem  
python -m venv venv  
venv\Scripts\activate          # Windows  
source venv/bin/activate       # Linux/Mac  
pip install -r requirements.txt  

下载模型  
python download_models.py  
配置 API 密钥  
复制 .env.example 为 .env，填入：  
DEEPSEEK_API_KEY — DeepSeek API Key (LLM推理)  
ZHIPU_API_KEY — 智谱 API Key (免费VLM，https://bigmodel.cn)  

运行  
python main.py                   # 交互式终端  
python benchmark_v14_clean.py    # 100帧完整基准  
python verify_all.py             # 端到端验证  

基准测试  
python benchmark_v14_clean.py    # 1.4 MOTA基准 (100帧, ~6分钟)  
python benchmark_v13_full.py     # 1.3 兼容基准  
python bench_tensorrt_speed.py   # TensorRT 速度测试  
python run_qualitative_analysis.py  # 定性分析效果图  

## 项目结构  
UAVagentSystem/  
├── agents/                 # 12 个智能体 (感知/推理/行动/学习/反思)  
├── core/                   # 核心模块  
│   ├── detection/          # 5 模型 WBF 融合 + 共识过滤v14 + SAHI  
│   ├── tracking/           # EKF 跟踪 + ReID (torchvision 2048维)  
│   ├── llm/                # LLM/VLM 客户端 (DeepSeek + 智谱免费)  
│   ├── memory/             # ChromaDB 向量记忆库  
│   ├── edge/               # TensorRT FP16/INT8 导出推理  
│   └── mavlink_connector   # MAVLink 飞控连接器 (1.4新增)  
├── training/               # 微调训练脚本 (v14优化版)	
├── evaluation/             # 评估 + 定性分析可视化  
├── tests/                  # 单元测试 + 集成测试	
├── config/                 # 全局配置 (1.4定型参数)  
├── api/                    # FastAPI 服务	
├── models/                 # 模型文件 (运行 download_models.py 获取)	
└── benchmark_v14_clean.py  # 1.4 完整基准测试  

## 引用  
Bibtex	
@software{UAVagent2026,  
  author = {GouZengrui},	
  title = {UAVagent: A Self-Evolving Multi-Agent System for UAV Detection and Tracking},	
  year = {2026},	
  version = {1.4},	
  url = {https://github.com/StarlitPupils/UAVagentSystem}  
}  

## 许可证	  
MIT License
