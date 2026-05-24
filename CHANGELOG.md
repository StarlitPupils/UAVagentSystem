# UAVagent 更新日志

## [1.4.0] - 2026-05-24

### 重大升级

#### 检测优化
- v14 微调模型：lr0=0.0005, 50 epochs, mAP50 0.313 (+72% vs 1.3)
- 共识过滤 v14：适配新模型置信度分布 (primary=0.55, single=0.70)
- VisDrone 100帧: MOTA 0.858 (+2.0pp vs 1.3), FP 423

#### VLM 视觉推理
- 智谱 GLM-4V-Flash 免费接入（国内可用）
- 场景分析 + 异常检测接口完整

#### MAVLink 飞控对接
- 完整 MAVLink v2 连接器 (UDP/Serial/TCP)
- 支持 PX4/ArduPilot，模拟模式测试通过

#### SAHI 大图验证
- 拼贴大图检测数 +68%，推理加速 3-8x
- 自适应切片大小（640/800/1024/1280）

#### TensorRT 加速
- FP16 实测 2.2x 加速 (6.0ms/引擎)
- INT8 量化框架就绪

### 性能对比 (VisDrone 100帧)

| 指标 | 1.3 融合 | 1.4 融合 | 提升 |
|------|---------|----------|------|
| MOTA | 0.838 | **0.858** | +2.0 pp |
| IDF1 | 0.922 | **0.932** | +1.0 pp |
| Recall | 0.955 | **0.973** | +1.8 pp |
| FP | 431 | **423** | -8 |
| FN | 164 | **99** | -65 |

### 修复
- 修复 TensorRT engine `Path.with_suffix()` 后缀错误
- 修复 `__init__.py` 导出函数名不一致
- 修复 v14 模型置信度校准偏移 (共识过滤重新调参)
- 修复自适应阈值过度降低 conf 导致 FP 暴增

### 新增文件（10+）
- core/mavlink_connector.py
- tests/test_vlm_integration.py (v2)
- tests/test_p2_connectors.py
- tests/test_p3_adaptive.py
- benchmark_v14_clean.py

---

## [1.3.0] - 2026-05-22

### 重大升级
- VisDrone 专训 YOLO11x 微调模型（检测数 +125.9%, Recall 0.990）
- 5 引擎 TensorRT FP16 WBF 融合（推理速度 3x 提升, <50ms）
- 自适应共识过滤 v1.3.1（主力模型放宽阈值）
- torchvision ResNet50 ReID (2048维)
- VLM 视觉推理模块
- SAHI 切片推理框架（实验性模块）
- VisDrone 100帧: MOTA 0.838 (+1.3pp vs 1.2)

### 修复
- 修复 TensorRT 引擎 "Invalid device id" 错误（9 轮迭代）
- 修复 5 引擎 WBF 融合路径
- 修复共识过滤过严/过松平衡（6 轮调参）
- 修复跟踪器 min_hits=5 延迟问题
- 修复 ReID 特征维度冲突

---

## [1.2.0] - 2025-01-XX
- 5模型集成融合 + CLAHE 预处理 + EKF 跟踪
- Ollama 本地LLM + ReAct 多轮推理
- VisDrone 100帧: MOTA 0.825

## [1.1.0] - 2025-01-XX
- 12 Agent + 3模型WBF + 卡尔曼跟踪 + ChromaDB
- VisDrone 100帧: MOTA 0.784

## [1.0.0] - 2024-12-XX
- 9 Agent 基础框架 + YOLOv8x + Transformer 跟踪器
- VisDrone MOTA: 0.254