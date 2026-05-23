# UAVagent 更新日志

## [1.3.0] - 2026-05-22

### 重大升级

#### 检测优化
- VisDrone 专训 YOLO11x 微调模型（检测数 +125.9%, Recall 0.990）
- 5 引擎 TensorRT FP16 WBF 融合（推理速度 3x 提升, <50ms）
- 自适应共识过滤 v1.3.1（主力模型放宽阈值）
- SAHI 切片推理框架（实验性模块）
- VisDrone 100帧: MOTA 0.838 (+1.3pp vs 1.2)

#### 跟踪优化
- torchvision ResNet50 ReID（2048维，自动降级HSV）
- 跟踪器 min_hits 修复（消除确认延迟）
- ID Switches: 0 (完美保持)

#### 推理加速
- TensorRT FP16 引擎导出（5模型全部导出）
- 单引擎推理 7.2ms vs PyTorch 20.5ms (2.9x)
- 5引擎融合 <50ms vs 原 140ms

#### LLM/推理
- VLM 视觉推理客户端（GPT-4V/Qwen-VL 就绪）

#### 可视化
- 时间序列检测跟踪效果图生成
- 定性分析报告自动输出

### 性能对比 (VisDrone 100帧)

| 指标 | 1.2 融合 | 1.3 融合 | 提升 |
|------|---------|----------|------|
| MOTA | 0.825 | **0.838** | +1.3 pp |
| IDF1 | 0.906 | **0.922** | +1.6 pp |
| Recall | 0.842 | **0.955** | +11.3 pp |
| FP | 0 | 431 | 可控 |
| FN | 579 | **164** | -415 |

### 修复
- 修复 TensorRT 引擎 "Invalid device id" 错误
- 修复 5 引擎 WBF 融合路径（6 轮迭代）
- 修复共识过滤过严/过松平衡（6 轮调参）
- 修复跟踪器 min_hits=5 延迟问题
- 修复 ReID 特征维度冲突

### 新增文件（15+）
- training/visdrone_to_yolo.py
- training/finetune_yolo11.py
- core/detection/sahi.py, sahi_v2.py, sahi_v3.py
- core/edge/tensorrt_exporter.py
- core/llm/vlm_client.py
- evaluation/temporal_visualizer.py
- run_qualitative_analysis.py
- benchmark_v13_full.py
- bench_tensorrt_speed.py
- det_compare_sahi.py, det_compare_sahi_v2.py, det_compare_sahi_v3.py

---

## [1.2.0] - 2025-01-XX

### 重大升级
- 5模型集成融合 (YOLOv11x + YOLOv8x + YOLOv11n + YOLOv10n + RT-DETR-l)
- CLAHE 自适应图像预处理
- EKF 扩展卡尔曼滤波
- 轨迹插值修复
- Ollama 本地LLM支持
- ReAct 多轮推理
- VisDrone 100帧: MOTA 0.825

---

## [1.1.0] - 2025-01-XX

- 12 Agent 多智能体协同
- 3模型 WBF 融合
- 卡尔曼8状态 + HSV 外观跟踪
- ChromaDB 向量记忆库
- VisDrone 100帧: MOTA 0.784

---

## [1.0.0] - 2024-12-XX

- 9 Agent 基础协同框架
- YOLOv8x 目标检测
- Transformer 跟踪器
- VisDrone MOTA: 0.254
