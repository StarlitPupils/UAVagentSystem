# core/edge/__init__.py
from .tensorrt_exporter import (
    TensorRTExporter,
    TensorRTInference,
    benchmark_all_precisions,
)

# 向后兼容别名
benchmark_tensorrt = benchmark_all_precisions

__all__ = [
    "TensorRTExporter",
    "TensorRTInference",
    "benchmark_all_precisions",
    "benchmark_tensorrt",
]