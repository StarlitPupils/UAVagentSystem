# core/detection/__init__.py (v1.3)
from .ensemble_detector import EnsembleDetector
from .sahi import SAHIInference, create_sahi_detector

__all__ = ["EnsembleDetector", "SAHIInference", "create_sahi_detector"]
