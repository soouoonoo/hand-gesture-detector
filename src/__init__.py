"""
手势识别项目包
版本: 1.0.0
作者: soouoonoo
"""

__version__ = "1.0.0"
__author__ = "手势识别项目团队"

from .utils.hand_detector import HandDetector
from .data.collector import GestureCollector
from .main import main

__all__ = ["HandDetector", "GestureCollector", "main"]
