"""
模块包初始化
"""

__version__ = "1.0.0"
__author__ = "Gesture Recognition Team"
__email__ = "team@example.com"

# 模块导入
from . import image_processing
from . import pose_detection  
from . import distance_estimation
from . import gesture_recognition

__all__ = [
    'image_processing',
    'pose_detection',
    'distance_estimation', 
    'gesture_recognition'
]
