"""
模块包初始化
"""

__version__ = "1.0.0"
__author__ = "Gesture Recognition Team"
__email__ = "team@example.com"

# 模块导入 - 使用延迟导入以避免缺少依赖时的错误
__all__ = [
    'image_processing',
    'pose_detection',
    'distance_estimation', 
    'gesture_recognition',
    'fisheye_validation'
]

try:
    from . import image_processing
except ImportError:
    image_processing = None

try:
    from . import pose_detection
except ImportError:
    pose_detection = None

try:
    from . import distance_estimation
except ImportError:
    distance_estimation = None

try:
    from . import gesture_recognition
except ImportError:
    gesture_recognition = None

try:
    from . import fisheye_validation
except ImportError:
    fisheye_validation = None
