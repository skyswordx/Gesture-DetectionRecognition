"""
可视化模块 - 用于姿态校正距离测量的可视化

本模块提供以下可视化功能：
1. 姿态骨架绘制 (Requirements 7.1)
2. 测量信息显示 (Requirements 7.2)
3. 距离比值显示与颜色编码 (Requirements 7.3)
4. 方向指示器 (Requirements 7.4)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

# MediaPipe is optional - only needed for draw_pose_skeleton
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

if TYPE_CHECKING:
    from .pose_corrected_distance import BodyMeasurement, RelativeDistanceResult


# =============================================================================
# Color Constants
# =============================================================================

# 颜色定义 (BGR格式)
COLOR_GREEN = (0, 255, 0)       # 稳定状态
COLOR_ORANGE = (0, 165, 255)    # 显著变化
COLOR_RED = (0, 0, 255)         # 远离
COLOR_YELLOW = (0, 255, 255)    # 信息文本
COLOR_CYAN = (255, 255, 0)      # 测量数据
COLOR_WHITE = (255, 255, 255)   # 一般文本
COLOR_GRAY = (200, 200, 200)    # 次要信息
COLOR_PURPLE = (255, 128, 128)  # 提示信息

# 稳定范围阈值
STABLE_RATIO_MIN = 0.95
STABLE_RATIO_MAX = 1.05


# =============================================================================
# Task 13.1: Pose Skeleton Drawing (Requirements 7.1)
# =============================================================================

def draw_pose_skeleton(
    image: np.ndarray,
    pose_landmarks,
    landmark_color: Tuple[int, int, int] = COLOR_GREEN,
    connection_color: Tuple[int, int, int] = (0, 0, 255),
    landmark_thickness: int = 2,
    landmark_radius: int = 2,
    connection_thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制MediaPipe姿态骨架
    
    Args:
        image: 输入图像 (BGR格式)
        pose_landmarks: MediaPipe的pose_landmarks结果
        landmark_color: 关键点颜色 (BGR)
        connection_color: 连接线颜色 (BGR)
        landmark_thickness: 关键点线条粗细
        landmark_radius: 关键点圆圈半径
        connection_thickness: 连接线粗细
        
    Returns:
        绘制了骨架的图像
        
    **Validates: Requirements 7.1**
    """
    if pose_landmarks is None:
        return image
    
    if not MEDIAPIPE_AVAILABLE:
        # If mediapipe is not available, return the image unchanged
        return image
    
    output_image = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # 绘制骨架
    mp_drawing.draw_landmarks(
        output_image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=landmark_color,
            thickness=landmark_thickness,
            circle_radius=landmark_radius
        ),
        mp_drawing.DrawingSpec(
            color=connection_color,
            thickness=connection_thickness
        )
    )
    
    return output_image


# =============================================================================
# Task 13.2: Measurement Info Display (Requirements 7.2)
# =============================================================================

def draw_measurement_info(
    image: np.ndarray,
    measurement: 'BodyMeasurement',
    corrected_shoulder_width: float,
    corrected_hip_width: float,
    start_y: int = 30,
    line_height: int = 25,
    font_scale: float = 0.6
) -> Tuple[np.ndarray, int]:
    """
    在图像上显示测量信息
    
    显示内容：
    - 3D肩宽 (米)
    - 3D胯宽 (米)
    - Body_Yaw角度 (度)
    - 校正后的肩宽 (像素)
    - 校正后的胯宽 (像素)
    
    Args:
        image: 输入图像
        measurement: 身体测量数据
        corrected_shoulder_width: 校正后的肩宽 (像素)
        corrected_hip_width: 校正后的胯宽 (像素)
        start_y: 起始Y坐标
        line_height: 行高
        font_scale: 字体大小
        
    Returns:
        (绘制后的图像, 下一行Y坐标)
        
    **Validates: Requirements 7.2**
    """
    output_image = image.copy()
    panel_y = start_y
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 3D肩宽
    cv2.putText(
        output_image,
        f"3D Shoulder: {measurement.shoulder_width_3d:.3f}m",
        (10, panel_y),
        font,
        font_scale,
        COLOR_CYAN,
        2
    )
    panel_y += line_height
    
    # 3D胯宽
    cv2.putText(
        output_image,
        f"3D Hip: {measurement.hip_width_3d:.3f}m",
        (10, panel_y),
        font,
        font_scale,
        COLOR_CYAN,
        2
    )
    panel_y += line_height
    
    # Body_Yaw角度
    cv2.putText(
        output_image,
        f"Body Yaw: {measurement.body_yaw:.1f} deg",
        (10, panel_y),
        font,
        font_scale,
        COLOR_YELLOW,
        2
    )
    panel_y += line_height
    
    # 校正后的肩宽
    cv2.putText(
        output_image,
        f"Corrected Shoulder: {corrected_shoulder_width:.1f}px",
        (10, panel_y),
        font,
        font_scale,
        COLOR_GREEN,
        2
    )
    panel_y += line_height
    
    # 校正后的胯宽
    cv2.putText(
        output_image,
        f"Corrected Hip: {corrected_hip_width:.1f}px",
        (10, panel_y),
        font,
        font_scale,
        COLOR_GREEN,
        2
    )
    panel_y += line_height
    
    return output_image, panel_y


# =============================================================================
# Task 13.3: Distance Ratio Display with Color Coding (Requirements 7.3)
# =============================================================================

def get_ratio_color(
    ratio: float,
    stable_min: float = STABLE_RATIO_MIN,
    stable_max: float = STABLE_RATIO_MAX
) -> Tuple[int, int, int]:
    """
    根据距离比值获取颜色编码
    
    颜色规则：
    - 绿色: 稳定 (0.95 <= ratio <= 1.05)
    - 橙色: 显著变化 (ratio < 0.95 或 ratio > 1.05)
    
    Args:
        ratio: 相对距离比值
        stable_min: 稳定范围最小值 (默认0.95)
        stable_max: 稳定范围最大值 (默认1.05)
        
    Returns:
        颜色元组 (BGR格式)
        
    **Validates: Requirements 7.3**
    """
    if stable_min <= ratio <= stable_max:
        return COLOR_GREEN
    else:
        return COLOR_ORANGE


def draw_distance_ratio(
    image: np.ndarray,
    ratio: float,
    start_y: int,
    is_calibrated: bool = True,
    font_scale: float = 0.8
) -> Tuple[np.ndarray, int]:
    """
    在图像上显示距离比值，带颜色编码
    
    颜色编码：
    - 绿色: 稳定 (0.95-1.05)
    - 橙色: 显著变化
    
    Args:
        image: 输入图像
        ratio: 相对距离比值
        start_y: 起始Y坐标
        is_calibrated: 是否已校准
        font_scale: 字体大小
        
    Returns:
        (绘制后的图像, 下一行Y坐标)
        
    **Validates: Requirements 7.3**
    """
    output_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30
    
    if is_calibrated:
        # 获取颜色编码
        ratio_color = get_ratio_color(ratio)
        
        # 绘制距离比值
        cv2.putText(
            output_image,
            f"Distance Ratio: {ratio:.3f}",
            (10, start_y),
            font,
            font_scale,
            ratio_color,
            2
        )
        return output_image, start_y + line_height
    else:
        # 未校准时显示提示
        cv2.putText(
            output_image,
            "Press 'C' to calibrate",
            (10, start_y),
            font,
            0.6,
            COLOR_PURPLE,
            2
        )
        return output_image, start_y + line_height


# =============================================================================
# Task 13.4: Direction Indicator (Requirements 7.4)
# =============================================================================

def get_direction_text_and_color(
    ratio: float,
    stable_min: float = STABLE_RATIO_MIN,
    stable_max: float = STABLE_RATIO_MAX
) -> Tuple[str, Tuple[int, int, int]]:
    """
    根据距离比值获取方向指示文本和颜色
    
    规则：
    - ratio > stable_max (1.05): "MOVING AWAY" (红色)
    - ratio < stable_min (0.95): "APPROACHING" (绿色)
    - stable_min <= ratio <= stable_max: "STABLE" (黄色)
    
    Args:
        ratio: 相对距离比值
        stable_min: 稳定范围最小值
        stable_max: 稳定范围最大值
        
    Returns:
        (方向文本, 颜色元组)
        
    **Validates: Requirements 7.4**
    """
    if ratio > stable_max:
        return "MOVING AWAY", COLOR_RED
    elif ratio < stable_min:
        return "APPROACHING", COLOR_GREEN
    else:
        return "STABLE", COLOR_YELLOW


def draw_direction_indicator(
    image: np.ndarray,
    ratio: float,
    start_y: int,
    is_calibrated: bool = True,
    font_scale: float = 0.8
) -> Tuple[np.ndarray, int]:
    """
    在图像上显示方向指示器
    
    显示内容：
    - APPROACHING: 靠近 (绿色)
    - MOVING AWAY: 远离 (红色)
    - STABLE: 稳定 (黄色)
    
    Args:
        image: 输入图像
        ratio: 相对距离比值
        start_y: 起始Y坐标
        is_calibrated: 是否已校准
        font_scale: 字体大小
        
    Returns:
        (绘制后的图像, 下一行Y坐标)
        
    **Validates: Requirements 7.4**
    """
    output_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30
    
    if not is_calibrated:
        return output_image, start_y
    
    # 获取方向文本和颜色
    direction_text, color = get_direction_text_and_color(ratio)
    
    # 绘制方向指示
    cv2.putText(
        output_image,
        direction_text,
        (10, start_y),
        font,
        font_scale,
        color,
        2
    )
    
    return output_image, start_y + line_height


# =============================================================================
# Additional Visualization Helpers
# =============================================================================

def draw_confidence(
    image: np.ndarray,
    confidence: float,
    start_y: int,
    font_scale: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    显示置信度
    
    Args:
        image: 输入图像
        confidence: 置信度 (0-1)
        start_y: 起始Y坐标
        font_scale: 字体大小
        
    Returns:
        (绘制后的图像, 下一行Y坐标)
    """
    output_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    
    cv2.putText(
        output_image,
        f"Confidence: {confidence:.2f}",
        (10, start_y),
        font,
        font_scale,
        COLOR_GRAY,
        1
    )
    
    return output_image, start_y + line_height


def draw_orientation_indicator(
    image: np.ndarray,
    is_frontal: bool,
    start_y: int,
    font_scale: float = 0.6
) -> Tuple[np.ndarray, int]:
    """
    显示正面/侧面指示
    
    Args:
        image: 输入图像
        is_frontal: 是否正面
        start_y: 起始Y坐标
        font_scale: 字体大小
        
    Returns:
        (绘制后的图像, 下一行Y坐标)
    """
    output_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 25
    
    orientation_text = "FRONTAL" if is_frontal else "SIDE VIEW"
    orientation_color = COLOR_GREEN if is_frontal else COLOR_ORANGE
    
    cv2.putText(
        output_image,
        orientation_text,
        (10, start_y),
        font,
        font_scale,
        orientation_color,
        2
    )
    
    return output_image, start_y + line_height


# =============================================================================
# Complete Visualization Function
# =============================================================================

def draw_complete_visualization(
    image: np.ndarray,
    pose_landmarks,
    measurement: 'BodyMeasurement',
    result: 'RelativeDistanceResult',
    is_calibrated: bool
) -> np.ndarray:
    """
    绘制完整的可视化信息
    
    包含：
    1. 姿态骨架 (Requirements 7.1)
    2. 测量信息 (Requirements 7.2)
    3. 距离比值与颜色编码 (Requirements 7.3)
    4. 方向指示器 (Requirements 7.4)
    
    Args:
        image: 输入图像
        pose_landmarks: MediaPipe姿态关键点
        measurement: 身体测量数据
        result: 相对距离结果
        is_calibrated: 是否已校准
        
    Returns:
        绘制了所有可视化信息的图像
        
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    # 1. 绘制姿态骨架 (Task 13.1)
    output_image = draw_pose_skeleton(image, pose_landmarks)
    
    # 2. 绘制测量信息 (Task 13.2)
    output_image, panel_y = draw_measurement_info(
        output_image,
        measurement,
        result.corrected_shoulder_width,
        result.corrected_hip_width
    )
    
    # 3. 绘制距离比值 (Task 13.3)
    output_image, panel_y = draw_distance_ratio(
        output_image,
        result.relative_distance_ratio,
        panel_y,
        is_calibrated
    )
    
    # 4. 绘制方向指示器 (Task 13.4)
    output_image, panel_y = draw_direction_indicator(
        output_image,
        result.relative_distance_ratio,
        panel_y,
        is_calibrated
    )
    
    # 5. 绘制置信度
    output_image, panel_y = draw_confidence(
        output_image,
        result.confidence,
        panel_y
    )
    
    # 6. 绘制正面/侧面指示
    output_image, panel_y = draw_orientation_indicator(
        output_image,
        result.is_frontal,
        panel_y
    )
    
    return output_image
