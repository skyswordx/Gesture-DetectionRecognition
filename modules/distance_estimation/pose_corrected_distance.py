"""
基于MediaPipe 3D世界坐标的姿态校正距离估算模块

核心思路：
1. 使用MediaPipe的pose_world_landmarks获取真实3D世界坐标（以髋部中心为原点，单位为米）
2. 从3D世界坐标计算真实的肩宽和胯宽（不受姿态角度影响）
3. 通过比较3D真实宽度与2D投影宽度的比值，推算距离的相对变化
4. 利用肩宽/胯宽的稳定性作为距离变化的参考基准

关键点：
- pose_world_landmarks: 以髋部中心为原点的3D坐标，单位为米，不受相机位置影响
- pose_landmarks: 归一化的2D图像坐标 + 相对深度z
"""

import numpy as np
import cv2
import mediapipe as mp
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def euclidean_distance_3d(p1: 'Landmark3D', p2: 'Landmark3D') -> float:
    """
    计算两个3D关键点之间的欧氏距离
    
    公式: sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
    
    Args:
        p1: 第一个3D关键点
        p2: 第二个3D关键点
        
    Returns:
        两点之间的欧氏距离 (米)
        
    **Validates: Requirements 1.2**
    """
    return math.sqrt(
        (p2.x - p1.x) ** 2 + 
        (p2.y - p1.y) ** 2 + 
        (p2.z - p1.z) ** 2
    )


# =============================================================================
# Body Measurement Calculation Functions (Task 3)
# =============================================================================

def calculate_shoulder_width_3d(
    left_shoulder: 'Landmark3D', 
    right_shoulder: 'Landmark3D'
) -> float:
    """
    计算3D肩宽（从3D世界坐标）
    
    使用欧氏距离计算左右肩膀之间的真实3D距离。
    这个值不受姿态角度影响，表示真实的肩宽（单位：米）。
    
    Args:
        left_shoulder: 左肩3D关键点
        right_shoulder: 右肩3D关键点
        
    Returns:
        3D肩宽 (米)
        
    **Validates: Requirements 1.2**
    """
    return euclidean_distance_3d(left_shoulder, right_shoulder)


def calculate_hip_width_3d(
    left_hip: 'Landmark3D', 
    right_hip: 'Landmark3D'
) -> float:
    """
    计算3D胯宽（从3D世界坐标）
    
    使用欧氏距离计算左右髋部之间的真实3D距离。
    这个值不受姿态角度影响，表示真实的胯宽（单位：米）。
    
    Args:
        left_hip: 左髋3D关键点
        right_hip: 右髋3D关键点
        
    Returns:
        3D胯宽 (米)
        
    **Validates: Requirements 1.2**
    """
    return euclidean_distance_3d(left_hip, right_hip)


def calculate_shoulder_width_2d(
    left_shoulder: 'Landmark2D', 
    right_shoulder: 'Landmark2D',
    frame_width: int
) -> float:
    """
    计算2D肩宽（像素）
    
    从归一化的2D坐标计算肩宽的像素距离。
    注意：这个值会受到姿态角度的影响（侧身时会变小）。
    
    Args:
        left_shoulder: 左肩2D关键点（归一化坐标）
        right_shoulder: 右肩2D关键点（归一化坐标）
        frame_width: 图像宽度（像素）
        
    Returns:
        2D肩宽 (像素)
        
    **Validates: Requirements 3.1**
    """
    return abs(left_shoulder.x - right_shoulder.x) * frame_width


def calculate_hip_width_2d(
    left_hip: 'Landmark2D', 
    right_hip: 'Landmark2D',
    frame_width: int
) -> float:
    """
    计算2D胯宽（像素）
    
    从归一化的2D坐标计算胯宽的像素距离。
    注意：这个值会受到姿态角度的影响（侧身时会变小）。
    
    Args:
        left_hip: 左髋2D关键点（归一化坐标）
        right_hip: 右髋2D关键点（归一化坐标）
        frame_width: 图像宽度（像素）
        
    Returns:
        2D胯宽 (像素)
        
    **Validates: Requirements 3.1**
    """
    return abs(left_hip.x - right_hip.x) * frame_width


def compute_body_measurement(
    landmarks_2d: List['Landmark2D'],
    landmarks_3d: List['Landmark3D'],
    frame_width: int,
    frame_height: int,
    left_shoulder_idx: int = 11,
    right_shoulder_idx: int = 12,
    left_hip_idx: int = 23,
    right_hip_idx: int = 24
) -> 'BodyMeasurement':
    """
    计算身体测量数据
    
    从MediaPipe提取的2D和3D关键点计算身体测量值，包括：
    - 3D肩宽和胯宽（真实尺寸，单位：米）
    - 2D肩宽和胯宽（像素）
    - 基于可见性的置信度
    
    关键：使用3D世界坐标计算真实尺寸，不受姿态影响。
    使用可见性过滤来排除不可靠的关键点。
    
    Args:
        landmarks_2d: 2D关键点列表
        landmarks_3d: 3D关键点列表
        frame_width: 图像宽度（像素）
        frame_height: 图像高度（像素）
        left_shoulder_idx: 左肩关键点索引（默认11）
        right_shoulder_idx: 右肩关键点索引（默认12）
        left_hip_idx: 左髋关键点索引（默认23）
        right_hip_idx: 右髋关键点索引（默认24）
        
    Returns:
        BodyMeasurement数据结构
        
    **Validates: Requirements 1.2, 1.3**
    """
    measurement = BodyMeasurement(timestamp=time.time())
    
    # 获取关键点
    ls_2d = landmarks_2d[left_shoulder_idx]
    rs_2d = landmarks_2d[right_shoulder_idx]
    lh_2d = landmarks_2d[left_hip_idx]
    rh_2d = landmarks_2d[right_hip_idx]
    
    ls_3d = landmarks_3d[left_shoulder_idx]
    rs_3d = landmarks_3d[right_shoulder_idx]
    lh_3d = landmarks_3d[left_hip_idx]
    rh_3d = landmarks_3d[right_hip_idx]
    
    # 使用可见性过滤函数检查关键点可见性
    shoulder_visible, shoulder_vis = get_landmark_pair_visibility(ls_2d, rs_2d)
    hip_visible, hip_vis = get_landmark_pair_visibility(lh_2d, rh_2d)
    
    # 如果肩膀和髋部都不可见，返回低置信度结果
    if not shoulder_visible and not hip_visible:
        measurement.confidence = 0.0
        return measurement
    
    # 计算3D肩宽（欧氏距离，单位：米）- 仅当肩膀可见时
    if shoulder_visible:
        measurement.shoulder_width_3d = calculate_shoulder_width_3d(ls_3d, rs_3d)
        measurement.shoulder_width_2d = calculate_shoulder_width_2d(ls_2d, rs_2d, frame_width)
    
    # 计算3D胯宽 - 仅当髋部可见时
    if hip_visible:
        measurement.hip_width_3d = calculate_hip_width_3d(lh_3d, rh_3d)
        measurement.hip_width_2d = calculate_hip_width_2d(lh_2d, rh_2d, frame_width)
    
    # 计算躯干高度（肩膀中心到髋部中心）- 仅当两者都可见时
    if shoulder_visible and hip_visible:
        shoulder_center_x = (ls_3d.x + rs_3d.x) / 2
        shoulder_center_y = (ls_3d.y + rs_3d.y) / 2
        shoulder_center_z = (ls_3d.z + rs_3d.z) / 2
        
        hip_center_x = (lh_3d.x + rh_3d.x) / 2
        hip_center_y = (lh_3d.y + rh_3d.y) / 2
        hip_center_z = (lh_3d.z + rh_3d.z) / 2
        
        shoulder_center = Landmark3D(shoulder_center_x, shoulder_center_y, shoulder_center_z, 1.0)
        hip_center = Landmark3D(hip_center_x, hip_center_y, hip_center_z, 1.0)
        measurement.torso_height_3d = euclidean_distance_3d(shoulder_center, hip_center)
    
    # 基于可见关键点计算置信度
    # 置信度 = 可见关键点对的平均可见性
    measurement.confidence = (shoulder_vis + hip_vis) / 2
    
    return measurement


# =============================================================================
# Body Orientation Estimation Functions (Task 5)
# =============================================================================

def estimate_body_yaw(left_shoulder: 'Landmark3D', right_shoulder: 'Landmark3D') -> float:
    """
    估算身体偏航角（左右转动）
    
    原理：当人侧身时，左右肩膀的z坐标会有差异。
    使用atan2(dz, dx)计算肩膀连线在xz平面的角度。
    
    Args:
        left_shoulder: 左肩3D关键点
        right_shoulder: 右肩3D关键点
        
    Returns:
        角度（度），正值表示向右转，负值表示向左转
        
    **Validates: Requirements 2.1, 2.4**
    """
    dx = right_shoulder.x - left_shoulder.x
    dz = right_shoulder.z - left_shoulder.z
    
    # 计算偏航角
    yaw = math.atan2(dz, dx)
    return math.degrees(yaw)


def classify_pose_orientation(body_yaw: float) -> bool:
    """
    根据身体偏航角分类姿态为正面或侧面
    
    分类规则：
    - 正面 (frontal): |yaw| < 30°
    - 侧面 (side view): |yaw| >= 30°
    
    Args:
        body_yaw: 身体偏航角（度）
        
    Returns:
        True 如果是正面，False 如果是侧面
        
    **Validates: Requirements 2.2**
    """
    return abs(body_yaw) < 30.0


def adjust_confidence_for_extreme_yaw(confidence: float, body_yaw: float) -> float:
    """
    根据极端偏航角调整置信度
    
    当|yaw| > 70°时，测量变得不可靠，需要降低置信度。
    
    Args:
        confidence: 原始置信度 (0-1)
        body_yaw: 身体偏航角（度）
        
    Returns:
        调整后的置信度
        
    **Validates: Requirements 2.3**
    """
    if abs(body_yaw) > 70.0:
        return confidence * 0.5
    return confidence


# =============================================================================
# Pose Correction Functions (Task 6)
# =============================================================================

def compute_corrected_width(width_2d: float, body_yaw: float) -> float:
    """
    计算校正后的宽度
    
    公式：Corrected_Width = 2D_Width / |cos(yaw)|
    
    当人侧身时，2D投影宽度会缩小，除以cos(yaw)可以恢复到正面时的宽度。
    为防止除零，当cos(yaw)小于0.1时，将其钳制到0.1。
    
    Args:
        width_2d: 2D宽度（像素）
        body_yaw: 身体偏航角（度）
        
    Returns:
        校正后的宽度（像素）
        
    **Validates: Requirements 3.1, 3.2**
    """
    yaw_rad = math.radians(body_yaw)
    cos_yaw = max(0.1, abs(math.cos(yaw_rad)))  # 防止除零，钳制到最小0.1
    return width_2d / cos_yaw


def compute_corrected_shoulder_width(shoulder_width_2d: float, body_yaw: float) -> float:
    """
    计算校正后的肩宽
    
    Args:
        shoulder_width_2d: 2D肩宽（像素）
        body_yaw: 身体偏航角（度）
        
    Returns:
        校正后的肩宽（像素）
        
    **Validates: Requirements 3.1, 3.2, 3.4**
    """
    return compute_corrected_width(shoulder_width_2d, body_yaw)


def compute_corrected_hip_width(hip_width_2d: float, body_yaw: float) -> float:
    """
    计算校正后的胯宽
    
    Args:
        hip_width_2d: 2D胯宽（像素）
        body_yaw: 身体偏航角（度）
        
    Returns:
        校正后的胯宽（像素）
        
    **Validates: Requirements 3.1, 3.2, 3.4**
    """
    return compute_corrected_width(hip_width_2d, body_yaw)


# =============================================================================
# Relative Distance Calculation Functions (Task 9)
# =============================================================================

def compute_relative_distance_ratio(
    corrected_width_ref: float,
    corrected_width_current: float
) -> float:
    """
    计算相对距离比值
    
    基于针孔相机模型：
    - 2D_Width = 3D_Width × f / Z
    - 因此：Z_current / Z_ref = Corrected_Width_ref / Corrected_Width_current
    
    公式：ratio = Corrected_Width_ref / Corrected_Width_current
    
    Args:
        corrected_width_ref: 参考帧的校正后宽度
        corrected_width_current: 当前帧的校正后宽度
        
    Returns:
        相对距离比值，>1表示远离，<1表示靠近，=1表示稳定
        
    **Validates: Requirements 5.1**
    """
    if corrected_width_current <= 0:
        return 1.0  # 防止除零，返回稳定状态
    return corrected_width_ref / corrected_width_current


def get_direction_indication(relative_distance_ratio: float) -> str:
    """
    根据相对距离比值获取方向指示
    
    规则：
    - ratio > 1.0: 远离 (MOVING AWAY)
    - ratio < 1.0: 靠近 (APPROACHING)
    - ratio == 1.0: 稳定 (STABLE)
    
    Args:
        relative_distance_ratio: 相对距离比值
        
    Returns:
        方向指示字符串: "MOVING AWAY", "APPROACHING", 或 "STABLE"
        
    **Validates: Requirements 5.2, 5.3**
    """
    if relative_distance_ratio > 1.0:
        return "MOVING AWAY"
    elif relative_distance_ratio < 1.0:
        return "APPROACHING"
    else:
        return "STABLE"


def compute_weighted_ratio_combination(
    shoulder_ratio: float,
    hip_ratio: float,
    shoulder_weight: float = 0.6,
    hip_weight: float = 0.4
) -> float:
    """
    计算加权融合的距离比值
    
    使用肩宽和胯宽的比值进行加权融合，默认权重为：
    - 肩宽权重: 0.6
    - 胯宽权重: 0.4
    
    公式：combined_ratio = shoulder_weight × shoulder_ratio + hip_weight × hip_ratio
    
    Args:
        shoulder_ratio: 基于肩宽的距离比值
        hip_ratio: 基于胯宽的距离比值
        shoulder_weight: 肩宽权重，默认0.6
        hip_weight: 胯宽权重，默认0.4
        
    Returns:
        加权融合后的距离比值
        
    **Validates: Requirements 5.4**
    """
    return shoulder_weight * shoulder_ratio + hip_weight * hip_ratio


# =============================================================================
# Calibration and Reference Frame Management Functions (Task 8)
# =============================================================================

def validate_calibration(measurement: 'BodyMeasurement', min_confidence: float = 0.5) -> bool:
    """
    验证测量数据是否适合用于校准
    
    Args:
        measurement: 要验证的测量数据
        min_confidence: 最小置信度阈值，默认0.5
        
    Returns:
        bool: True 如果测量数据有效（置信度 >= min_confidence），否则 False
        
    **Validates: Requirements 4.3**
    """
    if measurement is None:
        return False
    return measurement.confidence >= min_confidence


def store_reference_frame(
    measurement: 'BodyMeasurement',
    current_reference: 'BodyMeasurement',
    is_calibrated: bool,
    min_confidence: float = 0.5
) -> tuple:
    """
    存储参考帧，如果校准验证通过
    
    如果置信度 < min_confidence，则拒绝校准并保持之前的状态。
    
    Args:
        measurement: 要存储为参考帧的测量数据
        current_reference: 当前的参考帧（用于在拒绝时保持状态）
        is_calibrated: 当前的校准状态
        min_confidence: 最小置信度阈值
        
    Returns:
        tuple: (new_reference, new_is_calibrated, success)
            - new_reference: 新的参考帧（如果成功）或保持原来的
            - new_is_calibrated: 新的校准状态
            - success: 是否成功存储
            
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    # 验证校准
    if not validate_calibration(measurement, min_confidence):
        # 拒绝校准，保持之前的状态
        return current_reference, is_calibrated, False
    
    # 校准成功：存储参考帧并设置标志
    return measurement, True, True


def reset_calibration() -> tuple:
    """
    重置校准状态
    
    Returns:
        tuple: (reference_frame, is_calibrated)
            - reference_frame: None（已清除）
            - is_calibrated: False
            
    **Validates: Requirements 4.4**
    """
    return None, False


# =============================================================================
# Landmark Extraction Functions (Task 2.1)
# =============================================================================

def extract_landmark_2d(mediapipe_landmark) -> 'Landmark2D':
    """
    从MediaPipe关键点提取2D坐标
    
    Args:
        mediapipe_landmark: MediaPipe的单个landmark对象
        
    Returns:
        Landmark2D数据结构
        
    **Validates: Requirements 1.1, 1.4**
    """
    return Landmark2D(
        x=mediapipe_landmark.x,
        y=mediapipe_landmark.y,
        z=mediapipe_landmark.z,
        visibility=mediapipe_landmark.visibility
    )


def extract_landmark_3d(mediapipe_landmark) -> 'Landmark3D':
    """
    从MediaPipe世界坐标关键点提取3D坐标
    
    Args:
        mediapipe_landmark: MediaPipe的单个world_landmark对象
        
    Returns:
        Landmark3D数据结构
        
    **Validates: Requirements 1.1, 1.4**
    """
    return Landmark3D(
        x=mediapipe_landmark.x,
        y=mediapipe_landmark.y,
        z=mediapipe_landmark.z,
        visibility=mediapipe_landmark.visibility
    )


def extract_all_landmarks_2d(pose_landmarks) -> List['Landmark2D']:
    """
    从MediaPipe结果提取所有2D关键点
    
    Args:
        pose_landmarks: MediaPipe的pose_landmarks结果
        
    Returns:
        Landmark2D列表
        
    **Validates: Requirements 1.1, 1.4**
    """
    return [extract_landmark_2d(lm) for lm in pose_landmarks.landmark]


def extract_all_landmarks_3d(pose_world_landmarks) -> List['Landmark3D']:
    """
    从MediaPipe结果提取所有3D世界坐标关键点
    
    Args:
        pose_world_landmarks: MediaPipe的pose_world_landmarks结果
        
    Returns:
        Landmark3D列表
        
    **Validates: Requirements 1.1, 1.4**
    """
    return [extract_landmark_3d(lm) for lm in pose_world_landmarks.landmark]


# =============================================================================
# Visibility Filtering Functions (Task 2.2)
# =============================================================================

VISIBILITY_THRESHOLD = 0.5


def is_landmark_visible(landmark: 'Landmark3D', threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """
    检查关键点是否可见
    
    Args:
        landmark: 3D或2D关键点
        threshold: 可见性阈值，默认0.5
        
    Returns:
        True如果可见性 >= threshold，否则False
        
    **Validates: Requirements 1.3**
    """
    return landmark.visibility >= threshold


def is_landmark_2d_visible(landmark: 'Landmark2D', threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """
    检查2D关键点是否可见
    
    Args:
        landmark: 2D关键点
        threshold: 可见性阈值，默认0.5
        
    Returns:
        True如果可见性 >= threshold，否则False
        
    **Validates: Requirements 1.3**
    """
    return landmark.visibility >= threshold


def filter_visible_landmarks_3d(
    landmarks: List['Landmark3D'], 
    threshold: float = VISIBILITY_THRESHOLD
) -> List['Landmark3D']:
    """
    过滤出可见的3D关键点
    
    Args:
        landmarks: 3D关键点列表
        threshold: 可见性阈值
        
    Returns:
        可见性 >= threshold的关键点列表
        
    **Validates: Requirements 1.3**
    """
    return [lm for lm in landmarks if lm.visibility >= threshold]


def filter_visible_landmarks_2d(
    landmarks: List['Landmark2D'], 
    threshold: float = VISIBILITY_THRESHOLD
) -> List['Landmark2D']:
    """
    过滤出可见的2D关键点
    
    Args:
        landmarks: 2D关键点列表
        threshold: 可见性阈值
        
    Returns:
        可见性 >= threshold的关键点列表
        
    **Validates: Requirements 1.3**
    """
    return [lm for lm in landmarks if lm.visibility >= threshold]


def compute_measurement_confidence(
    landmarks_2d: List['Landmark2D'],
    required_indices: List[int],
    threshold: float = VISIBILITY_THRESHOLD
) -> float:
    """
    基于关键点可见性计算测量置信度
    
    置信度 = 可见关键点数 / 所需关键点数
    
    Args:
        landmarks_2d: 2D关键点列表
        required_indices: 所需关键点的索引列表
        threshold: 可见性阈值
        
    Returns:
        置信度 (0-1)
        
    **Validates: Requirements 1.3**
    """
    if not required_indices:
        return 0.0
    
    visible_count = sum(
        1 for idx in required_indices 
        if idx < len(landmarks_2d) and landmarks_2d[idx].visibility >= threshold
    )
    
    return visible_count / len(required_indices)


def get_landmark_pair_visibility(
    landmark1: 'Landmark2D',
    landmark2: 'Landmark2D',
    threshold: float = VISIBILITY_THRESHOLD
) -> Tuple[bool, float]:
    """
    获取一对关键点的可见性状态和置信度
    
    Args:
        landmark1: 第一个关键点
        landmark2: 第二个关键点
        threshold: 可见性阈值
        
    Returns:
        (both_visible, min_visibility) - 是否都可见，以及最小可见性值
        
    **Validates: Requirements 1.3**
    """
    min_vis = min(landmark1.visibility, landmark2.visibility)
    both_visible = min_vis >= threshold
    return both_visible, min_vis


@dataclass
class Landmark3D:
    """
    3D世界坐标关键点数据结构
    
    MediaPipe的pose_world_landmarks输出，以髋部中心为原点，单位为米。
    这些坐标不受相机位置影响，表示真实的3D空间位置。
    
    Attributes:
        x: 世界坐标x (米)
        y: 世界坐标y (米)  
        z: 世界坐标z (米)
        visibility: 可见性 (0-1)，低于0.5表示不可靠
    """
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class Landmark2D:
    """
    2D图像坐标关键点数据结构
    
    MediaPipe的pose_landmarks输出，归一化的2D图像坐标加相对深度。
    
    Attributes:
        x: 归一化x坐标 (0-1)
        y: 归一化y坐标 (0-1)
        z: 相对深度（相对于髋部）
        visibility: 可见性 (0-1)
    """
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class BodyMeasurement:
    """
    身体测量数据
    
    包含从MediaPipe提取的3D和2D身体测量值，以及姿态角度估算。
    
    Attributes:
        shoulder_width_3d: 3D肩宽 (米)，从pose_world_landmarks计算
        hip_width_3d: 3D胯宽 (米)
        torso_height_3d: 3D躯干高度 (米)
        shoulder_width_2d: 2D肩宽 (像素)
        hip_width_2d: 2D胯宽 (像素)
        body_yaw: 身体偏航角 (度)，正值表示向右转
        body_pitch: 身体俯仰角 (度)
        confidence: 测量置信度 (0-1)
        timestamp: 时间戳
    """
    # 3D世界坐标测量（真实尺寸，单位：米）
    shoulder_width_3d: float = 0.0
    hip_width_3d: float = 0.0
    torso_height_3d: float = 0.0
    
    # 2D投影测量（像素）
    shoulder_width_2d: float = 0.0
    hip_width_2d: float = 0.0
    
    # 姿态角度估算
    body_yaw: float = 0.0
    body_pitch: float = 0.0
    
    # 置信度
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class RelativeDistanceResult:
    """
    相对距离测量结果
    
    包含相对于参考帧的距离变化比值和校正后的测量值。
    
    Attributes:
        relative_distance_ratio: 相对距离比值，>1表示远离，<1表示靠近
        shoulder_ratio: 基于肩宽的距离比值
        hip_ratio: 基于胯宽的距离比值
        combined_ratio: 加权融合比值 (肩宽0.6 + 胯宽0.4)
        corrected_shoulder_width: 校正后的肩宽 (像素)，消除姿态影响
        corrected_hip_width: 校正后的胯宽 (像素)
        body_orientation: 身体朝向角度 (度)
        is_frontal: 是否正面朝向 (|yaw| < 30°)
        confidence: 测量置信度 (0-1)
        method: 使用的测量方法
        raw_measurement: 原始测量数据
    """
    # 相对距离比值（相对于参考帧，>1表示远离，<1表示靠近）
    relative_distance_ratio: float = 1.0
    
    # 基于不同测量的比值
    shoulder_ratio: float = 1.0
    hip_ratio: float = 1.0
    combined_ratio: float = 1.0
    
    # 校正后的测量值
    corrected_shoulder_width: float = 0.0
    corrected_hip_width: float = 0.0
    
    # 姿态信息
    body_orientation: float = 0.0
    is_frontal: bool = True
    
    # 置信度和元数据
    confidence: float = 0.0
    method: str = ""
    raw_measurement: Optional[BodyMeasurement] = None


class PoseCorrectedDistanceEstimator:
    """基于姿态校正的距离估算器"""
    
    # MediaPipe关键点索引
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_EAR = 7
    RIGHT_EAR = 8
    NOSE = 0
    
    def __init__(self, 
                 reference_shoulder_width: float = 0.40,  # 参考肩宽（米）
                 reference_hip_width: float = 0.30,       # 参考胯宽（米）
                 smoothing_window: int = 5):
        """
        初始化估算器
        
        Args:
            reference_shoulder_width: 参考肩宽（用于绝对距离估算）
            reference_hip_width: 参考胯宽
            smoothing_window: 平滑窗口大小
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 使用最高精度模型
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 参考测量值
        self.reference_shoulder_width = reference_shoulder_width
        self.reference_hip_width = reference_hip_width
        
        # 基准帧数据（用于计算相对变化）
        self.reference_measurement: Optional[BodyMeasurement] = None
        self.is_calibrated = False
        
        # 历史数据用于平滑
        self.measurement_history: deque = deque(maxlen=smoothing_window)
        self.ratio_history: deque = deque(maxlen=smoothing_window)
        
        # 统计
        self.frame_count = 0
        
    def process_frame(self, image: np.ndarray) -> Tuple[Optional[RelativeDistanceResult], np.ndarray]:
        """
        处理单帧图像
        
        Args:
            image: BGR图像
            
        Returns:
            (距离结果, 标注后的图像)
        """
        if image is None:
            return None, image
            
        self.frame_count += 1
        height, width = image.shape[:2]
        
        # 转换颜色空间
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        output_image = image.copy()
        
        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None, output_image
        
        # 提取2D和3D关键点
        landmarks_2d = self._extract_landmarks_2d(results.pose_landmarks)
        landmarks_3d = self._extract_landmarks_3d(results.pose_world_landmarks)
        
        # 计算身体测量
        measurement = self._compute_body_measurement(
            landmarks_2d, landmarks_3d, width, height
        )
        
        if measurement.confidence < 0.5:
            return None, output_image
        
        # 添加到历史
        self.measurement_history.append(measurement)
        
        # 计算相对距离
        result = self._compute_relative_distance(measurement)
        
        # 绘制可视化
        output_image = self._draw_visualization(
            output_image, results.pose_landmarks, measurement, result
        )
        
        return result, output_image

    def _extract_landmarks_2d(self, pose_landmarks) -> List[Landmark2D]:
        """
        提取2D关键点
        
        **Validates: Requirements 1.1, 1.4**
        """
        return extract_all_landmarks_2d(pose_landmarks)
    
    def _extract_landmarks_3d(self, pose_world_landmarks) -> List[Landmark3D]:
        """
        提取3D世界坐标关键点
        
        **Validates: Requirements 1.1, 1.4**
        """
        return extract_all_landmarks_3d(pose_world_landmarks)
    
    def _compute_body_measurement(self, 
                                   landmarks_2d: List[Landmark2D],
                                   landmarks_3d: List[Landmark3D],
                                   frame_width: int,
                                   frame_height: int) -> BodyMeasurement:
        """
        计算身体测量数据
        
        关键：使用3D世界坐标计算真实尺寸，不受姿态影响
        使用可见性过滤来排除不可靠的关键点
        
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        # 使用独立函数计算基本测量
        measurement = compute_body_measurement(
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            frame_width=frame_width,
            frame_height=frame_height,
            left_shoulder_idx=self.LEFT_SHOULDER,
            right_shoulder_idx=self.RIGHT_SHOULDER,
            left_hip_idx=self.LEFT_HIP,
            right_hip_idx=self.RIGHT_HIP
        )
        
        # 如果置信度为0，直接返回
        if measurement.confidence == 0.0:
            return measurement
        
        # 获取肩膀关键点用于姿态估算
        ls_2d = landmarks_2d[self.LEFT_SHOULDER]
        rs_2d = landmarks_2d[self.RIGHT_SHOULDER]
        ls_3d = landmarks_3d[self.LEFT_SHOULDER]
        rs_3d = landmarks_3d[self.RIGHT_SHOULDER]
        
        # 检查肩膀可见性
        shoulder_visible, _ = get_landmark_pair_visibility(ls_2d, rs_2d)
        
        # 估算身体朝向角度（基于肩膀的z坐标差异）
        if shoulder_visible:
            measurement.body_yaw = self._estimate_body_yaw(ls_3d, rs_3d)
        measurement.body_pitch = self._estimate_body_pitch(landmarks_3d)
        
        return measurement
    
    def _euclidean_distance_3d(self, p1: Landmark3D, p2: Landmark3D) -> float:
        """计算3D欧氏距离"""
        return euclidean_distance_3d(p1, p2)
    
    def _midpoint_3d(self, p1: Landmark3D, p2: Landmark3D) -> Tuple[float, float, float]:
        """计算3D中点"""
        return (
            (p1.x + p2.x) / 2,
            (p1.y + p2.y) / 2,
            (p1.z + p2.z) / 2
        )
    
    def _estimate_body_yaw(self, left_shoulder: Landmark3D, right_shoulder: Landmark3D) -> float:
        """
        估算身体偏航角（左右转动）
        
        原理：当人侧身时，左右肩膀的z坐标会有差异
        返回：角度（度），正值表示向右转，负值表示向左转
        
        **Validates: Requirements 2.1, 2.4**
        """
        return estimate_body_yaw(left_shoulder, right_shoulder)
    
    def _estimate_body_pitch(self, landmarks_3d: List[Landmark3D]) -> float:
        """估算身体俯仰角（前后倾斜）"""
        # 使用肩膀和髋部的y坐标差异
        shoulder_center_y = (landmarks_3d[self.LEFT_SHOULDER].y + 
                            landmarks_3d[self.RIGHT_SHOULDER].y) / 2
        hip_center_y = (landmarks_3d[self.LEFT_HIP].y + 
                       landmarks_3d[self.RIGHT_HIP].y) / 2
        
        shoulder_center_z = (landmarks_3d[self.LEFT_SHOULDER].z + 
                            landmarks_3d[self.RIGHT_SHOULDER].z) / 2
        hip_center_z = (landmarks_3d[self.LEFT_HIP].z + 
                       landmarks_3d[self.RIGHT_HIP].z) / 2
        
        dy = shoulder_center_y - hip_center_y
        dz = shoulder_center_z - hip_center_z
        
        pitch = math.atan2(dz, abs(dy))
        return math.degrees(pitch)
    
    def set_reference_frame(self, measurement: Optional[BodyMeasurement] = None) -> bool:
        """
        设置参考帧（用于计算相对距离变化）
        
        校准验证：如果置信度 < 0.5，则拒绝校准并保持之前的状态。
        
        Args:
            measurement: 如果为None，使用当前最新测量作为参考
            
        Returns:
            bool: True 如果校准成功，False 如果校准被拒绝
            
        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        # 确定要使用的测量数据
        candidate_measurement = measurement
        if candidate_measurement is None and self.measurement_history:
            candidate_measurement = self.measurement_history[-1]
        
        # 如果没有可用的测量数据，拒绝校准
        if candidate_measurement is None:
            logger.warning("校准失败: 没有可用的测量数据")
            return False
        
        # 验证置信度 - 如果 < 0.5，拒绝校准并保持之前的状态
        if candidate_measurement.confidence < 0.5:
            logger.warning(f"校准被拒绝: 置信度 {candidate_measurement.confidence:.2f} < 0.5，保持之前的状态")
            return False
        
        # 校准成功：存储参考帧并设置标志
        self.reference_measurement = candidate_measurement
        self.is_calibrated = True
        
        logger.info(f"参考帧已设置: 肩宽3D={self.reference_measurement.shoulder_width_3d:.3f}m, "
                   f"胯宽3D={self.reference_measurement.hip_width_3d:.3f}m, "
                   f"置信度={self.reference_measurement.confidence:.2f}")
        return True
    
    def _compute_relative_distance(self, measurement: BodyMeasurement) -> RelativeDistanceResult:
        """
        计算相对距离变化
        
        核心算法：
        1. 3D世界坐标中的肩宽/胯宽是恒定的（不受姿态影响）
        2. 2D投影宽度 = 3D真实宽度 * f / Z（f是焦距，Z是距离）
        3. 因此：Z1/Z2 = (3D宽度/2D宽度1) / (3D宽度/2D宽度2) = 2D宽度2 / 2D宽度1
        
        但是！2D宽度受姿态影响，所以我们用3D宽度来校正：
        - 校正后的2D宽度 = 2D宽度 / cos(yaw)  （简化模型）
        - 或者直接使用3D宽度的比值作为距离变化的指标
        
        **Validates: Requirements 3.1, 3.2, 3.4**
        """
        result = RelativeDistanceResult()
        result.raw_measurement = measurement
        
        # 计算身体朝向
        result.body_orientation = measurement.body_yaw
        result.is_frontal = classify_pose_orientation(measurement.body_yaw)
        
        # 校正2D测量值（消除姿态影响）- 使用独立函数
        # 对肩宽和胯宽分别独立应用校正公式
        if measurement.shoulder_width_2d > 0:
            result.corrected_shoulder_width = compute_corrected_shoulder_width(
                measurement.shoulder_width_2d, measurement.body_yaw
            )
        
        if measurement.hip_width_2d > 0:
            result.corrected_hip_width = compute_corrected_hip_width(
                measurement.hip_width_2d, measurement.body_yaw
            )
        
        # 如果有参考帧，计算相对距离比值
        if self.is_calibrated and self.reference_measurement:
            ref = self.reference_measurement
            
            # 基于肩宽的距离比值
            if measurement.shoulder_width_2d > 0 and ref.shoulder_width_2d > 0:
                # 校正后的参考肩宽
                ref_corrected_shoulder = compute_corrected_shoulder_width(
                    ref.shoulder_width_2d, ref.body_yaw
                )
                # 距离比值 = 参考宽度 / 当前宽度（宽度越小，距离越远）
                result.shoulder_ratio = ref_corrected_shoulder / result.corrected_shoulder_width
            
            # 基于胯宽的距离比值
            if measurement.hip_width_2d > 0 and ref.hip_width_2d > 0:
                ref_corrected_hip = compute_corrected_hip_width(
                    ref.hip_width_2d, ref.body_yaw
                )
                result.hip_ratio = ref_corrected_hip / result.corrected_hip_width
            
            # 融合比值（加权平均）
            weights = []
            ratios = []
            
            if result.shoulder_ratio > 0:
                weights.append(0.6)  # 肩宽权重更高
                ratios.append(result.shoulder_ratio)
            
            if result.hip_ratio > 0:
                weights.append(0.4)
                ratios.append(result.hip_ratio)
            
            if weights:
                result.combined_ratio = sum(w * r for w, r in zip(weights, ratios)) / sum(weights)
                result.relative_distance_ratio = result.combined_ratio
        
        # 计算置信度
        # 首先应用极端偏航角的置信度调整 (|yaw| > 70°)
        result.confidence = adjust_confidence_for_extreme_yaw(
            measurement.confidence, measurement.body_yaw
        )
        # 侧身时额外降低置信度
        if not result.is_frontal:
            result.confidence *= 0.8
        
        result.method = "pose_corrected_3d"
        
        return result

    def _draw_visualization(self, 
                           image: np.ndarray, 
                           pose_landmarks,
                           measurement: BodyMeasurement,
                           result: RelativeDistanceResult) -> np.ndarray:
        """
        绘制可视化信息
        
        使用visualization模块中的函数绘制：
        1. 姿态骨架 (Requirements 7.1)
        2. 测量信息 (Requirements 7.2)
        3. 距离比值与颜色编码 (Requirements 7.3)
        4. 方向指示器 (Requirements 7.4)
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        from .visualization import draw_complete_visualization
        
        return draw_complete_visualization(
            image=image,
            pose_landmarks=pose_landmarks,
            measurement=measurement,
            result=result,
            is_calibrated=self.is_calibrated
        )
    
    def get_smoothed_ratio(self) -> float:
        """获取平滑后的距离比值"""
        if not self.ratio_history:
            return 1.0
        return np.median(list(self.ratio_history))
    
    def reset(self):
        """
        重置估算器状态
        
        清除参考帧并将 is_calibrated 设置为 false。
        
        **Validates: Requirements 4.4**
        """
        self.reference_measurement = None
        self.is_calibrated = False
        self.measurement_history.clear()
        self.ratio_history.clear()
        self.frame_count = 0
        logger.info("估算器已重置: 参考帧已清除，is_calibrated=False")


class RelativeDistanceTracker:
    """
    相对距离追踪器
    
    用于追踪距离的相对变化，提供历史数据管理和统计功能。
    
    功能：
    - 历史数据管理：维护30帧窗口的测量历史
    - 统计计算：计算均值、标准差、最小/最大值等
    - 稳定性检测：检测距离比值是否稳定（标准差 < 0.15）
    - 方向追踪：追踪移动方向（靠近/远离/稳定）
    
    **Validates: Requirements 5.1, 6.2**
    """
    
    def __init__(self, history_size: int = 30):
        """
        初始化追踪器
        
        Args:
            history_size: 历史记录大小，默认30帧（用于计算统计信息）
                         根据Requirements 6.2，需要30帧窗口来计算标准差
        """
        self.estimator = PoseCorrectedDistanceEstimator()
        self.history_size = history_size
        
        # 历史记录 - 用于追踪距离变化和计算统计信息
        self.ratio_history: deque = deque(maxlen=history_size)
        self.shoulder_ratio_history: deque = deque(maxlen=history_size)
        self.hip_ratio_history: deque = deque(maxlen=history_size)
        self.shoulder_width_history: deque = deque(maxlen=history_size)
        self.hip_width_history: deque = deque(maxlen=history_size)
        self.yaw_history: deque = deque(maxlen=history_size)
        self.confidence_history: deque = deque(maxlen=history_size)
        self.timestamp_history: deque = deque(maxlen=history_size)
        
        # 统计信息
        self.min_ratio = float('inf')
        self.max_ratio = 0.0
        self.total_frames_processed = 0
        self.valid_frames_count = 0
        
        # 方向追踪
        self._last_direction = "STABLE"
        self._direction_change_count = 0
        
    def process(self, image: np.ndarray) -> Tuple[Optional[RelativeDistanceResult], np.ndarray]:
        """
        处理帧并追踪距离变化
        
        处理单帧图像，更新历史记录，并返回距离测量结果。
        
        Args:
            image: BGR图像
            
        Returns:
            (RelativeDistanceResult, 标注后的图像) 或 (None, 原图像)
            
        **Validates: Requirements 5.1**
        """
        self.total_frames_processed += 1
        result, output_image = self.estimator.process_frame(image)
        
        if result and result.confidence > 0.5:
            self.valid_frames_count += 1
            
            # 更新历史记录
            self.ratio_history.append(result.relative_distance_ratio)
            self.shoulder_ratio_history.append(result.shoulder_ratio)
            self.hip_ratio_history.append(result.hip_ratio)
            self.confidence_history.append(result.confidence)
            self.timestamp_history.append(time.time())
            
            if result.corrected_shoulder_width > 0:
                self.shoulder_width_history.append(result.corrected_shoulder_width)
            if result.corrected_hip_width > 0:
                self.hip_width_history.append(result.corrected_hip_width)
            
            # 记录身体朝向角度
            if result.raw_measurement:
                self.yaw_history.append(result.raw_measurement.body_yaw)
            
            # 更新统计
            if self.estimator.is_calibrated:
                self.min_ratio = min(self.min_ratio, result.relative_distance_ratio)
                self.max_ratio = max(self.max_ratio, result.relative_distance_ratio)
            
            # 追踪方向变化
            current_direction = get_direction_indication(result.relative_distance_ratio)
            if current_direction != self._last_direction:
                self._direction_change_count += 1
                self._last_direction = current_direction
        
        return result, output_image
    
    def calibrate(self) -> bool:
        """
        校准（设置当前帧为参考）
        
        Returns:
            bool: True 如果校准成功，False 如果校准失败
            
        **Validates: Requirements 4.1**
        """
        success = self.estimator.set_reference_frame()
        if success:
            # 重置统计信息
            self.min_ratio = 1.0
            self.max_ratio = 1.0
            self.ratio_history.clear()
            self.shoulder_ratio_history.clear()
            self.hip_ratio_history.clear()
            self._direction_change_count = 0
            self._last_direction = "STABLE"
        return success
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        返回包含以下信息的字典：
        - 校准状态
        - 帧计数
        - 距离比值统计（均值、标准差、最小/最大值）
        - 肩宽/胯宽统计
        - 稳定性指标（标准差是否 < 0.15）
        
        **Validates: Requirements 6.2**
        """
        stats = {
            "is_calibrated": self.estimator.is_calibrated,
            "total_frames_processed": self.total_frames_processed,
            "valid_frames_count": self.valid_frames_count,
            "history_size": len(self.ratio_history),
            "history_capacity": self.history_size,
        }
        
        # 距离比值统计
        if self.ratio_history:
            ratios = list(self.ratio_history)
            stats["current_ratio"] = ratios[-1]
            stats["mean_ratio"] = float(np.mean(ratios))
            stats["std_ratio"] = float(np.std(ratios))
            stats["min_ratio"] = self.min_ratio if self.min_ratio != float('inf') else None
            stats["max_ratio"] = self.max_ratio if self.max_ratio > 0 else None
            stats["median_ratio"] = float(np.median(ratios))
            
            # 稳定性检测 - Requirements 6.2: 标准差应 < 0.15
            stats["is_stable"] = stats["std_ratio"] < 0.15
            stats["stability_threshold"] = 0.15
        
        # 肩宽比值统计
        if self.shoulder_ratio_history:
            shoulder_ratios = list(self.shoulder_ratio_history)
            stats["shoulder_mean_ratio"] = float(np.mean(shoulder_ratios))
            stats["shoulder_std_ratio"] = float(np.std(shoulder_ratios))
        
        # 胯宽比值统计
        if self.hip_ratio_history:
            hip_ratios = list(self.hip_ratio_history)
            stats["hip_mean_ratio"] = float(np.mean(hip_ratios))
            stats["hip_std_ratio"] = float(np.std(hip_ratios))
        
        # 校正后宽度统计
        if self.shoulder_width_history:
            stats["mean_shoulder_px"] = float(np.mean(list(self.shoulder_width_history)))
            stats["std_shoulder_px"] = float(np.std(list(self.shoulder_width_history)))
        
        if self.hip_width_history:
            stats["mean_hip_px"] = float(np.mean(list(self.hip_width_history)))
            stats["std_hip_px"] = float(np.std(list(self.hip_width_history)))
        
        # 身体朝向统计
        if self.yaw_history:
            yaws = list(self.yaw_history)
            stats["mean_yaw"] = float(np.mean(yaws))
            stats["std_yaw"] = float(np.std(yaws))
            stats["min_yaw"] = float(np.min(yaws))
            stats["max_yaw"] = float(np.max(yaws))
        
        # 置信度统计
        if self.confidence_history:
            confidences = list(self.confidence_history)
            stats["mean_confidence"] = float(np.mean(confidences))
            stats["min_confidence"] = float(np.min(confidences))
        
        # 方向变化统计
        stats["direction_change_count"] = self._direction_change_count
        stats["current_direction"] = self._last_direction
        
        return stats
    
    def get_smoothed_ratio(self) -> float:
        """
        获取平滑后的距离比值
        
        使用中位数滤波来减少噪声影响。
        
        Returns:
            平滑后的距离比值，如果没有历史数据则返回1.0
        """
        if not self.ratio_history:
            return 1.0
        return float(np.median(list(self.ratio_history)))
    
    def get_ratio_std(self) -> float:
        """
        获取距离比值的标准差
        
        用于评估测量稳定性。根据Requirements 6.2，
        标准差应 < 0.15 才认为测量稳定。
        
        Returns:
            标准差，如果没有历史数据则返回0.0
            
        **Validates: Requirements 6.2**
        """
        if not self.ratio_history:
            return 0.0
        return float(np.std(list(self.ratio_history)))
    
    def is_measurement_stable(self) -> bool:
        """
        检查测量是否稳定
        
        根据Requirements 6.2，当标准差 < 0.15 时认为测量稳定。
        
        Returns:
            True 如果测量稳定（std < 0.15），否则 False
            
        **Validates: Requirements 6.2**
        """
        return self.get_ratio_std() < 0.15
    
    def get_movement_direction(self) -> str:
        """
        获取当前移动方向
        
        Returns:
            "APPROACHING", "MOVING AWAY", 或 "STABLE"
            
        **Validates: Requirements 5.2, 5.3**
        """
        return self._last_direction
    
    def get_history_summary(self) -> Dict:
        """
        获取历史数据摘要
        
        返回最近N帧的统计摘要，用于分析距离变化趋势。
        
        Returns:
            包含历史数据摘要的字典
        """
        summary = {
            "ratio_history": list(self.ratio_history),
            "shoulder_ratio_history": list(self.shoulder_ratio_history),
            "hip_ratio_history": list(self.hip_ratio_history),
            "yaw_history": list(self.yaw_history),
            "confidence_history": list(self.confidence_history),
        }
        return summary
    
    def reset(self):
        """
        重置追踪器
        
        清除所有历史数据和统计信息。
        
        **Validates: Requirements 4.4**
        """
        self.estimator.reset()
        self.ratio_history.clear()
        self.shoulder_ratio_history.clear()
        self.hip_ratio_history.clear()
        self.shoulder_width_history.clear()
        self.hip_width_history.clear()
        self.yaw_history.clear()
        self.confidence_history.clear()
        self.timestamp_history.clear()
        self.min_ratio = float('inf')
        self.max_ratio = 0.0
        self.total_frames_processed = 0
        self.valid_frames_count = 0
        self._direction_change_count = 0
        self._last_direction = "STABLE"
        logger.info("追踪器已重置: 所有历史数据已清除")


def run_live_demo(camera_index: int = 0, width: int = 1280, height: int = 720):
    """
    运行实时演示
    
    实时处理摄像头画面，显示姿态校正后的距离测量结果。
    
    Args:
        camera_index: 摄像头索引，默认为0（主摄像头）
        width: 视频宽度，默认1280
        height: 视频高度，默认720
        
    键盘控制:
        C - 校准（设置当前位置为参考点）
        R - 重置
        S - 显示统计信息
        Q - 退出
        
    **Validates: Requirements 4.1, 4.4, 7.1, 7.2, 7.3, 7.4**
    """
    print("=" * 60)
    print("基于3D姿态校正的相对距离测量演示")
    print("Pose-Corrected Distance Measurement Demo")
    print("=" * 60)
    print("\n操作说明 / Controls:")
    print("  C - 校准 / Calibrate (set current position as reference)")
    print("  R - 重置 / Reset")
    print("  S - 显示统计信息 / Show statistics")
    print("  Q - 退出 / Quit")
    print("\n请先站在摄像头前，按 'C' 进行校准")
    print("Stand in front of the camera and press 'C' to calibrate")
    print("然后前后移动，观察距离比值的变化")
    print("Then move forward/backward to observe distance ratio changes")
    print("=" * 60)
    
    # 创建追踪器
    tracker = RelativeDistanceTracker()
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_index}")
        print(f"Error: Cannot open camera {camera_index}")
        return False
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n摄像头分辨率 / Camera resolution: {actual_width}x{actual_height}")
    
    window_name = "Pose Corrected Distance"
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("警告: 无法读取帧 / Warning: Cannot read frame")
                break
            
            # 处理帧
            result, output_frame = tracker.process(frame)
            
            # 在窗口上显示帮助信息
            help_text = "C:Calibrate  R:Reset  S:Stats  Q:Quit"
            cv2.putText(
                output_frame,
                help_text,
                (output_frame.shape[1] - 350, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # 显示校准状态
            if tracker.estimator.is_calibrated:
                status_text = "CALIBRATED"
                status_color = (0, 255, 0)  # 绿色
            else:
                status_text = "NOT CALIBRATED - Press C"
                status_color = (0, 165, 255)  # 橙色
            
            cv2.putText(
                output_frame,
                status_text,
                (output_frame.shape[1] - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                2
            )
            
            # 显示图像
            cv2.imshow(window_name, output_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Q - 退出
                print("\n退出演示 / Exiting demo...")
                break
                
            elif key == ord('c') or key == ord('C'):
                # C - 校准 (Requirements 4.1)
                success = tracker.estimator.set_reference_frame()
                if success:
                    tracker.min_ratio = 1.0
                    tracker.max_ratio = 1.0
                    tracker.ratio_history.clear()
                    print("\n[校准完成 / Calibrated] 当前位置设为参考点 / Current position set as reference")
                else:
                    print("\n[校准失败 / Calibration failed] 置信度过低，请调整姿势 / Low confidence, adjust pose")
                    
            elif key == ord('r') or key == ord('R'):
                # R - 重置 (Requirements 4.4)
                tracker.reset()
                print("\n[重置完成 / Reset] 所有数据已清除 / All data cleared")
                
            elif key == ord('s') or key == ord('S'):
                # S - 显示统计信息
                stats = tracker.get_statistics()
                print("\n" + "=" * 40)
                print("统计信息 / Statistics")
                print("=" * 40)
                for k, v in stats.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                print("=" * 40)
    
    except KeyboardInterrupt:
        print("\n用户中断 / User interrupted")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放 / Resources released")
    
    return True


if __name__ == "__main__":
    run_live_demo()
