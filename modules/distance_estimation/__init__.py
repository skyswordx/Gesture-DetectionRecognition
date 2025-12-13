"""
距离估算模块

This module provides distance estimation functionality including:
- Basic distance estimation (DistanceEstimator)
- Pose-corrected distance estimation (PoseCorrectedDistanceEstimator)
- Relative distance tracking (RelativeDistanceTracker)
"""

from .distance_estimator import (
    DistanceEstimator,
    DistanceVisualizer,
    CameraCalibration,
    KalmanFilter,
    DistanceResult
)

# Pose-corrected distance estimation (mediapipe required)
try:
    from .pose_corrected_distance import (
        PoseCorrectedDistanceEstimator,
        RelativeDistanceTracker,
        Landmark3D,
        Landmark2D,
        BodyMeasurement,
        RelativeDistanceResult,
        # Core functions
        euclidean_distance_3d,
        estimate_body_yaw,
        classify_pose_orientation,
        compute_corrected_width,
        compute_corrected_shoulder_width,
        compute_corrected_hip_width,
        compute_relative_distance_ratio,
        get_direction_indication,
        compute_weighted_ratio_combination,
        # Live demo function
        run_live_demo,
    )
    _POSE_CORRECTED_AVAILABLE = True
except ImportError:
    _POSE_CORRECTED_AVAILABLE = False

# Visualization functions (mediapipe optional)
try:
    from .visualization import (
        draw_pose_skeleton,
        draw_measurement_info,
        draw_distance_ratio,
        draw_direction_indicator,
        draw_confidence,
        draw_orientation_indicator,
        draw_complete_visualization,
        get_ratio_color,
        get_direction_text_and_color,
        STABLE_RATIO_MIN,
        STABLE_RATIO_MAX,
        COLOR_GREEN,
        COLOR_ORANGE,
        COLOR_RED,
        COLOR_YELLOW,
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

__all__ = [
    'DistanceEstimator',
    'DistanceVisualizer',
    'CameraCalibration', 
    'KalmanFilter',
    'DistanceResult',
]

# Add pose-corrected exports if available
if _POSE_CORRECTED_AVAILABLE:
    __all__.extend([
        'PoseCorrectedDistanceEstimator',
        'RelativeDistanceTracker',
        'Landmark3D',
        'Landmark2D',
        'BodyMeasurement',
        'RelativeDistanceResult',
        'euclidean_distance_3d',
        'estimate_body_yaw',
        'classify_pose_orientation',
        'compute_corrected_width',
        'compute_corrected_shoulder_width',
        'compute_corrected_hip_width',
        'compute_relative_distance_ratio',
        'get_direction_indication',
        'compute_weighted_ratio_combination',
        'run_live_demo',
    ])

# Add visualization exports if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        'draw_pose_skeleton',
        'draw_measurement_info',
        'draw_distance_ratio',
        'draw_direction_indicator',
        'draw_confidence',
        'draw_orientation_indicator',
        'draw_complete_visualization',
        'get_ratio_color',
        'get_direction_text_and_color',
        'STABLE_RATIO_MIN',
        'STABLE_RATIO_MAX',
        'COLOR_GREEN',
        'COLOR_ORANGE',
        'COLOR_RED',
        'COLOR_YELLOW',
    ])

# Synthetic data generator for testing (no external dependencies)
from .synthetic_data_generator import (
    generate_synthetic_landmarks,
    generate_rotation_sequence,
    generate_distance_sequence,
    verify_3d_shoulder_width_constant,
    compute_expected_2d_width,
    SyntheticLandmarkResult,
    SyntheticBodyConfig,
    CameraConfig,
    project_3d_to_2d,
    normalize_2d_coordinates,
    rotate_point_around_y,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    NUM_LANDMARKS,
)

__all__.extend([
    'generate_synthetic_landmarks',
    'generate_rotation_sequence',
    'generate_distance_sequence',
    'verify_3d_shoulder_width_constant',
    'compute_expected_2d_width',
    'SyntheticLandmarkResult',
    'SyntheticBodyConfig',
    'CameraConfig',
    'project_3d_to_2d',
    'normalize_2d_coordinates',
    'rotate_point_around_y',
    'LEFT_SHOULDER',
    'RIGHT_SHOULDER',
    'LEFT_HIP',
    'RIGHT_HIP',
    'NUM_LANDMARKS',
])
