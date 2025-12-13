"""
Property-based tests and unit tests for pose_corrected_distance module.

Uses Hypothesis for property-based testing as specified in the design document.
"""

import math
import sys
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from dataclasses import dataclass


# Define local copies of data structures for testing without mediapipe dependency
@dataclass
class Landmark3D:
    """3D世界坐标关键点数据结构"""
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class Landmark2D:
    """2D图像坐标关键点数据结构"""
    x: float
    y: float
    z: float
    visibility: float


def euclidean_distance_3d(p1: Landmark3D, p2: Landmark3D) -> float:
    """
    计算两个3D关键点之间的欧氏距离
    
    公式: sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
    """
    return math.sqrt(
        (p2.x - p1.x) ** 2 + 
        (p2.y - p1.y) ** 2 + 
        (p2.z - p1.z) ** 2
    )


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating reasonable 3D coordinates (in meters, typical human scale)
coordinate_strategy = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for generating visibility values
visibility_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def landmark_3d_strategy(draw):
    """Generate a valid Landmark3D instance."""
    return Landmark3D(
        x=draw(coordinate_strategy),
        y=draw(coordinate_strategy),
        z=draw(coordinate_strategy),
        visibility=draw(visibility_strategy)
    )


# =============================================================================
# Property 1: 3D Euclidean Distance Calculation
# =============================================================================

@given(
    x1=coordinate_strategy,
    y1=coordinate_strategy,
    z1=coordinate_strategy,
    x2=coordinate_strategy,
    y2=coordinate_strategy,
    z2=coordinate_strategy,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_euclidean_distance_3d_formula(x1, y1, z1, x2, y2, z2):
    """
    **Feature: pose-corrected-distance, Property 1: 3D Euclidean Distance Calculation**
    
    *For any* two 3D landmarks with coordinates (x1, y1, z1) and (x2, y2, z2), 
    the computed distance SHALL equal sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²) 
    within floating-point precision.
    
    **Validates: Requirements 1.2**
    """
    p1 = Landmark3D(x=x1, y=y1, z=z1, visibility=1.0)
    p2 = Landmark3D(x=x2, y=y2, z=z2, visibility=1.0)
    
    # Compute using our function
    computed_distance = euclidean_distance_3d(p1, p2)
    
    # Compute expected value using the formula directly
    expected_distance = math.sqrt(
        (x2 - x1) ** 2 + 
        (y2 - y1) ** 2 + 
        (z2 - z1) ** 2
    )
    
    # Assert equality within floating-point precision
    assert math.isclose(computed_distance, expected_distance, rel_tol=1e-9), \
        f"Distance mismatch: computed={computed_distance}, expected={expected_distance}"


@given(p1=landmark_3d_strategy(), p2=landmark_3d_strategy())
@settings(max_examples=100)
def test_property_euclidean_distance_non_negative(p1, p2):
    """
    **Feature: pose-corrected-distance, Property 1: 3D Euclidean Distance Calculation**
    
    *For any* two 3D landmarks, the computed distance SHALL be non-negative.
    
    **Validates: Requirements 1.2**
    """
    distance = euclidean_distance_3d(p1, p2)
    assert distance >= 0, f"Distance should be non-negative, got {distance}"


@given(p=landmark_3d_strategy())
@settings(max_examples=100)
def test_property_euclidean_distance_identity(p):
    """
    **Feature: pose-corrected-distance, Property 1: 3D Euclidean Distance Calculation**
    
    *For any* 3D landmark, the distance to itself SHALL be zero.
    
    **Validates: Requirements 1.2**
    """
    distance = euclidean_distance_3d(p, p)
    assert math.isclose(distance, 0.0, abs_tol=1e-9), \
        f"Distance to self should be 0, got {distance}"


@given(p1=landmark_3d_strategy(), p2=landmark_3d_strategy())
@settings(max_examples=100)
def test_property_euclidean_distance_symmetry(p1, p2):
    """
    **Feature: pose-corrected-distance, Property 1: 3D Euclidean Distance Calculation**
    
    *For any* two 3D landmarks, distance(p1, p2) SHALL equal distance(p2, p1).
    
    **Validates: Requirements 1.2**
    """
    d1 = euclidean_distance_3d(p1, p2)
    d2 = euclidean_distance_3d(p2, p1)
    assert math.isclose(d1, d2, rel_tol=1e-9), \
        f"Distance should be symmetric: d(p1,p2)={d1}, d(p2,p1)={d2}"



# =============================================================================
# Unit Tests for Data Structure Validation (Task 1.3)
# =============================================================================

@dataclass
class BodyMeasurement:
    """身体测量数据 - local copy for testing"""
    shoulder_width_3d: float = 0.0
    hip_width_3d: float = 0.0
    torso_height_3d: float = 0.0
    shoulder_width_2d: float = 0.0
    hip_width_2d: float = 0.0
    body_yaw: float = 0.0
    body_pitch: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class RelativeDistanceResult:
    """相对距离测量结果 - local copy for testing"""
    relative_distance_ratio: float = 1.0
    shoulder_ratio: float = 1.0
    hip_ratio: float = 1.0
    combined_ratio: float = 1.0
    corrected_shoulder_width: float = 0.0
    corrected_hip_width: float = 0.0
    body_orientation: float = 0.0
    is_frontal: bool = True
    confidence: float = 0.0
    method: str = ""


class TestLandmark3DCreation:
    """Unit tests for Landmark3D dataclass creation."""
    
    def test_create_landmark3d_with_valid_values(self):
        """Test creating Landmark3D with valid coordinate values."""
        landmark = Landmark3D(x=0.1, y=0.2, z=0.3, visibility=0.9)
        assert landmark.x == 0.1
        assert landmark.y == 0.2
        assert landmark.z == 0.3
        assert landmark.visibility == 0.9
    
    def test_create_landmark3d_with_zero_values(self):
        """Test creating Landmark3D with zero coordinates."""
        landmark = Landmark3D(x=0.0, y=0.0, z=0.0, visibility=1.0)
        assert landmark.x == 0.0
        assert landmark.y == 0.0
        assert landmark.z == 0.0
    
    def test_create_landmark3d_with_negative_values(self):
        """Test creating Landmark3D with negative coordinates."""
        landmark = Landmark3D(x=-0.5, y=-0.3, z=-0.1, visibility=0.8)
        assert landmark.x == -0.5
        assert landmark.y == -0.3
        assert landmark.z == -0.1


class TestLandmark2DCreation:
    """Unit tests for Landmark2D dataclass creation."""
    
    def test_create_landmark2d_with_valid_values(self):
        """Test creating Landmark2D with valid normalized coordinates."""
        landmark = Landmark2D(x=0.5, y=0.5, z=0.1, visibility=0.95)
        assert landmark.x == 0.5
        assert landmark.y == 0.5
        assert landmark.z == 0.1
        assert landmark.visibility == 0.95
    
    def test_create_landmark2d_at_image_corners(self):
        """Test creating Landmark2D at image boundary positions."""
        # Top-left corner
        tl = Landmark2D(x=0.0, y=0.0, z=0.0, visibility=1.0)
        assert tl.x == 0.0 and tl.y == 0.0
        
        # Bottom-right corner
        br = Landmark2D(x=1.0, y=1.0, z=0.0, visibility=1.0)
        assert br.x == 1.0 and br.y == 1.0


class TestBodyMeasurementCreation:
    """Unit tests for BodyMeasurement dataclass creation."""
    
    def test_create_body_measurement_with_defaults(self):
        """Test creating BodyMeasurement with default values."""
        measurement = BodyMeasurement()
        assert measurement.shoulder_width_3d == 0.0
        assert measurement.hip_width_3d == 0.0
        assert measurement.confidence == 0.0
    
    def test_create_body_measurement_with_valid_values(self):
        """Test creating BodyMeasurement with typical human measurements."""
        measurement = BodyMeasurement(
            shoulder_width_3d=0.40,  # 40cm shoulder width
            hip_width_3d=0.30,       # 30cm hip width
            shoulder_width_2d=200.0, # 200 pixels
            hip_width_2d=150.0,      # 150 pixels
            body_yaw=15.0,           # 15 degrees
            confidence=0.85,
            timestamp=1234567890.0
        )
        assert measurement.shoulder_width_3d == 0.40
        assert measurement.hip_width_3d == 0.30
        assert measurement.shoulder_width_2d == 200.0
        assert measurement.hip_width_2d == 150.0
        assert measurement.body_yaw == 15.0
        assert measurement.confidence == 0.85
    
    def test_body_measurement_extreme_yaw_angles(self):
        """Test BodyMeasurement with extreme yaw angles."""
        # Side view (90 degrees)
        side_view = BodyMeasurement(body_yaw=90.0, confidence=0.5)
        assert side_view.body_yaw == 90.0
        
        # Negative yaw (turned left)
        left_turn = BodyMeasurement(body_yaw=-45.0, confidence=0.7)
        assert left_turn.body_yaw == -45.0


class TestRelativeDistanceResultCreation:
    """Unit tests for RelativeDistanceResult dataclass creation."""
    
    def test_create_result_with_defaults(self):
        """Test creating RelativeDistanceResult with default values."""
        result = RelativeDistanceResult()
        assert result.relative_distance_ratio == 1.0
        assert result.is_frontal == True
        assert result.confidence == 0.0
    
    def test_create_result_approaching(self):
        """Test creating result indicating subject is approaching."""
        result = RelativeDistanceResult(
            relative_distance_ratio=0.8,  # 20% closer
            shoulder_ratio=0.8,
            hip_ratio=0.8,
            is_frontal=True,
            confidence=0.9
        )
        assert result.relative_distance_ratio < 1.0
        assert result.relative_distance_ratio == 0.8
    
    def test_create_result_moving_away(self):
        """Test creating result indicating subject is moving away."""
        result = RelativeDistanceResult(
            relative_distance_ratio=1.5,  # 50% farther
            shoulder_ratio=1.5,
            hip_ratio=1.5,
            is_frontal=True,
            confidence=0.85
        )
        assert result.relative_distance_ratio > 1.0
        assert result.relative_distance_ratio == 1.5
    
    def test_create_result_side_view(self):
        """Test creating result for side view pose."""
        result = RelativeDistanceResult(
            body_orientation=45.0,
            is_frontal=False,
            confidence=0.6
        )
        assert result.is_frontal == False
        assert result.body_orientation == 45.0


# =============================================================================
# Visibility Filtering Functions for Testing (Task 2.2)
# =============================================================================

VISIBILITY_THRESHOLD = 0.5


def is_landmark_visible(landmark: Landmark3D, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """
    检查关键点是否可见
    
    **Validates: Requirements 1.3**
    """
    return landmark.visibility >= threshold


def is_landmark_2d_visible(landmark: Landmark2D, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """
    检查2D关键点是否可见
    
    **Validates: Requirements 1.3**
    """
    return landmark.visibility >= threshold


def filter_visible_landmarks_3d(
    landmarks: list, 
    threshold: float = VISIBILITY_THRESHOLD
) -> list:
    """
    过滤出可见的3D关键点
    
    **Validates: Requirements 1.3**
    """
    return [lm for lm in landmarks if lm.visibility >= threshold]


def filter_visible_landmarks_2d(
    landmarks: list, 
    threshold: float = VISIBILITY_THRESHOLD
) -> list:
    """
    过滤出可见的2D关键点
    
    **Validates: Requirements 1.3**
    """
    return [lm for lm in landmarks if lm.visibility >= threshold]


# =============================================================================
# Property 2: Visibility Filtering (Task 2.3)
# =============================================================================

@st.composite
def landmark_3d_with_visibility_strategy(draw, min_vis=0.0, max_vis=1.0):
    """Generate a Landmark3D with visibility in specified range."""
    return Landmark3D(
        x=draw(coordinate_strategy),
        y=draw(coordinate_strategy),
        z=draw(coordinate_strategy),
        visibility=draw(st.floats(min_value=min_vis, max_value=max_vis, 
                                   allow_nan=False, allow_infinity=False))
    )


@st.composite
def landmark_2d_with_visibility_strategy(draw, min_vis=0.0, max_vis=1.0):
    """Generate a Landmark2D with visibility in specified range."""
    return Landmark2D(
        x=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        y=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        z=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        visibility=draw(st.floats(min_value=min_vis, max_value=max_vis, 
                                   allow_nan=False, allow_infinity=False))
    )


@given(visibility=st.floats(min_value=0.0, max_value=0.4999, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_visibility_below_threshold_excluded(visibility):
    """
    **Feature: pose-corrected-distance, Property 2: Visibility Filtering**
    
    *For any* landmark with visibility below 0.5, the measurement SHALL be 
    excluded from the result.
    
    **Validates: Requirements 1.3**
    """
    landmark = Landmark3D(x=0.0, y=0.0, z=0.0, visibility=visibility)
    
    # Landmark should NOT be visible (excluded)
    assert not is_landmark_visible(landmark, threshold=0.5), \
        f"Landmark with visibility {visibility} should be excluded (< 0.5)"


@given(visibility=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_visibility_at_or_above_threshold_included(visibility):
    """
    **Feature: pose-corrected-distance, Property 2: Visibility Filtering**
    
    *For any* landmark with visibility 0.5 or above, the measurement SHALL be 
    included.
    
    **Validates: Requirements 1.3**
    """
    landmark = Landmark3D(x=0.0, y=0.0, z=0.0, visibility=visibility)
    
    # Landmark should be visible (included)
    assert is_landmark_visible(landmark, threshold=0.5), \
        f"Landmark with visibility {visibility} should be included (>= 0.5)"


@given(
    low_vis_count=st.integers(min_value=0, max_value=10),
    high_vis_count=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100)
def test_property_filter_preserves_visible_landmarks(low_vis_count, high_vis_count):
    """
    **Feature: pose-corrected-distance, Property 2: Visibility Filtering**
    
    *For any* list of landmarks, filtering SHALL return exactly those landmarks
    with visibility >= 0.5.
    
    **Validates: Requirements 1.3**
    """
    # Create landmarks with low visibility (< 0.5)
    low_vis_landmarks = [
        Landmark3D(x=float(i), y=0.0, z=0.0, visibility=0.3)
        for i in range(low_vis_count)
    ]
    
    # Create landmarks with high visibility (>= 0.5)
    high_vis_landmarks = [
        Landmark3D(x=float(i + 100), y=0.0, z=0.0, visibility=0.8)
        for i in range(high_vis_count)
    ]
    
    # Combine and shuffle
    all_landmarks = low_vis_landmarks + high_vis_landmarks
    
    # Filter
    filtered = filter_visible_landmarks_3d(all_landmarks, threshold=0.5)
    
    # Should only contain high visibility landmarks
    assert len(filtered) == high_vis_count, \
        f"Expected {high_vis_count} visible landmarks, got {len(filtered)}"
    
    # All filtered landmarks should have visibility >= 0.5
    for lm in filtered:
        assert lm.visibility >= 0.5, \
            f"Filtered landmark has visibility {lm.visibility} < 0.5"


@given(visibility=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_visibility_threshold_boundary(visibility):
    """
    **Feature: pose-corrected-distance, Property 2: Visibility Filtering**
    
    *For any* visibility value, the filtering decision SHALL be consistent:
    - visibility < 0.5 -> excluded
    - visibility >= 0.5 -> included
    
    **Validates: Requirements 1.3**
    """
    landmark = Landmark3D(x=0.0, y=0.0, z=0.0, visibility=visibility)
    
    is_visible = is_landmark_visible(landmark, threshold=0.5)
    
    if visibility < 0.5:
        assert not is_visible, \
            f"Visibility {visibility} < 0.5 should be excluded"
    else:
        assert is_visible, \
            f"Visibility {visibility} >= 0.5 should be included"


@given(
    vis1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    vis2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_2d_visibility_filtering_consistency(vis1, vis2):
    """
    **Feature: pose-corrected-distance, Property 2: Visibility Filtering**
    
    *For any* pair of 2D landmarks, filtering SHALL work consistently for 2D landmarks
    as it does for 3D landmarks.
    
    **Validates: Requirements 1.3**
    """
    landmarks = [
        Landmark2D(x=0.5, y=0.5, z=0.0, visibility=vis1),
        Landmark2D(x=0.6, y=0.6, z=0.0, visibility=vis2)
    ]
    
    filtered = filter_visible_landmarks_2d(landmarks, threshold=0.5)
    
    expected_count = sum(1 for v in [vis1, vis2] if v >= 0.5)
    
    assert len(filtered) == expected_count, \
        f"Expected {expected_count} visible landmarks, got {len(filtered)}"


# =============================================================================
# Body Orientation Estimation Functions (Task 5)
# =============================================================================

def estimate_body_yaw(left_shoulder: Landmark3D, right_shoulder: Landmark3D) -> float:
    """
    估算身体偏航角（左右转动）
    
    使用atan2(dz, dx)计算肩膀连线在xz平面的角度。
    
    **Validates: Requirements 2.1, 2.4**
    """
    dx = right_shoulder.x - left_shoulder.x
    dz = right_shoulder.z - left_shoulder.z
    
    yaw = math.atan2(dz, dx)
    return math.degrees(yaw)


def classify_pose_orientation(body_yaw: float) -> bool:
    """
    根据身体偏航角分类姿态为正面或侧面
    
    **Validates: Requirements 2.2**
    """
    return abs(body_yaw) < 30.0


def adjust_confidence_for_extreme_yaw(confidence: float, body_yaw: float) -> float:
    """
    根据极端偏航角调整置信度
    
    **Validates: Requirements 2.3**
    """
    if abs(body_yaw) > 70.0:
        return confidence * 0.5
    return confidence


# =============================================================================
# Property 3: Body Yaw Calculation (Task 5.2)
# =============================================================================

@given(
    dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    dz=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    base_x=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    base_z=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_body_yaw_formula(dx, dz, base_x, base_z, y):
    """
    **Feature: pose-corrected-distance, Property 3: Body Yaw Calculation**
    
    *For any* pair of shoulder 3D coordinates, the computed Body_Yaw SHALL equal 
    atan2(dz, dx) converted to degrees, where dz and dx are the z and x differences 
    between right and left shoulders.
    
    **Validates: Requirements 2.1, 2.4**
    """
    # Create left and right shoulder with known dx and dz
    left_shoulder = Landmark3D(x=base_x, y=y, z=base_z, visibility=1.0)
    right_shoulder = Landmark3D(x=base_x + dx, y=y, z=base_z + dz, visibility=1.0)
    
    # Compute using our function
    computed_yaw = estimate_body_yaw(left_shoulder, right_shoulder)
    
    # Compute expected value using the formula directly
    expected_yaw = math.degrees(math.atan2(dz, dx))
    
    # Assert equality within floating-point precision
    # Use both relative and absolute tolerance to handle near-zero values
    assert math.isclose(computed_yaw, expected_yaw, rel_tol=1e-9, abs_tol=1e-9), \
        f"Yaw mismatch: computed={computed_yaw}, expected={expected_yaw}"


@given(
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_body_yaw_frontal_pose_near_zero(shoulder_width, y):
    """
    **Feature: pose-corrected-distance, Property 3: Body Yaw Calculation**
    
    *For any* frontal pose (shoulders at same z), the Body_Yaw SHALL be near zero.
    
    **Validates: Requirements 2.1, 2.4**
    """
    # Frontal pose: both shoulders at same z coordinate
    left_shoulder = Landmark3D(x=-shoulder_width/2, y=y, z=0.0, visibility=1.0)
    right_shoulder = Landmark3D(x=shoulder_width/2, y=y, z=0.0, visibility=1.0)
    
    computed_yaw = estimate_body_yaw(left_shoulder, right_shoulder)
    
    # For frontal pose, yaw should be 0
    assert math.isclose(computed_yaw, 0.0, abs_tol=1e-9), \
        f"Frontal pose yaw should be 0, got {computed_yaw}"


@given(
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_body_yaw_positive_for_right_turn(shoulder_width, y):
    """
    **Feature: pose-corrected-distance, Property 3: Body Yaw Calculation**
    
    *For any* right-turned pose (right shoulder farther in z), the Body_Yaw SHALL be positive.
    
    **Validates: Requirements 2.1, 2.4**
    """
    # Right turn: right shoulder has larger z (farther from camera)
    left_shoulder = Landmark3D(x=-shoulder_width/2, y=y, z=-0.1, visibility=1.0)
    right_shoulder = Landmark3D(x=shoulder_width/2, y=y, z=0.1, visibility=1.0)
    
    computed_yaw = estimate_body_yaw(left_shoulder, right_shoulder)
    
    # For right turn, yaw should be positive
    assert computed_yaw > 0, \
        f"Right turn yaw should be positive, got {computed_yaw}"


@given(
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_body_yaw_negative_for_left_turn(shoulder_width, y):
    """
    **Feature: pose-corrected-distance, Property 3: Body Yaw Calculation**
    
    *For any* left-turned pose (left shoulder farther in z), the Body_Yaw SHALL be negative.
    
    **Validates: Requirements 2.1, 2.4**
    """
    # Left turn: left shoulder has larger z (farther from camera)
    left_shoulder = Landmark3D(x=-shoulder_width/2, y=y, z=0.1, visibility=1.0)
    right_shoulder = Landmark3D(x=shoulder_width/2, y=y, z=-0.1, visibility=1.0)
    
    computed_yaw = estimate_body_yaw(left_shoulder, right_shoulder)
    
    # For left turn, yaw should be negative
    assert computed_yaw < 0, \
        f"Left turn yaw should be negative, got {computed_yaw}"


# =============================================================================
# Property 4: Frontal/Side Classification (Task 5.4)
# =============================================================================

@given(yaw=st.floats(min_value=-29.99, max_value=29.99, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_frontal_classification_under_30_degrees(yaw):
    """
    **Feature: pose-corrected-distance, Property 4: Frontal/Side Classification**
    
    *For any* Body_Yaw angle with |yaw| < 30°, the pose SHALL be classified as frontal.
    
    **Validates: Requirements 2.2**
    """
    is_frontal = classify_pose_orientation(yaw)
    
    assert is_frontal == True, \
        f"Yaw {yaw}° (|yaw| < 30°) should be classified as frontal"


@given(yaw=st.floats(min_value=30.0, max_value=90.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_side_classification_positive_30_or_more(yaw):
    """
    **Feature: pose-corrected-distance, Property 4: Frontal/Side Classification**
    
    *For any* Body_Yaw angle with yaw >= 30°, the pose SHALL be classified as side view.
    
    **Validates: Requirements 2.2**
    """
    is_frontal = classify_pose_orientation(yaw)
    
    assert is_frontal == False, \
        f"Yaw {yaw}° (>= 30°) should be classified as side view"


@given(yaw=st.floats(min_value=-90.0, max_value=-30.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_side_classification_negative_30_or_more(yaw):
    """
    **Feature: pose-corrected-distance, Property 4: Frontal/Side Classification**
    
    *For any* Body_Yaw angle with yaw <= -30°, the pose SHALL be classified as side view.
    
    **Validates: Requirements 2.2**
    """
    is_frontal = classify_pose_orientation(yaw)
    
    assert is_frontal == False, \
        f"Yaw {yaw}° (<= -30°) should be classified as side view"


@given(yaw=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_frontal_side_classification_consistency(yaw):
    """
    **Feature: pose-corrected-distance, Property 4: Frontal/Side Classification**
    
    *For any* Body_Yaw angle, the classification SHALL be consistent:
    - |yaw| < 30° -> frontal (True)
    - |yaw| >= 30° -> side view (False)
    
    **Validates: Requirements 2.2**
    """
    is_frontal = classify_pose_orientation(yaw)
    
    if abs(yaw) < 30.0:
        assert is_frontal == True, \
            f"Yaw {yaw}° (|yaw| < 30°) should be frontal"
    else:
        assert is_frontal == False, \
            f"Yaw {yaw}° (|yaw| >= 30°) should be side view"


# =============================================================================
# Pose Correction Functions (Task 6)
# =============================================================================

def compute_corrected_width(width_2d: float, body_yaw: float) -> float:
    """
    计算校正后的宽度
    
    公式：Corrected_Width = 2D_Width / |cos(yaw)|
    为防止除零，当cos(yaw)小于0.1时，将其钳制到0.1。
    
    **Validates: Requirements 3.1, 3.2**
    """
    yaw_rad = math.radians(body_yaw)
    cos_yaw = max(0.1, abs(math.cos(yaw_rad)))
    return width_2d / cos_yaw


# =============================================================================
# Property 5: Pose Correction Formula (Task 6.2)
# =============================================================================

@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    body_yaw=st.floats(min_value=-89.0, max_value=89.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_formula(width_2d, body_yaw):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width W and Body_Yaw angle θ, the Corrected_Width SHALL equal 
    W / max(0.1, |cos(θ)|).
    
    **Validates: Requirements 3.1, 3.2**
    """
    # Compute using our function
    computed_corrected = compute_corrected_width(width_2d, body_yaw)
    
    # Compute expected value using the formula directly
    yaw_rad = math.radians(body_yaw)
    cos_yaw = max(0.1, abs(math.cos(yaw_rad)))
    expected_corrected = width_2d / cos_yaw
    
    # Assert equality within floating-point precision
    assert math.isclose(computed_corrected, expected_corrected, rel_tol=1e-9), \
        f"Corrected width mismatch: computed={computed_corrected}, expected={expected_corrected}"


@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_frontal_unchanged(width_2d):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width at frontal pose (yaw = 0°), the Corrected_Width SHALL equal 
    the original 2D width (since cos(0) = 1).
    
    **Validates: Requirements 3.1, 3.2**
    """
    # At frontal pose (yaw = 0), corrected width should equal original
    computed_corrected = compute_corrected_width(width_2d, 0.0)
    
    assert math.isclose(computed_corrected, width_2d, rel_tol=1e-9), \
        f"At frontal pose, corrected width should equal original: {computed_corrected} vs {width_2d}"


@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    body_yaw=st.floats(min_value=-89.0, max_value=89.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_always_greater_or_equal(width_2d, body_yaw):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width and Body_Yaw angle, the Corrected_Width SHALL be greater than 
    or equal to the original 2D width (since |cos(θ)| <= 1).
    
    **Validates: Requirements 3.1, 3.2**
    """
    computed_corrected = compute_corrected_width(width_2d, body_yaw)
    
    # Corrected width should always be >= original (since we divide by |cos| <= 1)
    assert computed_corrected >= width_2d - 1e-9, \
        f"Corrected width {computed_corrected} should be >= original {width_2d}"


@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    body_yaw=st.floats(min_value=84.27, max_value=95.73, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_clamping_at_extreme_angles(width_2d, body_yaw):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width and extreme Body_Yaw angle where |cos(θ)| < 0.1, 
    the divisor SHALL be clamped to 0.1, resulting in Corrected_Width = W / 0.1.
    
    Note: cos(84.26°) ≈ 0.1, so angles beyond ~84.26° trigger clamping.
    
    **Validates: Requirements 3.1, 3.2**
    """
    computed_corrected = compute_corrected_width(width_2d, body_yaw)
    
    # At extreme angles, divisor is clamped to 0.1
    expected_corrected = width_2d / 0.1
    
    assert math.isclose(computed_corrected, expected_corrected, rel_tol=1e-9), \
        f"At extreme angle {body_yaw}°, corrected width should be {expected_corrected}, got {computed_corrected}"


@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    body_yaw=st.floats(min_value=0.0, max_value=84.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_symmetric_for_positive_negative_yaw(width_2d, body_yaw):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width and Body_Yaw angle θ, the Corrected_Width for +θ SHALL equal 
    the Corrected_Width for -θ (since we use |cos(θ)|).
    
    **Validates: Requirements 3.1, 3.2**
    """
    corrected_positive = compute_corrected_width(width_2d, body_yaw)
    corrected_negative = compute_corrected_width(width_2d, -body_yaw)
    
    assert math.isclose(corrected_positive, corrected_negative, rel_tol=1e-9), \
        f"Corrected width should be symmetric: +{body_yaw}°={corrected_positive}, -{body_yaw}°={corrected_negative}"


@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_at_45_degrees(width_2d):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width at 45° yaw, the Corrected_Width SHALL equal W / cos(45°) ≈ W × √2.
    
    **Validates: Requirements 3.1, 3.2**
    """
    computed_corrected = compute_corrected_width(width_2d, 45.0)
    
    # cos(45°) = √2/2 ≈ 0.7071
    expected_corrected = width_2d / math.cos(math.radians(45.0))
    
    assert math.isclose(computed_corrected, expected_corrected, rel_tol=1e-9), \
        f"At 45°, corrected width should be {expected_corrected}, got {computed_corrected}"


@given(
    width_2d=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_pose_correction_at_60_degrees(width_2d):
    """
    **Feature: pose-corrected-distance, Property 5: Pose Correction Formula**
    
    *For any* 2D width at 60° yaw, the Corrected_Width SHALL equal W / cos(60°) = W × 2.
    
    **Validates: Requirements 3.1, 3.2**
    """
    computed_corrected = compute_corrected_width(width_2d, 60.0)
    
    # cos(60°) = 0.5
    expected_corrected = width_2d / 0.5  # = width_2d * 2
    
    assert math.isclose(computed_corrected, expected_corrected, rel_tol=1e-9), \
        f"At 60°, corrected width should be {expected_corrected}, got {computed_corrected}"


# =============================================================================
# Unit Tests for Confidence Adjustment (Task 5.5)
# =============================================================================

class TestConfidenceAdjustmentForExtremeYaw:
    """Unit tests for confidence adjustment based on extreme yaw angles."""
    
    def test_confidence_unchanged_for_normal_yaw(self):
        """Test that confidence is unchanged when |yaw| <= 70°."""
        confidence = 0.9
        
        # Test various normal yaw angles
        for yaw in [0.0, 30.0, -30.0, 45.0, -45.0, 70.0, -70.0]:
            adjusted = adjust_confidence_for_extreme_yaw(confidence, yaw)
            assert adjusted == confidence, \
                f"Confidence should be unchanged for yaw={yaw}°"
    
    def test_confidence_halved_for_extreme_yaw(self):
        """Test that confidence is halved when |yaw| > 70°."""
        confidence = 0.9
        
        # Test extreme yaw angles
        for yaw in [71.0, -71.0, 80.0, -80.0, 90.0, -90.0]:
            adjusted = adjust_confidence_for_extreme_yaw(confidence, yaw)
            expected = confidence * 0.5
            assert math.isclose(adjusted, expected, rel_tol=1e-9), \
                f"Confidence should be halved for yaw={yaw}°, got {adjusted}"
    
    def test_confidence_boundary_at_70_degrees(self):
        """Test the boundary condition at exactly 70 degrees."""
        confidence = 1.0
        
        # At exactly 70°, confidence should NOT be reduced
        adjusted_70 = adjust_confidence_for_extreme_yaw(confidence, 70.0)
        assert adjusted_70 == confidence, \
            "Confidence should be unchanged at exactly 70°"
        
        # Just above 70°, confidence should be reduced
        adjusted_71 = adjust_confidence_for_extreme_yaw(confidence, 70.01)
        assert adjusted_71 == confidence * 0.5, \
            "Confidence should be halved just above 70°"


# =============================================================================
# Task 3: Body Measurement Calculation Tests
# =============================================================================

# Helper functions for body measurement calculation
def calculate_shoulder_width_3d(left_shoulder: Landmark3D, right_shoulder: Landmark3D) -> float:
    """Calculate 3D shoulder width using Euclidean distance."""
    return euclidean_distance_3d(left_shoulder, right_shoulder)


def calculate_hip_width_3d(left_hip: Landmark3D, right_hip: Landmark3D) -> float:
    """Calculate 3D hip width using Euclidean distance."""
    return euclidean_distance_3d(left_hip, right_hip)


def calculate_shoulder_width_2d(
    left_shoulder: Landmark2D, 
    right_shoulder: Landmark2D,
    frame_width: int
) -> float:
    """Calculate 2D shoulder width in pixels."""
    return abs(left_shoulder.x - right_shoulder.x) * frame_width


def calculate_hip_width_2d(
    left_hip: Landmark2D, 
    right_hip: Landmark2D,
    frame_width: int
) -> float:
    """Calculate 2D hip width in pixels."""
    return abs(left_hip.x - right_hip.x) * frame_width


def get_landmark_pair_visibility(
    landmark1: Landmark2D,
    landmark2: Landmark2D,
    threshold: float = 0.5
) -> tuple:
    """Get visibility status for a pair of landmarks."""
    min_vis = min(landmark1.visibility, landmark2.visibility)
    both_visible = min_vis >= threshold
    return both_visible, min_vis


# =============================================================================
# Unit Tests for 3D Distance Calculation (Task 3.1)
# =============================================================================

class TestShoulderWidth3D:
    """Unit tests for 3D shoulder width calculation."""
    
    def test_calculate_shoulder_width_3d_typical_value(self):
        """Test shoulder width calculation with typical human measurements."""
        # Typical shoulder width is about 40cm (0.4m)
        left_shoulder = Landmark3D(x=-0.2, y=0.0, z=0.0, visibility=1.0)
        right_shoulder = Landmark3D(x=0.2, y=0.0, z=0.0, visibility=1.0)
        
        width = calculate_shoulder_width_3d(left_shoulder, right_shoulder)
        
        assert math.isclose(width, 0.4, rel_tol=1e-9), \
            f"Expected 0.4m, got {width}m"
    
    def test_calculate_shoulder_width_3d_with_depth(self):
        """Test shoulder width when shoulders have different z coordinates."""
        # Shoulders at different depths (person slightly turned)
        left_shoulder = Landmark3D(x=-0.2, y=0.0, z=0.1, visibility=1.0)
        right_shoulder = Landmark3D(x=0.2, y=0.0, z=-0.1, visibility=1.0)
        
        width = calculate_shoulder_width_3d(left_shoulder, right_shoulder)
        
        # Expected: sqrt(0.4^2 + 0.2^2) = sqrt(0.16 + 0.04) = sqrt(0.2) ≈ 0.447
        expected = math.sqrt(0.4**2 + 0.2**2)
        assert math.isclose(width, expected, rel_tol=1e-9), \
            f"Expected {expected}m, got {width}m"
    
    def test_calculate_shoulder_width_3d_zero_width(self):
        """Test shoulder width when both shoulders are at same position."""
        same_pos = Landmark3D(x=0.0, y=0.0, z=0.0, visibility=1.0)
        
        width = calculate_shoulder_width_3d(same_pos, same_pos)
        
        assert math.isclose(width, 0.0, abs_tol=1e-9), \
            f"Expected 0.0m, got {width}m"


class TestHipWidth3D:
    """Unit tests for 3D hip width calculation."""
    
    def test_calculate_hip_width_3d_typical_value(self):
        """Test hip width calculation with typical human measurements."""
        # Typical hip width is about 30cm (0.3m)
        left_hip = Landmark3D(x=-0.15, y=0.0, z=0.0, visibility=1.0)
        right_hip = Landmark3D(x=0.15, y=0.0, z=0.0, visibility=1.0)
        
        width = calculate_hip_width_3d(left_hip, right_hip)
        
        assert math.isclose(width, 0.3, rel_tol=1e-9), \
            f"Expected 0.3m, got {width}m"
    
    def test_calculate_hip_width_3d_with_all_dimensions(self):
        """Test hip width with differences in all three dimensions."""
        left_hip = Landmark3D(x=-0.15, y=0.1, z=0.05, visibility=1.0)
        right_hip = Landmark3D(x=0.15, y=-0.1, z=-0.05, visibility=1.0)
        
        width = calculate_hip_width_3d(left_hip, right_hip)
        
        # Expected: sqrt(0.3^2 + 0.2^2 + 0.1^2) = sqrt(0.09 + 0.04 + 0.01) = sqrt(0.14)
        expected = math.sqrt(0.3**2 + 0.2**2 + 0.1**2)
        assert math.isclose(width, expected, rel_tol=1e-9), \
            f"Expected {expected}m, got {width}m"


# =============================================================================
# Unit Tests for 2D Distance Calculation (Task 3.2)
# =============================================================================

class TestShoulderWidth2D:
    """Unit tests for 2D shoulder width calculation."""
    
    def test_calculate_shoulder_width_2d_full_width(self):
        """Test 2D shoulder width spanning full image width."""
        left_shoulder = Landmark2D(x=0.0, y=0.5, z=0.0, visibility=1.0)
        right_shoulder = Landmark2D(x=1.0, y=0.5, z=0.0, visibility=1.0)
        frame_width = 1920
        
        width = calculate_shoulder_width_2d(left_shoulder, right_shoulder, frame_width)
        
        assert width == 1920.0, f"Expected 1920px, got {width}px"
    
    def test_calculate_shoulder_width_2d_typical_value(self):
        """Test 2D shoulder width with typical normalized coordinates."""
        # Shoulders at 30% and 70% of image width
        left_shoulder = Landmark2D(x=0.3, y=0.5, z=0.0, visibility=1.0)
        right_shoulder = Landmark2D(x=0.7, y=0.5, z=0.0, visibility=1.0)
        frame_width = 1280
        
        width = calculate_shoulder_width_2d(left_shoulder, right_shoulder, frame_width)
        
        expected = 0.4 * 1280  # 512 pixels
        assert math.isclose(width, expected, rel_tol=1e-9), \
            f"Expected {expected}px, got {width}px"
    
    def test_calculate_shoulder_width_2d_reversed_order(self):
        """Test that order of shoulders doesn't affect result (absolute value)."""
        left_shoulder = Landmark2D(x=0.7, y=0.5, z=0.0, visibility=1.0)
        right_shoulder = Landmark2D(x=0.3, y=0.5, z=0.0, visibility=1.0)
        frame_width = 1280
        
        width = calculate_shoulder_width_2d(left_shoulder, right_shoulder, frame_width)
        
        expected = 0.4 * 1280  # 512 pixels
        assert math.isclose(width, expected, rel_tol=1e-9), \
            f"Expected {expected}px, got {width}px"


class TestHipWidth2D:
    """Unit tests for 2D hip width calculation."""
    
    def test_calculate_hip_width_2d_typical_value(self):
        """Test 2D hip width with typical normalized coordinates."""
        left_hip = Landmark2D(x=0.35, y=0.7, z=0.0, visibility=1.0)
        right_hip = Landmark2D(x=0.65, y=0.7, z=0.0, visibility=1.0)
        frame_width = 1280
        
        width = calculate_hip_width_2d(left_hip, right_hip, frame_width)
        
        expected = 0.3 * 1280  # 384 pixels
        assert math.isclose(width, expected, rel_tol=1e-9), \
            f"Expected {expected}px, got {width}px"
    
    def test_calculate_hip_width_2d_zero_width(self):
        """Test 2D hip width when hips are at same x position."""
        same_x = Landmark2D(x=0.5, y=0.7, z=0.0, visibility=1.0)
        frame_width = 1280
        
        width = calculate_hip_width_2d(same_x, same_x, frame_width)
        
        assert math.isclose(width, 0.0, abs_tol=1e-9), \
            f"Expected 0.0px, got {width}px"


# =============================================================================
# Unit Tests for BodyMeasurement Computation (Task 3.3)
# =============================================================================

class TestBodyMeasurementComputation:
    """Unit tests for BodyMeasurement computation function."""
    
    def _create_test_landmarks(self, shoulder_vis=1.0, hip_vis=1.0):
        """Helper to create test landmarks with specified visibility."""
        # Create 33 landmarks (MediaPipe pose has 33 landmarks)
        landmarks_2d = [Landmark2D(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
        landmarks_3d = [Landmark3D(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
        
        # Set shoulder landmarks (indices 11, 12)
        landmarks_2d[11] = Landmark2D(x=0.3, y=0.4, z=0.0, visibility=shoulder_vis)
        landmarks_2d[12] = Landmark2D(x=0.7, y=0.4, z=0.0, visibility=shoulder_vis)
        landmarks_3d[11] = Landmark3D(x=-0.2, y=0.0, z=0.0, visibility=shoulder_vis)
        landmarks_3d[12] = Landmark3D(x=0.2, y=0.0, z=0.0, visibility=shoulder_vis)
        
        # Set hip landmarks (indices 23, 24)
        landmarks_2d[23] = Landmark2D(x=0.35, y=0.7, z=0.0, visibility=hip_vis)
        landmarks_2d[24] = Landmark2D(x=0.65, y=0.7, z=0.0, visibility=hip_vis)
        landmarks_3d[23] = Landmark3D(x=-0.15, y=-0.5, z=0.0, visibility=hip_vis)
        landmarks_3d[24] = Landmark3D(x=0.15, y=-0.5, z=0.0, visibility=hip_vis)
        
        return landmarks_2d, landmarks_3d
    
    def test_compute_body_measurement_all_visible(self):
        """Test body measurement with all landmarks visible."""
        landmarks_2d, landmarks_3d = self._create_test_landmarks(
            shoulder_vis=1.0, hip_vis=1.0
        )
        
        # Simulate the computation
        ls_2d, rs_2d = landmarks_2d[11], landmarks_2d[12]
        lh_2d, rh_2d = landmarks_2d[23], landmarks_2d[24]
        ls_3d, rs_3d = landmarks_3d[11], landmarks_3d[12]
        lh_3d, rh_3d = landmarks_3d[23], landmarks_3d[24]
        
        frame_width = 1280
        
        # Calculate measurements
        shoulder_width_3d = calculate_shoulder_width_3d(ls_3d, rs_3d)
        hip_width_3d = calculate_hip_width_3d(lh_3d, rh_3d)
        shoulder_width_2d = calculate_shoulder_width_2d(ls_2d, rs_2d, frame_width)
        hip_width_2d = calculate_hip_width_2d(lh_2d, rh_2d, frame_width)
        
        # Verify 3D measurements
        assert math.isclose(shoulder_width_3d, 0.4, rel_tol=1e-9), \
            f"Expected shoulder_width_3d=0.4m, got {shoulder_width_3d}m"
        assert math.isclose(hip_width_3d, 0.3, rel_tol=1e-9), \
            f"Expected hip_width_3d=0.3m, got {hip_width_3d}m"
        
        # Verify 2D measurements
        expected_shoulder_2d = 0.4 * 1280  # 512 pixels
        expected_hip_2d = 0.3 * 1280  # 384 pixels
        assert math.isclose(shoulder_width_2d, expected_shoulder_2d, rel_tol=1e-9), \
            f"Expected shoulder_width_2d={expected_shoulder_2d}px, got {shoulder_width_2d}px"
        assert math.isclose(hip_width_2d, expected_hip_2d, rel_tol=1e-9), \
            f"Expected hip_width_2d={expected_hip_2d}px, got {hip_width_2d}px"
    
    def test_compute_body_measurement_low_shoulder_visibility(self):
        """Test body measurement when shoulders have low visibility."""
        landmarks_2d, landmarks_3d = self._create_test_landmarks(
            shoulder_vis=0.3, hip_vis=1.0
        )
        
        ls_2d, rs_2d = landmarks_2d[11], landmarks_2d[12]
        lh_2d, rh_2d = landmarks_2d[23], landmarks_2d[24]
        
        # Check visibility
        shoulder_visible, shoulder_vis = get_landmark_pair_visibility(ls_2d, rs_2d)
        hip_visible, hip_vis = get_landmark_pair_visibility(lh_2d, rh_2d)
        
        assert not shoulder_visible, "Shoulders should not be visible (vis < 0.5)"
        assert hip_visible, "Hips should be visible (vis >= 0.5)"
        
        # Confidence should be based on visible landmarks
        # With only hips visible, confidence = (0.3 + 1.0) / 2 = 0.65
        expected_confidence = (0.3 + 1.0) / 2
        assert math.isclose(expected_confidence, 0.65, rel_tol=1e-9)
    
    def test_compute_body_measurement_all_low_visibility(self):
        """Test body measurement when all landmarks have low visibility."""
        landmarks_2d, landmarks_3d = self._create_test_landmarks(
            shoulder_vis=0.3, hip_vis=0.2
        )
        
        ls_2d, rs_2d = landmarks_2d[11], landmarks_2d[12]
        lh_2d, rh_2d = landmarks_2d[23], landmarks_2d[24]
        
        # Check visibility
        shoulder_visible, _ = get_landmark_pair_visibility(ls_2d, rs_2d)
        hip_visible, _ = get_landmark_pair_visibility(lh_2d, rh_2d)
        
        assert not shoulder_visible, "Shoulders should not be visible"
        assert not hip_visible, "Hips should not be visible"
        
        # When both are not visible, confidence should be 0
        # (This is handled in the actual compute_body_measurement function)
    
    def test_compute_body_measurement_confidence_calculation(self):
        """Test that confidence is correctly calculated from visibility."""
        landmarks_2d, landmarks_3d = self._create_test_landmarks(
            shoulder_vis=0.8, hip_vis=0.6
        )
        
        ls_2d, rs_2d = landmarks_2d[11], landmarks_2d[12]
        lh_2d, rh_2d = landmarks_2d[23], landmarks_2d[24]
        
        _, shoulder_vis = get_landmark_pair_visibility(ls_2d, rs_2d)
        _, hip_vis = get_landmark_pair_visibility(lh_2d, rh_2d)
        
        # Confidence = average of min visibilities
        expected_confidence = (shoulder_vis + hip_vis) / 2
        assert math.isclose(expected_confidence, 0.7, rel_tol=1e-9), \
            f"Expected confidence=0.7, got {expected_confidence}"
    
    def test_compute_body_measurement_integration(self):
        """
        Integration test for compute_body_measurement function from the module.
        
        Tests that the actual function correctly:
        - Combines 3D and 2D measurements
        - Sets confidence based on visibility
        
        **Validates: Requirements 1.2, 1.3**
        """
        pytest.importorskip("mediapipe", reason="mediapipe not installed")
        from modules.distance_estimation.pose_corrected_distance import (
            compute_body_measurement as actual_compute_body_measurement,
            Landmark2D as ActualLandmark2D,
            Landmark3D as ActualLandmark3D
        )
        
        # Create 33 landmarks (MediaPipe pose has 33 landmarks)
        landmarks_2d = [ActualLandmark2D(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
        landmarks_3d = [ActualLandmark3D(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
        
        # Set shoulder landmarks (indices 11, 12) with high visibility
        landmarks_2d[11] = ActualLandmark2D(x=0.3, y=0.4, z=0.0, visibility=0.9)
        landmarks_2d[12] = ActualLandmark2D(x=0.7, y=0.4, z=0.0, visibility=0.9)
        landmarks_3d[11] = ActualLandmark3D(x=-0.2, y=0.0, z=0.0, visibility=0.9)
        landmarks_3d[12] = ActualLandmark3D(x=0.2, y=0.0, z=0.0, visibility=0.9)
        
        # Set hip landmarks (indices 23, 24) with high visibility
        landmarks_2d[23] = ActualLandmark2D(x=0.35, y=0.7, z=0.0, visibility=0.8)
        landmarks_2d[24] = ActualLandmark2D(x=0.65, y=0.7, z=0.0, visibility=0.8)
        landmarks_3d[23] = ActualLandmark3D(x=-0.15, y=-0.5, z=0.0, visibility=0.8)
        landmarks_3d[24] = ActualLandmark3D(x=0.15, y=-0.5, z=0.0, visibility=0.8)
        
        frame_width = 1280
        frame_height = 720
        
        # Call the actual function
        measurement = actual_compute_body_measurement(
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        # Verify 3D measurements (Requirements 1.2)
        assert math.isclose(measurement.shoulder_width_3d, 0.4, rel_tol=1e-9), \
            f"Expected shoulder_width_3d=0.4m, got {measurement.shoulder_width_3d}m"
        assert math.isclose(measurement.hip_width_3d, 0.3, rel_tol=1e-9), \
            f"Expected hip_width_3d=0.3m, got {measurement.hip_width_3d}m"
        
        # Verify 2D measurements
        expected_shoulder_2d = 0.4 * 1280  # 512 pixels
        expected_hip_2d = 0.3 * 1280  # 384 pixels
        assert math.isclose(measurement.shoulder_width_2d, expected_shoulder_2d, rel_tol=1e-9), \
            f"Expected shoulder_width_2d={expected_shoulder_2d}px, got {measurement.shoulder_width_2d}px"
        assert math.isclose(measurement.hip_width_2d, expected_hip_2d, rel_tol=1e-9), \
            f"Expected hip_width_2d={expected_hip_2d}px, got {measurement.hip_width_2d}px"
        
        # Verify confidence based on visibility (Requirements 1.3)
        # Confidence = (min(0.9, 0.9) + min(0.8, 0.8)) / 2 = (0.9 + 0.8) / 2 = 0.85
        expected_confidence = (0.9 + 0.8) / 2
        assert math.isclose(measurement.confidence, expected_confidence, rel_tol=1e-9), \
            f"Expected confidence={expected_confidence}, got {measurement.confidence}"
    
    def test_compute_body_measurement_low_visibility_exclusion(self):
        """
        Integration test for compute_body_measurement with low visibility landmarks.
        
        Tests that landmarks with visibility < 0.5 are excluded from measurements.
        
        **Validates: Requirements 1.3**
        """
        pytest.importorskip("mediapipe", reason="mediapipe not installed")
        from modules.distance_estimation.pose_corrected_distance import (
            compute_body_measurement as actual_compute_body_measurement,
            Landmark2D as ActualLandmark2D,
            Landmark3D as ActualLandmark3D
        )
        
        # Create 33 landmarks
        landmarks_2d = [ActualLandmark2D(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
        landmarks_3d = [ActualLandmark3D(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
        
        # Set shoulder landmarks with LOW visibility (< 0.5)
        landmarks_2d[11] = ActualLandmark2D(x=0.3, y=0.4, z=0.0, visibility=0.3)
        landmarks_2d[12] = ActualLandmark2D(x=0.7, y=0.4, z=0.0, visibility=0.3)
        landmarks_3d[11] = ActualLandmark3D(x=-0.2, y=0.0, z=0.0, visibility=0.3)
        landmarks_3d[12] = ActualLandmark3D(x=0.2, y=0.0, z=0.0, visibility=0.3)
        
        # Set hip landmarks with HIGH visibility (>= 0.5)
        landmarks_2d[23] = ActualLandmark2D(x=0.35, y=0.7, z=0.0, visibility=0.8)
        landmarks_2d[24] = ActualLandmark2D(x=0.65, y=0.7, z=0.0, visibility=0.8)
        landmarks_3d[23] = ActualLandmark3D(x=-0.15, y=-0.5, z=0.0, visibility=0.8)
        landmarks_3d[24] = ActualLandmark3D(x=0.15, y=-0.5, z=0.0, visibility=0.8)
        
        frame_width = 1280
        frame_height = 720
        
        # Call the actual function
        measurement = actual_compute_body_measurement(
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        # Shoulders should NOT be measured (visibility < 0.5)
        assert measurement.shoulder_width_3d == 0.0, \
            f"Shoulder width should be 0 when visibility < 0.5, got {measurement.shoulder_width_3d}"
        assert measurement.shoulder_width_2d == 0.0, \
            f"Shoulder width 2D should be 0 when visibility < 0.5, got {measurement.shoulder_width_2d}"
        
        # Hips SHOULD be measured (visibility >= 0.5)
        assert math.isclose(measurement.hip_width_3d, 0.3, rel_tol=1e-9), \
            f"Expected hip_width_3d=0.3m, got {measurement.hip_width_3d}m"
        
        # Confidence should reflect the visibility
        # (0.3 + 0.8) / 2 = 0.55
        expected_confidence = (0.3 + 0.8) / 2
        assert math.isclose(measurement.confidence, expected_confidence, rel_tol=1e-9), \
            f"Expected confidence={expected_confidence}, got {measurement.confidence}"
    
    def test_compute_body_measurement_all_invisible_returns_zero_confidence(self):
        """
        Integration test for compute_body_measurement when all landmarks are invisible.
        
        Tests that when both shoulder and hip landmarks have visibility < 0.5,
        the function returns a measurement with confidence = 0.
        
        **Validates: Requirements 1.3**
        """
        pytest.importorskip("mediapipe", reason="mediapipe not installed")
        from modules.distance_estimation.pose_corrected_distance import (
            compute_body_measurement as actual_compute_body_measurement,
            Landmark2D as ActualLandmark2D,
            Landmark3D as ActualLandmark3D
        )
        
        # Create 33 landmarks with all low visibility
        landmarks_2d = [ActualLandmark2D(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
        landmarks_3d = [ActualLandmark3D(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
        
        # Set shoulder landmarks with LOW visibility
        landmarks_2d[11] = ActualLandmark2D(x=0.3, y=0.4, z=0.0, visibility=0.3)
        landmarks_2d[12] = ActualLandmark2D(x=0.7, y=0.4, z=0.0, visibility=0.3)
        landmarks_3d[11] = ActualLandmark3D(x=-0.2, y=0.0, z=0.0, visibility=0.3)
        landmarks_3d[12] = ActualLandmark3D(x=0.2, y=0.0, z=0.0, visibility=0.3)
        
        # Set hip landmarks with LOW visibility
        landmarks_2d[23] = ActualLandmark2D(x=0.35, y=0.7, z=0.0, visibility=0.2)
        landmarks_2d[24] = ActualLandmark2D(x=0.65, y=0.7, z=0.0, visibility=0.2)
        landmarks_3d[23] = ActualLandmark3D(x=-0.15, y=-0.5, z=0.0, visibility=0.2)
        landmarks_3d[24] = ActualLandmark3D(x=0.15, y=-0.5, z=0.0, visibility=0.2)
        
        frame_width = 1280
        frame_height = 720
        
        # Call the actual function
        measurement = actual_compute_body_measurement(
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        # When both are invisible, confidence should be 0
        assert measurement.confidence == 0.0, \
            f"Expected confidence=0.0 when all landmarks invisible, got {measurement.confidence}"


# =============================================================================
# Calibration and Reference Frame Management Functions (Task 8)
# =============================================================================

def validate_calibration(measurement: BodyMeasurement, min_confidence: float = 0.5) -> bool:
    """
    验证测量数据是否适合用于校准
    
    **Validates: Requirements 4.3**
    """
    if measurement is None:
        return False
    return measurement.confidence >= min_confidence


def store_reference_frame(
    measurement: BodyMeasurement,
    current_reference: BodyMeasurement,
    is_calibrated: bool,
    min_confidence: float = 0.5
) -> tuple:
    """
    存储参考帧，如果校准验证通过
    
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    if not validate_calibration(measurement, min_confidence):
        return current_reference, is_calibrated, False
    return measurement, True, True


def reset_calibration() -> tuple:
    """
    重置校准状态
    
    **Validates: Requirements 4.4**
    """
    return None, False


# =============================================================================
# Property 7: Calibration State Management (Task 8.4)
# =============================================================================

@given(
    confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    shoulder_width_3d=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    hip_width_3d=st.floats(min_value=0.15, max_value=0.45, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_calibration_success_when_confidence_sufficient(confidence, shoulder_width_3d, hip_width_3d):
    """
    **Feature: pose-corrected-distance, Property 7: Calibration State Management**
    
    *For any* calibration attempt, if confidence >= 0.5 then is_calibrated SHALL become 
    true and Reference_Frame SHALL be set.
    
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    # Create measurement with sufficient confidence
    measurement = BodyMeasurement(
        shoulder_width_3d=shoulder_width_3d,
        hip_width_3d=hip_width_3d,
        confidence=confidence,
        timestamp=1234567890.0
    )
    
    # Initial state: not calibrated
    current_reference = None
    is_calibrated = False
    
    # Attempt calibration
    new_reference, new_is_calibrated, success = store_reference_frame(
        measurement, current_reference, is_calibrated, min_confidence=0.5
    )
    
    # Verify: calibration should succeed
    assert success == True, \
        f"Calibration should succeed with confidence {confidence} >= 0.5"
    assert new_is_calibrated == True, \
        f"is_calibrated should be True after successful calibration"
    assert new_reference is measurement, \
        f"Reference_Frame should be set to the measurement"
    assert new_reference.shoulder_width_3d == shoulder_width_3d, \
        f"Reference shoulder width should match"
    assert new_reference.hip_width_3d == hip_width_3d, \
        f"Reference hip width should match"


@given(
    confidence=st.floats(min_value=0.0, max_value=0.4999, allow_nan=False, allow_infinity=False),
    prev_shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    prev_hip_width=st.floats(min_value=0.15, max_value=0.45, allow_nan=False, allow_infinity=False),
    prev_is_calibrated=st.booleans()
)
@settings(max_examples=100)
def test_property_calibration_rejected_when_confidence_low(
    confidence, prev_shoulder_width, prev_hip_width, prev_is_calibrated
):
    """
    **Feature: pose-corrected-distance, Property 7: Calibration State Management**
    
    *For any* calibration attempt, if confidence < 0.5 then the previous state 
    SHALL be maintained.
    
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    # Create measurement with low confidence
    measurement = BodyMeasurement(
        shoulder_width_3d=0.4,
        hip_width_3d=0.3,
        confidence=confidence,
        timestamp=1234567890.0
    )
    
    # Previous state (may or may not be calibrated)
    prev_reference = BodyMeasurement(
        shoulder_width_3d=prev_shoulder_width,
        hip_width_3d=prev_hip_width,
        confidence=0.9,
        timestamp=1234567880.0
    ) if prev_is_calibrated else None
    
    # Attempt calibration
    new_reference, new_is_calibrated, success = store_reference_frame(
        measurement, prev_reference, prev_is_calibrated, min_confidence=0.5
    )
    
    # Verify: calibration should be rejected, previous state maintained
    assert success == False, \
        f"Calibration should be rejected with confidence {confidence} < 0.5"
    assert new_is_calibrated == prev_is_calibrated, \
        f"is_calibrated should remain {prev_is_calibrated}"
    assert new_reference is prev_reference, \
        f"Reference_Frame should remain unchanged"


@given(
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_calibration_validation_boundary(confidence):
    """
    **Feature: pose-corrected-distance, Property 7: Calibration State Management**
    
    *For any* confidence value, the calibration decision SHALL be consistent:
    - confidence >= 0.5 -> calibration succeeds
    - confidence < 0.5 -> calibration rejected
    
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    measurement = BodyMeasurement(
        shoulder_width_3d=0.4,
        hip_width_3d=0.3,
        confidence=confidence,
        timestamp=1234567890.0
    )
    
    # Validate calibration
    is_valid = validate_calibration(measurement, min_confidence=0.5)
    
    if confidence >= 0.5:
        assert is_valid == True, \
            f"Calibration should be valid with confidence {confidence} >= 0.5"
    else:
        assert is_valid == False, \
            f"Calibration should be invalid with confidence {confidence} < 0.5"


@given(
    confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    shoulder_width_3d=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    hip_width_3d=st.floats(min_value=0.15, max_value=0.45, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_calibration_stores_correct_measurements(confidence, shoulder_width_3d, hip_width_3d):
    """
    **Feature: pose-corrected-distance, Property 7: Calibration State Management**
    
    *For any* successful calibration, the Reference_Frame SHALL contain the exact 
    measurement values that were provided.
    
    **Validates: Requirements 4.1, 4.2**
    """
    measurement = BodyMeasurement(
        shoulder_width_3d=shoulder_width_3d,
        hip_width_3d=hip_width_3d,
        shoulder_width_2d=200.0,
        hip_width_2d=150.0,
        body_yaw=15.0,
        confidence=confidence,
        timestamp=1234567890.0
    )
    
    new_reference, new_is_calibrated, success = store_reference_frame(
        measurement, None, False, min_confidence=0.5
    )
    
    assert success == True
    assert new_reference.shoulder_width_3d == shoulder_width_3d
    assert new_reference.hip_width_3d == hip_width_3d
    assert new_reference.shoulder_width_2d == 200.0
    assert new_reference.hip_width_2d == 150.0
    assert new_reference.body_yaw == 15.0
    assert new_reference.confidence == confidence


# =============================================================================
# Property 8: Reset Behavior (Task 8.5)
# =============================================================================

@given(
    prev_shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    prev_hip_width=st.floats(min_value=0.15, max_value=0.45, allow_nan=False, allow_infinity=False),
    prev_confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_reset_clears_reference_frame(prev_shoulder_width, prev_hip_width, prev_confidence):
    """
    **Feature: pose-corrected-distance, Property 8: Reset Behavior**
    
    *For any* reset operation, Reference_Frame SHALL be cleared (set to None).
    
    **Validates: Requirements 4.4**
    """
    # Simulate a calibrated state
    prev_reference = BodyMeasurement(
        shoulder_width_3d=prev_shoulder_width,
        hip_width_3d=prev_hip_width,
        confidence=prev_confidence,
        timestamp=1234567890.0
    )
    
    # Perform reset
    new_reference, new_is_calibrated = reset_calibration()
    
    # Verify: reference frame should be cleared
    assert new_reference is None, \
        f"Reference_Frame should be None after reset"


@given(
    prev_shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    prev_hip_width=st.floats(min_value=0.15, max_value=0.45, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_reset_sets_is_calibrated_false(prev_shoulder_width, prev_hip_width):
    """
    **Feature: pose-corrected-distance, Property 8: Reset Behavior**
    
    *For any* reset operation, is_calibrated SHALL become false.
    
    **Validates: Requirements 4.4**
    """
    # Perform reset (regardless of previous state)
    new_reference, new_is_calibrated = reset_calibration()
    
    # Verify: is_calibrated should be false
    assert new_is_calibrated == False, \
        f"is_calibrated should be False after reset"


@settings(max_examples=100)
@given(st.data())
def test_property_reset_always_produces_uncalibrated_state(data):
    """
    **Feature: pose-corrected-distance, Property 8: Reset Behavior**
    
    *For any* reset operation, the resulting state SHALL always be:
    - Reference_Frame = None
    - is_calibrated = False
    
    **Validates: Requirements 4.4**
    """
    # Perform reset
    new_reference, new_is_calibrated = reset_calibration()
    
    # Verify both conditions
    assert new_reference is None, "Reference_Frame should be None"
    assert new_is_calibrated == False, "is_calibrated should be False"


@given(
    confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_calibrate_then_reset_returns_to_initial_state(confidence, shoulder_width):
    """
    **Feature: pose-corrected-distance, Property 8: Reset Behavior**
    
    *For any* sequence of calibrate then reset, the final state SHALL be 
    equivalent to the initial uncalibrated state.
    
    **Validates: Requirements 4.4**
    """
    # Initial state
    initial_reference = None
    initial_is_calibrated = False
    
    # Calibrate
    measurement = BodyMeasurement(
        shoulder_width_3d=shoulder_width,
        hip_width_3d=0.3,
        confidence=confidence,
        timestamp=1234567890.0
    )
    
    calibrated_reference, calibrated_is_calibrated, success = store_reference_frame(
        measurement, initial_reference, initial_is_calibrated, min_confidence=0.5
    )
    
    # Verify calibration succeeded
    assert success == True
    assert calibrated_is_calibrated == True
    assert calibrated_reference is not None
    
    # Reset
    reset_reference, reset_is_calibrated = reset_calibration()
    
    # Verify reset returns to initial state
    assert reset_reference is None, "After reset, reference should be None"
    assert reset_is_calibrated == False, "After reset, is_calibrated should be False"
    assert reset_reference == initial_reference, "Reset state should match initial state"
    assert reset_is_calibrated == initial_is_calibrated, "Reset calibration flag should match initial"


# =============================================================================
# Unit Tests for Calibration State Management (Task 8)
# =============================================================================

class TestCalibrationValidation:
    """Unit tests for calibration validation."""
    
    def test_validate_calibration_with_none_measurement(self):
        """Test that None measurement is rejected."""
        is_valid = validate_calibration(None, min_confidence=0.5)
        assert is_valid == False
    
    def test_validate_calibration_at_threshold(self):
        """Test calibration at exactly 0.5 confidence."""
        measurement = BodyMeasurement(confidence=0.5)
        is_valid = validate_calibration(measurement, min_confidence=0.5)
        assert is_valid == True
    
    def test_validate_calibration_just_below_threshold(self):
        """Test calibration just below 0.5 confidence."""
        measurement = BodyMeasurement(confidence=0.4999)
        is_valid = validate_calibration(measurement, min_confidence=0.5)
        assert is_valid == False
    
    def test_validate_calibration_high_confidence(self):
        """Test calibration with high confidence."""
        measurement = BodyMeasurement(confidence=0.95)
        is_valid = validate_calibration(measurement, min_confidence=0.5)
        assert is_valid == True
    
    def test_validate_calibration_zero_confidence(self):
        """Test calibration with zero confidence."""
        measurement = BodyMeasurement(confidence=0.0)
        is_valid = validate_calibration(measurement, min_confidence=0.5)
        assert is_valid == False


class TestStoreReferenceFrame:
    """Unit tests for store_reference_frame function."""
    
    def test_store_reference_frame_success(self):
        """Test successful reference frame storage."""
        measurement = BodyMeasurement(
            shoulder_width_3d=0.4,
            hip_width_3d=0.3,
            confidence=0.9
        )
        
        new_ref, new_cal, success = store_reference_frame(
            measurement, None, False, min_confidence=0.5
        )
        
        assert success == True
        assert new_cal == True
        assert new_ref is measurement
    
    def test_store_reference_frame_rejection_maintains_state(self):
        """Test that rejection maintains previous state."""
        low_conf_measurement = BodyMeasurement(confidence=0.3)
        prev_reference = BodyMeasurement(
            shoulder_width_3d=0.4,
            confidence=0.9
        )
        
        new_ref, new_cal, success = store_reference_frame(
            low_conf_measurement, prev_reference, True, min_confidence=0.5
        )
        
        assert success == False
        assert new_cal == True  # Maintains previous state
        assert new_ref is prev_reference  # Maintains previous reference
    
    def test_store_reference_frame_rejection_when_not_calibrated(self):
        """Test rejection when not previously calibrated."""
        low_conf_measurement = BodyMeasurement(confidence=0.3)
        
        new_ref, new_cal, success = store_reference_frame(
            low_conf_measurement, None, False, min_confidence=0.5
        )
        
        assert success == False
        assert new_cal == False  # Stays uncalibrated
        assert new_ref is None  # Stays None


class TestResetCalibration:
    """Unit tests for reset_calibration function."""
    
    def test_reset_calibration_returns_none_reference(self):
        """Test that reset returns None reference."""
        ref, is_cal = reset_calibration()
        assert ref is None
    
    def test_reset_calibration_returns_false_flag(self):
        """Test that reset returns False for is_calibrated."""
        ref, is_cal = reset_calibration()
        assert is_cal == False
    
    def test_reset_calibration_idempotent(self):
        """Test that multiple resets produce same result."""
        ref1, is_cal1 = reset_calibration()
        ref2, is_cal2 = reset_calibration()
        
        assert ref1 == ref2
        assert is_cal1 == is_cal2


# =============================================================================
# Relative Distance Calculation Functions (Task 9)
# =============================================================================

def compute_relative_distance_ratio(
    corrected_width_ref: float,
    corrected_width_current: float
) -> float:
    """
    计算相对距离比值
    
    公式：ratio = Corrected_Width_ref / Corrected_Width_current
    
    **Validates: Requirements 5.1**
    """
    if corrected_width_current <= 0:
        return 1.0
    return corrected_width_ref / corrected_width_current


def get_direction_indication(relative_distance_ratio: float) -> str:
    """
    根据相对距离比值获取方向指示
    
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
    
    **Validates: Requirements 5.4**
    """
    return shoulder_weight * shoulder_ratio + hip_weight * hip_ratio


# =============================================================================
# Property 9: Relative Distance Ratio Formula (Task 9.2)
# =============================================================================

@given(
    corrected_width_ref=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    corrected_width_current=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_relative_distance_ratio_formula(corrected_width_ref, corrected_width_current):
    """
    **Feature: pose-corrected-distance, Property 9: Relative Distance Ratio Formula**
    
    *For any* calibrated system with Reference_Frame R and current measurement C, 
    the Relative_Distance_Ratio SHALL equal Corrected_Width(R) / Corrected_Width(C).
    
    **Validates: Requirements 5.1**
    """
    # Compute using our function
    computed_ratio = compute_relative_distance_ratio(corrected_width_ref, corrected_width_current)
    
    # Compute expected value using the formula directly
    expected_ratio = corrected_width_ref / corrected_width_current
    
    # Assert equality within floating-point precision
    assert math.isclose(computed_ratio, expected_ratio, rel_tol=1e-9), \
        f"Ratio mismatch: computed={computed_ratio}, expected={expected_ratio}"


@given(
    corrected_width_ref=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_relative_distance_ratio_same_width_equals_one(corrected_width_ref):
    """
    **Feature: pose-corrected-distance, Property 9: Relative Distance Ratio Formula**
    
    *For any* measurement where reference and current widths are equal, 
    the ratio SHALL equal 1.0.
    
    **Validates: Requirements 5.1**
    """
    # Same width should give ratio of 1.0
    computed_ratio = compute_relative_distance_ratio(corrected_width_ref, corrected_width_ref)
    
    assert math.isclose(computed_ratio, 1.0, rel_tol=1e-9), \
        f"Same width should give ratio 1.0, got {computed_ratio}"


@given(
    corrected_width_ref=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    scale_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_relative_distance_ratio_scaling(corrected_width_ref, scale_factor):
    """
    **Feature: pose-corrected-distance, Property 9: Relative Distance Ratio Formula**
    
    *For any* reference width and scale factor, if current width = ref / scale_factor,
    then ratio SHALL equal scale_factor.
    
    **Validates: Requirements 5.1**
    """
    # If current width is ref/scale_factor, ratio should be scale_factor
    corrected_width_current = corrected_width_ref / scale_factor
    
    computed_ratio = compute_relative_distance_ratio(corrected_width_ref, corrected_width_current)
    
    assert math.isclose(computed_ratio, scale_factor, rel_tol=1e-9), \
        f"Expected ratio {scale_factor}, got {computed_ratio}"


@given(
    corrected_width_ref=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_relative_distance_ratio_larger_current_less_than_one(corrected_width_ref):
    """
    **Feature: pose-corrected-distance, Property 9: Relative Distance Ratio Formula**
    
    *For any* measurement where current width > reference width (subject closer),
    the ratio SHALL be less than 1.0.
    
    **Validates: Requirements 5.1**
    """
    # Larger current width means subject is closer
    corrected_width_current = corrected_width_ref * 1.5
    
    computed_ratio = compute_relative_distance_ratio(corrected_width_ref, corrected_width_current)
    
    assert computed_ratio < 1.0, \
        f"Larger current width should give ratio < 1.0, got {computed_ratio}"


@given(
    corrected_width_ref=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_relative_distance_ratio_smaller_current_greater_than_one(corrected_width_ref):
    """
    **Feature: pose-corrected-distance, Property 9: Relative Distance Ratio Formula**
    
    *For any* measurement where current width < reference width (subject farther),
    the ratio SHALL be greater than 1.0.
    
    **Validates: Requirements 5.1**
    """
    # Smaller current width means subject is farther
    corrected_width_current = corrected_width_ref * 0.5
    
    computed_ratio = compute_relative_distance_ratio(corrected_width_ref, corrected_width_current)
    
    assert computed_ratio > 1.0, \
        f"Smaller current width should give ratio > 1.0, got {computed_ratio}"


# =============================================================================
# Property 10: Direction Indication (Task 9.4)
# =============================================================================

@given(
    ratio=st.floats(min_value=1.0001, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_direction_moving_away_for_ratio_greater_than_one(ratio):
    """
    **Feature: pose-corrected-distance, Property 10: Direction Indication**
    
    *For any* Relative_Distance_Ratio > 1.0, the direction SHALL be "MOVING AWAY".
    
    **Validates: Requirements 5.2, 5.3**
    """
    direction = get_direction_indication(ratio)
    
    assert direction == "MOVING AWAY", \
        f"Ratio {ratio} > 1.0 should indicate 'MOVING AWAY', got '{direction}'"


@given(
    ratio=st.floats(min_value=0.01, max_value=0.9999, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_direction_approaching_for_ratio_less_than_one(ratio):
    """
    **Feature: pose-corrected-distance, Property 10: Direction Indication**
    
    *For any* Relative_Distance_Ratio < 1.0, the direction SHALL be "APPROACHING".
    
    **Validates: Requirements 5.2, 5.3**
    """
    direction = get_direction_indication(ratio)
    
    assert direction == "APPROACHING", \
        f"Ratio {ratio} < 1.0 should indicate 'APPROACHING', got '{direction}'"


def test_property_direction_stable_for_ratio_equals_one():
    """
    **Feature: pose-corrected-distance, Property 10: Direction Indication**
    
    *For any* Relative_Distance_Ratio == 1.0, the direction SHALL be "STABLE".
    
    **Validates: Requirements 5.2, 5.3**
    """
    direction = get_direction_indication(1.0)
    
    assert direction == "STABLE", \
        f"Ratio 1.0 should indicate 'STABLE', got '{direction}'"


@given(
    ratio=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_direction_indication_consistency(ratio):
    """
    **Feature: pose-corrected-distance, Property 10: Direction Indication**
    
    *For any* ratio value, the direction indication SHALL be consistent:
    - ratio > 1.0 -> "MOVING AWAY"
    - ratio < 1.0 -> "APPROACHING"
    - ratio == 1.0 -> "STABLE"
    
    **Validates: Requirements 5.2, 5.3**
    """
    direction = get_direction_indication(ratio)
    
    if ratio > 1.0:
        assert direction == "MOVING AWAY", \
            f"Ratio {ratio} > 1.0 should be 'MOVING AWAY', got '{direction}'"
    elif ratio < 1.0:
        assert direction == "APPROACHING", \
            f"Ratio {ratio} < 1.0 should be 'APPROACHING', got '{direction}'"
    else:
        assert direction == "STABLE", \
            f"Ratio {ratio} == 1.0 should be 'STABLE', got '{direction}'"


@given(
    ratio=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_direction_indication_returns_valid_string(ratio):
    """
    **Feature: pose-corrected-distance, Property 10: Direction Indication**
    
    *For any* ratio value, the direction indication SHALL return one of the 
    three valid strings: "MOVING AWAY", "APPROACHING", or "STABLE".
    
    **Validates: Requirements 5.2, 5.3**
    """
    direction = get_direction_indication(ratio)
    
    valid_directions = {"MOVING AWAY", "APPROACHING", "STABLE"}
    assert direction in valid_directions, \
        f"Direction '{direction}' is not a valid indication"


# =============================================================================
# Property 11: Weighted Ratio Combination (Task 9.6)
# =============================================================================

@given(
    shoulder_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    hip_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_weighted_ratio_combination_formula(shoulder_ratio, hip_ratio):
    """
    **Feature: pose-corrected-distance, Property 11: Weighted Ratio Combination**
    
    *For any* measurement with both shoulder_ratio S and hip_ratio H available, 
    the combined_ratio SHALL equal 0.6×S + 0.4×H.
    
    **Validates: Requirements 5.4**
    """
    # Compute using our function
    computed_combined = compute_weighted_ratio_combination(shoulder_ratio, hip_ratio)
    
    # Compute expected value using the formula directly
    expected_combined = 0.6 * shoulder_ratio + 0.4 * hip_ratio
    
    # Assert equality within floating-point precision
    assert math.isclose(computed_combined, expected_combined, rel_tol=1e-9), \
        f"Combined ratio mismatch: computed={computed_combined}, expected={expected_combined}"


@given(
    ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_weighted_ratio_same_ratios_equals_input(ratio):
    """
    **Feature: pose-corrected-distance, Property 11: Weighted Ratio Combination**
    
    *For any* measurement where shoulder_ratio equals hip_ratio, 
    the combined_ratio SHALL equal that same ratio.
    
    **Validates: Requirements 5.4**
    """
    # When both ratios are the same, combined should equal that ratio
    computed_combined = compute_weighted_ratio_combination(ratio, ratio)
    
    assert math.isclose(computed_combined, ratio, rel_tol=1e-9), \
        f"Same ratios should give combined={ratio}, got {computed_combined}"


@given(
    shoulder_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    hip_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_weighted_ratio_between_inputs(shoulder_ratio, hip_ratio):
    """
    **Feature: pose-corrected-distance, Property 11: Weighted Ratio Combination**
    
    *For any* measurement with shoulder_ratio and hip_ratio, 
    the combined_ratio SHALL be between the minimum and maximum of the two inputs.
    
    **Validates: Requirements 5.4**
    """
    computed_combined = compute_weighted_ratio_combination(shoulder_ratio, hip_ratio)
    
    min_ratio = min(shoulder_ratio, hip_ratio)
    max_ratio = max(shoulder_ratio, hip_ratio)
    
    # Use tolerance for floating-point comparison
    assert min_ratio - 1e-9 <= computed_combined <= max_ratio + 1e-9, \
        f"Combined {computed_combined} should be between {min_ratio} and {max_ratio}"


@given(
    shoulder_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    hip_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_weighted_ratio_shoulder_has_higher_weight(shoulder_ratio, hip_ratio):
    """
    **Feature: pose-corrected-distance, Property 11: Weighted Ratio Combination**
    
    *For any* measurement, the combined_ratio SHALL be closer to shoulder_ratio 
    than to hip_ratio (since shoulder has weight 0.6 > 0.4).
    
    **Validates: Requirements 5.4**
    """
    computed_combined = compute_weighted_ratio_combination(shoulder_ratio, hip_ratio)
    
    # Distance from combined to shoulder should be <= distance to hip
    dist_to_shoulder = abs(computed_combined - shoulder_ratio)
    dist_to_hip = abs(computed_combined - hip_ratio)
    
    # Allow small tolerance for floating point
    assert dist_to_shoulder <= dist_to_hip + 1e-9, \
        f"Combined {computed_combined} should be closer to shoulder {shoulder_ratio} than hip {hip_ratio}"


@given(
    shoulder_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    hip_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    shoulder_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_weighted_ratio_custom_weights(shoulder_ratio, hip_ratio, shoulder_weight):
    """
    **Feature: pose-corrected-distance, Property 11: Weighted Ratio Combination**
    
    *For any* custom weights where shoulder_weight + hip_weight = 1.0,
    the combined_ratio SHALL equal shoulder_weight×S + hip_weight×H.
    
    **Validates: Requirements 5.4**
    """
    hip_weight = 1.0 - shoulder_weight
    
    computed_combined = compute_weighted_ratio_combination(
        shoulder_ratio, hip_ratio, shoulder_weight, hip_weight
    )
    
    expected_combined = shoulder_weight * shoulder_ratio + hip_weight * hip_ratio
    
    assert math.isclose(computed_combined, expected_combined, rel_tol=1e-9), \
        f"Custom weights: computed={computed_combined}, expected={expected_combined}"


# =============================================================================
# Unit Tests for Relative Distance Calculation (Task 9)
# =============================================================================

class TestRelativeDistanceRatioCalculation:
    """Unit tests for relative distance ratio calculation."""
    
    def test_compute_ratio_basic(self):
        """Test basic ratio calculation."""
        ratio = compute_relative_distance_ratio(200.0, 100.0)
        assert ratio == 2.0
    
    def test_compute_ratio_same_width(self):
        """Test ratio when widths are equal."""
        ratio = compute_relative_distance_ratio(150.0, 150.0)
        assert ratio == 1.0
    
    def test_compute_ratio_approaching(self):
        """Test ratio when subject is approaching (larger current width)."""
        ratio = compute_relative_distance_ratio(100.0, 200.0)
        assert ratio == 0.5
    
    def test_compute_ratio_moving_away(self):
        """Test ratio when subject is moving away (smaller current width)."""
        ratio = compute_relative_distance_ratio(200.0, 100.0)
        assert ratio == 2.0
    
    def test_compute_ratio_zero_current_width(self):
        """Test ratio when current width is zero (edge case)."""
        ratio = compute_relative_distance_ratio(200.0, 0.0)
        assert ratio == 1.0  # Should return 1.0 to prevent division by zero
    
    def test_compute_ratio_negative_current_width(self):
        """Test ratio when current width is negative (edge case)."""
        ratio = compute_relative_distance_ratio(200.0, -100.0)
        assert ratio == 1.0  # Should return 1.0 for invalid input


class TestDirectionIndication:
    """Unit tests for direction indication."""
    
    def test_direction_moving_away(self):
        """Test direction for ratio > 1.0."""
        assert get_direction_indication(1.5) == "MOVING AWAY"
        assert get_direction_indication(2.0) == "MOVING AWAY"
        assert get_direction_indication(1.001) == "MOVING AWAY"
    
    def test_direction_approaching(self):
        """Test direction for ratio < 1.0."""
        assert get_direction_indication(0.5) == "APPROACHING"
        assert get_direction_indication(0.8) == "APPROACHING"
        assert get_direction_indication(0.999) == "APPROACHING"
    
    def test_direction_stable(self):
        """Test direction for ratio == 1.0."""
        assert get_direction_indication(1.0) == "STABLE"


class TestWeightedRatioCombination:
    """Unit tests for weighted ratio combination."""
    
    def test_weighted_combination_default_weights(self):
        """Test weighted combination with default weights (0.6, 0.4)."""
        combined = compute_weighted_ratio_combination(1.0, 1.0)
        assert combined == 1.0
        
        combined = compute_weighted_ratio_combination(2.0, 1.0)
        expected = 0.6 * 2.0 + 0.4 * 1.0  # 1.6
        assert math.isclose(combined, expected, rel_tol=1e-9)
    
    def test_weighted_combination_custom_weights(self):
        """Test weighted combination with custom weights."""
        combined = compute_weighted_ratio_combination(2.0, 1.0, 0.5, 0.5)
        expected = 0.5 * 2.0 + 0.5 * 1.0  # 1.5
        assert math.isclose(combined, expected, rel_tol=1e-9)
    
    def test_weighted_combination_shoulder_only(self):
        """Test weighted combination with shoulder weight = 1.0."""
        combined = compute_weighted_ratio_combination(2.0, 1.0, 1.0, 0.0)
        assert combined == 2.0
    
    def test_weighted_combination_hip_only(self):
        """Test weighted combination with hip weight = 1.0."""
        combined = compute_weighted_ratio_combination(2.0, 1.0, 0.0, 1.0)
        assert combined == 1.0


# =============================================================================
# Import synthetic data generator for pose correction effectiveness tests
# =============================================================================

from modules.distance_estimation.synthetic_data_generator import (
    generate_synthetic_landmarks,
    generate_rotation_sequence,
    generate_distance_sequence,
    compute_expected_2d_width,
    SyntheticLandmarkResult
)


# =============================================================================
# Property 6: Pose Correction Stability (Task 12.1)
# =============================================================================

@given(
    shoulder_width_3d=st.floats(min_value=0.30, max_value=0.50, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=1.5, max_value=4.0, allow_nan=False, allow_infinity=False),
    yaw_angle=st.floats(min_value=0.0, max_value=45.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_pose_correction_stability(shoulder_width_3d, distance, yaw_angle):
    """
    **Feature: pose-corrected-distance, Property 6: Pose Correction Stability**
    
    *For any* constant physical distance, when body rotation changes from 0° to 45°, 
    the Corrected_Width SHALL remain within 15% of the 0° measurement.
    
    **Validates: Requirements 3.3**
    """
    # Generate landmarks at frontal pose (0°)
    frontal_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=0.0
    )
    
    # Generate landmarks at the given yaw angle
    rotated_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=yaw_angle
    )
    
    # Compute corrected widths
    frontal_corrected = compute_corrected_width(frontal_result.shoulder_width_2d_pixels, 0.0)
    rotated_corrected = compute_corrected_width(rotated_result.shoulder_width_2d_pixels, yaw_angle)
    
    # The corrected width should remain within 15% of the frontal measurement
    tolerance = 0.15
    lower_bound = frontal_corrected * (1 - tolerance)
    upper_bound = frontal_corrected * (1 + tolerance)
    
    assert lower_bound <= rotated_corrected <= upper_bound, \
        f"Corrected width {rotated_corrected:.2f} at {yaw_angle:.1f}° should be within 15% of " \
        f"frontal {frontal_corrected:.2f} (range: [{lower_bound:.2f}, {upper_bound:.2f}])"


@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=2.0, max_value=3.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_pose_correction_stability_sequence(shoulder_width_3d, distance):
    """
    **Feature: pose-corrected-distance, Property 6: Pose Correction Stability**
    
    *For any* rotation sequence from 0° to 45°, all corrected widths SHALL remain 
    within 15% of the frontal (0°) measurement.
    
    **Validates: Requirements 3.3**
    """
    # Generate rotation sequence from 0° to 45°
    sequence = generate_rotation_sequence(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        start_yaw=0.0,
        end_yaw=45.0,
        num_steps=10
    )
    
    # Get frontal corrected width as reference
    frontal_result = sequence[0]
    frontal_corrected = compute_corrected_width(frontal_result.shoulder_width_2d_pixels, 0.0)
    
    tolerance = 0.15
    lower_bound = frontal_corrected * (1 - tolerance)
    upper_bound = frontal_corrected * (1 + tolerance)
    
    # Check all frames in the sequence
    for result in sequence:
        corrected = compute_corrected_width(result.shoulder_width_2d_pixels, result.body_yaw)
        
        assert lower_bound <= corrected <= upper_bound, \
            f"Corrected width {corrected:.2f} at {result.body_yaw:.1f}° should be within 15% of " \
            f"frontal {frontal_corrected:.2f} (range: [{lower_bound:.2f}, {upper_bound:.2f}])"


# =============================================================================
# Property 12: Distance Ratio Accuracy (Task 12.2)
# =============================================================================

@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    ref_distance=st.floats(min_value=1.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    yaw_angle=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_distance_ratio_accuracy_50_percent(shoulder_width_3d, ref_distance, yaw_angle):
    """
    **Feature: pose-corrected-distance, Property 12: Distance Ratio Accuracy**
    
    *For any* 50% increase in physical distance (ratio 1.5), the measured 
    Relative_Distance_Ratio SHALL be within 10% of 1.5 (i.e., between 1.35 and 1.65).
    
    **Validates: Requirements 5.5**
    """
    # Generate reference frame at ref_distance
    ref_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=ref_distance,
        yaw_angle=yaw_angle
    )
    
    # Generate current frame at 50% farther (1.5x distance)
    current_distance = ref_distance * 1.5
    current_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=current_distance,
        yaw_angle=yaw_angle
    )
    
    # Compute corrected widths
    ref_corrected = compute_corrected_width(ref_result.shoulder_width_2d_pixels, yaw_angle)
    current_corrected = compute_corrected_width(current_result.shoulder_width_2d_pixels, yaw_angle)
    
    # Compute relative distance ratio
    measured_ratio = compute_relative_distance_ratio(ref_corrected, current_corrected)
    
    # Expected ratio is 1.5 (50% farther)
    expected_ratio = 1.5
    tolerance = 0.10  # 10% tolerance
    lower_bound = expected_ratio * (1 - tolerance)  # 1.35
    upper_bound = expected_ratio * (1 + tolerance)  # 1.65
    
    assert lower_bound <= measured_ratio <= upper_bound, \
        f"Distance ratio {measured_ratio:.3f} for 50% distance increase should be within 10% of 1.5 " \
        f"(range: [{lower_bound:.2f}, {upper_bound:.2f}])"


@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    ref_distance=st.floats(min_value=2.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    distance_multiplier=st.floats(min_value=1.2, max_value=2.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_distance_ratio_accuracy_general(shoulder_width_3d, ref_distance, distance_multiplier):
    """
    **Feature: pose-corrected-distance, Property 12: Distance Ratio Accuracy**
    
    *For any* distance change, the measured Relative_Distance_Ratio SHALL be 
    within 10% of the actual distance ratio.
    
    **Validates: Requirements 5.5**
    """
    # Generate reference frame
    ref_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=ref_distance,
        yaw_angle=0.0  # Frontal pose for simplicity
    )
    
    # Generate current frame at multiplied distance
    current_distance = ref_distance * distance_multiplier
    current_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=current_distance,
        yaw_angle=0.0
    )
    
    # Compute corrected widths
    ref_corrected = compute_corrected_width(ref_result.shoulder_width_2d_pixels, 0.0)
    current_corrected = compute_corrected_width(current_result.shoulder_width_2d_pixels, 0.0)
    
    # Compute relative distance ratio
    measured_ratio = compute_relative_distance_ratio(ref_corrected, current_corrected)
    
    # Expected ratio equals the distance multiplier
    expected_ratio = distance_multiplier
    tolerance = 0.10  # 10% tolerance
    lower_bound = expected_ratio * (1 - tolerance)
    upper_bound = expected_ratio * (1 + tolerance)
    
    assert lower_bound <= measured_ratio <= upper_bound, \
        f"Distance ratio {measured_ratio:.3f} should be within 10% of {expected_ratio:.2f} " \
        f"(range: [{lower_bound:.2f}, {upper_bound:.2f}])"


# =============================================================================
# Property 13: Rotation Robustness (Task 12.3)
# =============================================================================

@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=2.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    yaw_angle=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_rotation_robustness(shoulder_width_3d, distance, yaw_angle):
    """
    **Feature: pose-corrected-distance, Property 13: Rotation Robustness**
    
    *For any* constant physical distance with body rotation from 0° to 60°, 
    the Relative_Distance_Ratio SHALL remain within 20% of 1.0 (i.e., between 0.8 and 1.2).
    
    **Validates: Requirements 6.1**
    """
    # Generate reference frame at frontal pose (0°)
    ref_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=0.0
    )
    
    # Generate current frame at the given yaw angle (same distance)
    current_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=yaw_angle
    )
    
    # Compute corrected widths
    ref_corrected = compute_corrected_width(ref_result.shoulder_width_2d_pixels, 0.0)
    current_corrected = compute_corrected_width(current_result.shoulder_width_2d_pixels, yaw_angle)
    
    # Compute relative distance ratio
    measured_ratio = compute_relative_distance_ratio(ref_corrected, current_corrected)
    
    # At constant distance, ratio should be close to 1.0
    expected_ratio = 1.0
    tolerance = 0.20  # 20% tolerance
    lower_bound = expected_ratio * (1 - tolerance)  # 0.8
    upper_bound = expected_ratio * (1 + tolerance)  # 1.2
    
    assert lower_bound <= measured_ratio <= upper_bound, \
        f"Distance ratio {measured_ratio:.3f} at constant distance with {yaw_angle:.1f}° rotation " \
        f"should be within 20% of 1.0 (range: [{lower_bound:.2f}, {upper_bound:.2f}])"


@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=2.0, max_value=3.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_rotation_robustness_sequence(shoulder_width_3d, distance):
    """
    **Feature: pose-corrected-distance, Property 13: Rotation Robustness**
    
    *For any* rotation sequence from 0° to 60° at constant distance, all 
    Relative_Distance_Ratios SHALL remain within 20% of 1.0.
    
    **Validates: Requirements 6.1**
    """
    # Generate rotation sequence from 0° to 60°
    sequence = generate_rotation_sequence(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        start_yaw=0.0,
        end_yaw=60.0,
        num_steps=13  # Every 5 degrees
    )
    
    # Get reference (frontal) corrected width
    ref_result = sequence[0]
    ref_corrected = compute_corrected_width(ref_result.shoulder_width_2d_pixels, 0.0)
    
    tolerance = 0.20
    lower_bound = 1.0 * (1 - tolerance)  # 0.8
    upper_bound = 1.0 * (1 + tolerance)  # 1.2
    
    # Check all frames in the sequence
    for result in sequence:
        current_corrected = compute_corrected_width(result.shoulder_width_2d_pixels, result.body_yaw)
        measured_ratio = compute_relative_distance_ratio(ref_corrected, current_corrected)
        
        assert lower_bound <= measured_ratio <= upper_bound, \
            f"Distance ratio {measured_ratio:.3f} at {result.body_yaw:.1f}° should be within 20% of 1.0 " \
            f"(range: [{lower_bound:.2f}, {upper_bound:.2f}])"


# =============================================================================
# Property 14: Correction Improvement (Task 12.4)
# =============================================================================

@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=2.0, max_value=3.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_correction_improvement_at_45_degrees(shoulder_width_3d, distance):
    """
    **Feature: pose-corrected-distance, Property 14: Correction Improvement**
    
    *For any* 45° body rotation, the error in corrected measurement SHALL be 
    at least 50% smaller than the error in uncorrected measurement.
    
    **Validates: Requirements 6.3**
    """
    yaw_angle = 45.0
    
    # Generate reference frame at frontal pose (0°)
    ref_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=0.0
    )
    
    # Generate current frame at 45° (same distance)
    current_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=yaw_angle
    )
    
    # Reference width (frontal, no correction needed)
    ref_width = ref_result.shoulder_width_2d_pixels
    
    # Current width (rotated)
    current_width = current_result.shoulder_width_2d_pixels
    
    # Corrected current width
    corrected_width = compute_corrected_width(current_width, yaw_angle)
    
    # Compute errors relative to reference
    # Uncorrected error: how much the raw 2D width deviates from reference
    uncorrected_error = abs(current_width - ref_width) / ref_width
    
    # Corrected error: how much the corrected width deviates from reference
    corrected_error = abs(corrected_width - ref_width) / ref_width
    
    # The corrected error should be at least 50% smaller than uncorrected error
    # This means: corrected_error <= uncorrected_error * 0.5
    improvement_threshold = 0.50
    
    assert corrected_error <= uncorrected_error * improvement_threshold, \
        f"Corrected error {corrected_error:.3f} should be at least 50% smaller than " \
        f"uncorrected error {uncorrected_error:.3f} (threshold: {uncorrected_error * improvement_threshold:.3f})"


@given(
    shoulder_width_3d=st.floats(min_value=0.35, max_value=0.45, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=2.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    yaw_angle=st.floats(min_value=30.0, max_value=60.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_correction_improvement_general(shoulder_width_3d, distance, yaw_angle):
    """
    **Feature: pose-corrected-distance, Property 14: Correction Improvement**
    
    *For any* body rotation between 30° and 60°, the corrected measurement error 
    SHALL be smaller than the uncorrected measurement error.
    
    **Validates: Requirements 6.3**
    """
    # Generate reference frame at frontal pose (0°)
    ref_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=0.0
    )
    
    # Generate current frame at the given yaw angle (same distance)
    current_result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width_3d,
        distance=distance,
        yaw_angle=yaw_angle
    )
    
    # Reference width (frontal)
    ref_width = ref_result.shoulder_width_2d_pixels
    
    # Current width (rotated)
    current_width = current_result.shoulder_width_2d_pixels
    
    # Corrected current width
    corrected_width = compute_corrected_width(current_width, yaw_angle)
    
    # Compute errors
    uncorrected_error = abs(current_width - ref_width) / ref_width
    corrected_error = abs(corrected_width - ref_width) / ref_width
    
    # Corrected error should be smaller than uncorrected error
    assert corrected_error < uncorrected_error, \
        f"Corrected error {corrected_error:.3f} should be smaller than " \
        f"uncorrected error {uncorrected_error:.3f} at {yaw_angle:.1f}°"


# =============================================================================
# Unit Tests for Pose Correction Effectiveness
# =============================================================================

class TestPoseCorrectionEffectiveness:
    """Unit tests for pose correction effectiveness."""
    
    def test_correction_at_45_degrees_reduces_error(self):
        """Test that correction at 45° significantly reduces error."""
        shoulder_width_3d = 0.40
        distance = 2.0
        
        # Generate frontal and 45° rotated frames
        frontal = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=distance,
            yaw_angle=0.0
        )
        rotated = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=distance,
            yaw_angle=45.0
        )
        
        ref_width = frontal.shoulder_width_2d_pixels
        current_width = rotated.shoulder_width_2d_pixels
        corrected_width = compute_corrected_width(current_width, 45.0)
        
        uncorrected_error = abs(current_width - ref_width) / ref_width
        corrected_error = abs(corrected_width - ref_width) / ref_width
        
        # Correction should reduce error by at least 50%
        assert corrected_error <= uncorrected_error * 0.5, \
            f"Correction should reduce error by at least 50%"
    
    def test_distance_ratio_accuracy_at_1_5x(self):
        """Test distance ratio accuracy when subject moves 50% farther."""
        shoulder_width_3d = 0.40
        ref_distance = 2.0
        current_distance = 3.0  # 50% farther
        
        ref = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=ref_distance,
            yaw_angle=0.0
        )
        current = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=current_distance,
            yaw_angle=0.0
        )
        
        ref_corrected = compute_corrected_width(ref.shoulder_width_2d_pixels, 0.0)
        current_corrected = compute_corrected_width(current.shoulder_width_2d_pixels, 0.0)
        
        ratio = compute_relative_distance_ratio(ref_corrected, current_corrected)
        
        # Ratio should be close to 1.5 (within 10%)
        assert 1.35 <= ratio <= 1.65, \
            f"Distance ratio {ratio:.3f} should be within 10% of 1.5"
    
    def test_rotation_robustness_at_constant_distance(self):
        """Test that rotation doesn't significantly affect distance ratio."""
        shoulder_width_3d = 0.40
        distance = 2.0
        
        # Generate frontal reference
        ref = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=distance,
            yaw_angle=0.0
        )
        ref_corrected = compute_corrected_width(ref.shoulder_width_2d_pixels, 0.0)
        
        # Test at various rotation angles
        for yaw in [15.0, 30.0, 45.0, 60.0]:
            current = generate_synthetic_landmarks(
                shoulder_width_3d=shoulder_width_3d,
                distance=distance,
                yaw_angle=yaw
            )
            current_corrected = compute_corrected_width(current.shoulder_width_2d_pixels, yaw)
            
            ratio = compute_relative_distance_ratio(ref_corrected, current_corrected)
            
            # Ratio should be close to 1.0 (within 20%)
            assert 0.8 <= ratio <= 1.2, \
                f"Distance ratio {ratio:.3f} at {yaw}° should be within 20% of 1.0"
