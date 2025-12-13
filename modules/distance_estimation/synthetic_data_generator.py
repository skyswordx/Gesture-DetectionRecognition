"""
Synthetic Data Generator for Pose-Corrected Distance Testing

This module generates synthetic landmark data with known 3D positions and 
corresponding 2D projections for testing the pose correction algorithm.

The generator supports:
- Configurable body dimensions (shoulder width, hip width)
- Configurable distance from camera
- Configurable body yaw angle (rotation)
- Pinhole camera model for 2D projection

**Validates: Requirements 8.1, 8.4**
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


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


@dataclass
class SyntheticBodyConfig:
    """
    Configuration for synthetic body generation.
    
    Attributes:
        shoulder_width_3d: Real shoulder width in meters (default: 0.40m)
        hip_width_3d: Real hip width in meters (default: 0.30m)
        torso_height: Distance from shoulders to hips in meters (default: 0.50m)
        visibility: Visibility value for all landmarks (default: 1.0)
    """
    shoulder_width_3d: float = 0.40  # meters
    hip_width_3d: float = 0.30       # meters
    torso_height: float = 0.50       # meters
    visibility: float = 1.0


@dataclass
class CameraConfig:
    """
    Camera configuration for 2D projection.
    
    Attributes:
        focal_length: Focal length in pixels (default: 500)
        image_width: Image width in pixels (default: 1280)
        image_height: Image height in pixels (default: 720)
        principal_x: Principal point x (default: image_width/2)
        principal_y: Principal point y (default: image_height/2)
    """
    focal_length: float = 500.0
    image_width: int = 1280
    image_height: int = 720
    principal_x: Optional[float] = None
    principal_y: Optional[float] = None
    
    def __post_init__(self):
        if self.principal_x is None:
            self.principal_x = self.image_width / 2
        if self.principal_y is None:
            self.principal_y = self.image_height / 2


@dataclass
class SyntheticLandmarkResult:
    """
    Result of synthetic landmark generation.
    
    Attributes:
        landmarks_3d: List of 3D landmarks (world coordinates)
        landmarks_2d: List of 2D landmarks (normalized image coordinates)
        shoulder_width_3d: Actual 3D shoulder width (should be constant)
        shoulder_width_2d_pixels: 2D shoulder width in pixels
        hip_width_3d: Actual 3D hip width (should be constant)
        hip_width_2d_pixels: 2D hip width in pixels
        body_yaw: Body yaw angle in degrees
        distance: Distance from camera in meters
    """
    landmarks_3d: List[Landmark3D]
    landmarks_2d: List[Landmark2D]
    shoulder_width_3d: float
    shoulder_width_2d_pixels: float
    hip_width_3d: float
    hip_width_2d_pixels: float
    body_yaw: float
    distance: float


# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NUM_LANDMARKS = 33  # MediaPipe pose has 33 landmarks


def project_3d_to_2d(
    point_3d: Tuple[float, float, float],
    camera: CameraConfig
) -> Tuple[float, float]:
    """
    Project a 3D point to 2D image coordinates using pinhole camera model.
    
    The pinhole camera model:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
    
    Args:
        point_3d: (x, y, z) coordinates in camera frame (z is depth)
        camera: Camera configuration
        
    Returns:
        (u, v) pixel coordinates
        
    **Validates: Requirements 8.1**
    """
    x, y, z = point_3d
    
    # Avoid division by zero
    if z <= 0:
        z = 0.001
    
    # Project using pinhole model
    u = camera.focal_length * x / z + camera.principal_x
    v = camera.focal_length * y / z + camera.principal_y
    
    return u, v


def normalize_2d_coordinates(
    pixel_coords: Tuple[float, float],
    camera: CameraConfig
) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to [0, 1] range.
    
    Args:
        pixel_coords: (u, v) pixel coordinates
        camera: Camera configuration
        
    Returns:
        (x, y) normalized coordinates in [0, 1] range
    """
    u, v = pixel_coords
    x = u / camera.image_width
    y = v / camera.image_height
    return x, y


def rotate_point_around_y(
    point: Tuple[float, float, float],
    angle_degrees: float,
    center: Tuple[float, float, float] = (0, 0, 0)
) -> Tuple[float, float, float]:
    """
    Rotate a 3D point around the Y axis (vertical axis).
    
    This simulates body rotation (yaw).
    
    Args:
        point: (x, y, z) coordinates
        angle_degrees: Rotation angle in degrees (positive = clockwise when viewed from above)
        center: Center of rotation
        
    Returns:
        Rotated (x, y, z) coordinates
    """
    x, y, z = point
    cx, cy, cz = center
    
    # Translate to origin
    x -= cx
    z -= cz
    
    # Rotate around Y axis
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    new_x = x * cos_a + z * sin_a
    new_z = -x * sin_a + z * cos_a
    
    # Translate back
    return (new_x + cx, y, new_z + cz)


def generate_synthetic_landmarks(
    shoulder_width_3d: float,
    distance: float,
    yaw_angle: float,
    focal_length: float = 500.0,
    hip_width_3d: float = 0.30,
    torso_height: float = 0.50,
    image_width: int = 1280,
    image_height: int = 720,
    visibility: float = 1.0
) -> SyntheticLandmarkResult:
    """
    Generate synthetic landmark data with known 3D positions and corresponding 2D projections.
    
    This function creates a simplified body model with shoulders and hips, then:
    1. Positions the body at the specified distance from the camera
    2. Rotates the body by the specified yaw angle
    3. Projects the 3D points to 2D using a pinhole camera model
    
    The 3D shoulder width remains constant regardless of yaw angle.
    The 2D shoulder width varies as: 2D = 3D × |cos(yaw)| × f / distance
    
    Args:
        shoulder_width_3d: Real shoulder width in meters
        distance: Distance from camera in meters (Z coordinate)
        yaw_angle: Body yaw angle in degrees (0 = frontal, 90 = side view)
        focal_length: Camera focal length in pixels
        hip_width_3d: Real hip width in meters
        torso_height: Distance from shoulders to hips in meters
        image_width: Image width in pixels
        image_height: Image height in pixels
        visibility: Visibility value for all landmarks
        
    Returns:
        SyntheticLandmarkResult containing 3D and 2D landmarks with measurements
        
    **Validates: Requirements 8.1**
    """
    camera = CameraConfig(
        focal_length=focal_length,
        image_width=image_width,
        image_height=image_height
    )
    
    # Create body model centered at origin, facing camera (negative Z direction)
    # Body center is at (0, 0, distance)
    body_center = (0, 0, distance)
    
    # Shoulder positions (before rotation)
    # Left shoulder is at negative X, right shoulder at positive X
    half_shoulder = shoulder_width_3d / 2
    left_shoulder_local = (-half_shoulder, 0, 0)
    right_shoulder_local = (half_shoulder, 0, 0)
    
    # Hip positions (below shoulders)
    half_hip = hip_width_3d / 2
    left_hip_local = (-half_hip, torso_height, 0)  # Y increases downward
    right_hip_local = (half_hip, torso_height, 0)
    
    # Apply rotation around Y axis (body center)
    left_shoulder_rotated = rotate_point_around_y(left_shoulder_local, yaw_angle)
    right_shoulder_rotated = rotate_point_around_y(right_shoulder_local, yaw_angle)
    left_hip_rotated = rotate_point_around_y(left_hip_local, yaw_angle)
    right_hip_rotated = rotate_point_around_y(right_hip_local, yaw_angle)
    
    # Translate to world coordinates (add body center position)
    def translate(local, center):
        return (local[0] + center[0], local[1] + center[1], local[2] + center[2])
    
    left_shoulder_world = translate(left_shoulder_rotated, body_center)
    right_shoulder_world = translate(right_shoulder_rotated, body_center)
    left_hip_world = translate(left_hip_rotated, body_center)
    right_hip_world = translate(right_hip_rotated, body_center)
    
    # Create 3D landmarks (MediaPipe world coordinates are relative to hip center)
    # For simplicity, we use the rotated local coordinates as world landmarks
    landmarks_3d = [Landmark3D(0, 0, 0, visibility) for _ in range(NUM_LANDMARKS)]
    
    landmarks_3d[LEFT_SHOULDER] = Landmark3D(
        x=left_shoulder_rotated[0],
        y=left_shoulder_rotated[1],
        z=left_shoulder_rotated[2],
        visibility=visibility
    )
    landmarks_3d[RIGHT_SHOULDER] = Landmark3D(
        x=right_shoulder_rotated[0],
        y=right_shoulder_rotated[1],
        z=right_shoulder_rotated[2],
        visibility=visibility
    )
    landmarks_3d[LEFT_HIP] = Landmark3D(
        x=left_hip_rotated[0],
        y=left_hip_rotated[1],
        z=left_hip_rotated[2],
        visibility=visibility
    )
    landmarks_3d[RIGHT_HIP] = Landmark3D(
        x=right_hip_rotated[0],
        y=right_hip_rotated[1],
        z=right_hip_rotated[2],
        visibility=visibility
    )
    
    # Project to 2D
    left_shoulder_2d = project_3d_to_2d(left_shoulder_world, camera)
    right_shoulder_2d = project_3d_to_2d(right_shoulder_world, camera)
    left_hip_2d = project_3d_to_2d(left_hip_world, camera)
    right_hip_2d = project_3d_to_2d(right_hip_world, camera)
    
    # Normalize to [0, 1]
    left_shoulder_norm = normalize_2d_coordinates(left_shoulder_2d, camera)
    right_shoulder_norm = normalize_2d_coordinates(right_shoulder_2d, camera)
    left_hip_norm = normalize_2d_coordinates(left_hip_2d, camera)
    right_hip_norm = normalize_2d_coordinates(right_hip_2d, camera)
    
    # Create 2D landmarks
    landmarks_2d = [Landmark2D(0.5, 0.5, 0, visibility) for _ in range(NUM_LANDMARKS)]
    
    # Calculate relative depth (z) for 2D landmarks
    # Use the z-offset from body center as relative depth
    landmarks_2d[LEFT_SHOULDER] = Landmark2D(
        x=left_shoulder_norm[0],
        y=left_shoulder_norm[1],
        z=left_shoulder_rotated[2] / distance,  # Relative depth
        visibility=visibility
    )
    landmarks_2d[RIGHT_SHOULDER] = Landmark2D(
        x=right_shoulder_norm[0],
        y=right_shoulder_norm[1],
        z=right_shoulder_rotated[2] / distance,
        visibility=visibility
    )
    landmarks_2d[LEFT_HIP] = Landmark2D(
        x=left_hip_norm[0],
        y=left_hip_norm[1],
        z=left_hip_rotated[2] / distance,
        visibility=visibility
    )
    landmarks_2d[RIGHT_HIP] = Landmark2D(
        x=right_hip_norm[0],
        y=right_hip_norm[1],
        z=right_hip_rotated[2] / distance,
        visibility=visibility
    )
    
    # Calculate actual measurements
    # 3D shoulder width (should be constant regardless of yaw)
    actual_shoulder_3d = math.sqrt(
        (right_shoulder_rotated[0] - left_shoulder_rotated[0]) ** 2 +
        (right_shoulder_rotated[1] - left_shoulder_rotated[1]) ** 2 +
        (right_shoulder_rotated[2] - left_shoulder_rotated[2]) ** 2
    )
    
    # 2D shoulder width in pixels
    shoulder_width_2d_pixels = abs(right_shoulder_2d[0] - left_shoulder_2d[0])
    
    # 3D hip width
    actual_hip_3d = math.sqrt(
        (right_hip_rotated[0] - left_hip_rotated[0]) ** 2 +
        (right_hip_rotated[1] - left_hip_rotated[1]) ** 2 +
        (right_hip_rotated[2] - left_hip_rotated[2]) ** 2
    )
    
    # 2D hip width in pixels
    hip_width_2d_pixels = abs(right_hip_2d[0] - left_hip_2d[0])
    
    return SyntheticLandmarkResult(
        landmarks_3d=landmarks_3d,
        landmarks_2d=landmarks_2d,
        shoulder_width_3d=actual_shoulder_3d,
        shoulder_width_2d_pixels=shoulder_width_2d_pixels,
        hip_width_3d=actual_hip_3d,
        hip_width_2d_pixels=hip_width_2d_pixels,
        body_yaw=yaw_angle,
        distance=distance
    )



def generate_rotation_sequence(
    shoulder_width_3d: float = 0.40,
    distance: float = 2.0,
    start_yaw: float = 0.0,
    end_yaw: float = 90.0,
    num_steps: int = 10,
    focal_length: float = 500.0,
    hip_width_3d: float = 0.30,
    torso_height: float = 0.50,
    image_width: int = 1280,
    image_height: int = 720,
    visibility: float = 1.0
) -> List[SyntheticLandmarkResult]:
    """
    Generate a sequence of synthetic landmarks simulating body rotation.
    
    This function generates landmark data for a body rotating from start_yaw to end_yaw
    while maintaining constant 3D shoulder width. This is useful for testing that:
    1. The 3D shoulder width remains constant during rotation
    2. The pose correction algorithm properly compensates for rotation
    
    Args:
        shoulder_width_3d: Real shoulder width in meters
        distance: Distance from camera in meters
        start_yaw: Starting yaw angle in degrees (default: 0 = frontal)
        end_yaw: Ending yaw angle in degrees (default: 90 = side view)
        num_steps: Number of frames in the sequence
        focal_length: Camera focal length in pixels
        hip_width_3d: Real hip width in meters
        torso_height: Distance from shoulders to hips in meters
        image_width: Image width in pixels
        image_height: Image height in pixels
        visibility: Visibility value for all landmarks
        
    Returns:
        List of SyntheticLandmarkResult, one for each rotation step
        
    **Validates: Requirements 8.4**
    """
    results = []
    
    if num_steps <= 1:
        # Single frame at start_yaw
        result = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=distance,
            yaw_angle=start_yaw,
            focal_length=focal_length,
            hip_width_3d=hip_width_3d,
            torso_height=torso_height,
            image_width=image_width,
            image_height=image_height,
            visibility=visibility
        )
        results.append(result)
    else:
        # Generate sequence from start_yaw to end_yaw
        yaw_step = (end_yaw - start_yaw) / (num_steps - 1)
        
        for i in range(num_steps):
            yaw_angle = start_yaw + i * yaw_step
            
            result = generate_synthetic_landmarks(
                shoulder_width_3d=shoulder_width_3d,
                distance=distance,
                yaw_angle=yaw_angle,
                focal_length=focal_length,
                hip_width_3d=hip_width_3d,
                torso_height=torso_height,
                image_width=image_width,
                image_height=image_height,
                visibility=visibility
            )
            results.append(result)
    
    return results


def generate_distance_sequence(
    shoulder_width_3d: float = 0.40,
    start_distance: float = 1.0,
    end_distance: float = 3.0,
    yaw_angle: float = 0.0,
    num_steps: int = 10,
    focal_length: float = 500.0,
    hip_width_3d: float = 0.30,
    torso_height: float = 0.50,
    image_width: int = 1280,
    image_height: int = 720,
    visibility: float = 1.0
) -> List[SyntheticLandmarkResult]:
    """
    Generate a sequence of synthetic landmarks simulating distance change.
    
    This function generates landmark data for a body moving from start_distance to 
    end_distance while maintaining constant yaw angle. This is useful for testing
    the relative distance calculation.
    
    Args:
        shoulder_width_3d: Real shoulder width in meters
        start_distance: Starting distance from camera in meters
        end_distance: Ending distance from camera in meters
        yaw_angle: Body yaw angle in degrees (constant throughout)
        num_steps: Number of frames in the sequence
        focal_length: Camera focal length in pixels
        hip_width_3d: Real hip width in meters
        torso_height: Distance from shoulders to hips in meters
        image_width: Image width in pixels
        image_height: Image height in pixels
        visibility: Visibility value for all landmarks
        
    Returns:
        List of SyntheticLandmarkResult, one for each distance step
    """
    results = []
    
    if num_steps <= 1:
        result = generate_synthetic_landmarks(
            shoulder_width_3d=shoulder_width_3d,
            distance=start_distance,
            yaw_angle=yaw_angle,
            focal_length=focal_length,
            hip_width_3d=hip_width_3d,
            torso_height=torso_height,
            image_width=image_width,
            image_height=image_height,
            visibility=visibility
        )
        results.append(result)
    else:
        distance_step = (end_distance - start_distance) / (num_steps - 1)
        
        for i in range(num_steps):
            distance = start_distance + i * distance_step
            
            result = generate_synthetic_landmarks(
                shoulder_width_3d=shoulder_width_3d,
                distance=distance,
                yaw_angle=yaw_angle,
                focal_length=focal_length,
                hip_width_3d=hip_width_3d,
                torso_height=torso_height,
                image_width=image_width,
                image_height=image_height,
                visibility=visibility
            )
            results.append(result)
    
    return results


def verify_3d_shoulder_width_constant(
    sequence: List[SyntheticLandmarkResult],
    tolerance: float = 1e-9
) -> Tuple[bool, float]:
    """
    Verify that 3D shoulder width remains constant throughout a sequence.
    
    Args:
        sequence: List of SyntheticLandmarkResult from rotation or distance sequence
        tolerance: Maximum allowed deviation from the first frame's shoulder width
        
    Returns:
        (is_constant, max_deviation) - Whether width is constant and the max deviation found
        
    **Validates: Requirements 8.4**
    """
    if not sequence:
        return True, 0.0
    
    reference_width = sequence[0].shoulder_width_3d
    max_deviation = 0.0
    
    for result in sequence:
        deviation = abs(result.shoulder_width_3d - reference_width)
        max_deviation = max(max_deviation, deviation)
    
    is_constant = max_deviation <= tolerance
    return is_constant, max_deviation


def compute_expected_2d_width(
    width_3d: float,
    distance: float,
    yaw_angle: float,
    focal_length: float
) -> float:
    """
    Compute the expected 2D width based on the pinhole camera model.
    
    Formula: 2D_Width = 3D_Width × |cos(yaw)| × focal_length / distance
    
    Args:
        width_3d: 3D width in meters
        distance: Distance from camera in meters
        yaw_angle: Body yaw angle in degrees
        focal_length: Camera focal length in pixels
        
    Returns:
        Expected 2D width in pixels
    """
    yaw_rad = math.radians(yaw_angle)
    projected_width = width_3d * abs(math.cos(yaw_rad))
    return projected_width * focal_length / distance
