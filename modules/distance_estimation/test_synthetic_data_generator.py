"""
Tests for the Synthetic Data Generator module.

This module tests the synthetic landmark generation functionality used for
testing the pose correction algorithm.

**Validates: Requirements 8.1, 8.4**
"""

import math
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from .synthetic_data_generator import (
    Landmark3D,
    Landmark2D,
    SyntheticBodyConfig,
    CameraConfig,
    SyntheticLandmarkResult,
    project_3d_to_2d,
    normalize_2d_coordinates,
    rotate_point_around_y,
    generate_synthetic_landmarks,
    generate_rotation_sequence,
    generate_distance_sequence,
    verify_3d_shoulder_width_constant,
    compute_expected_2d_width,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)


# =============================================================================
# Unit Tests for Basic Functions
# =============================================================================

class TestProjection:
    """Tests for 3D to 2D projection."""
    
    def test_project_point_at_center(self):
        """Point at camera center should project to principal point."""
        camera = CameraConfig(focal_length=500, image_width=1280, image_height=720)
        point = (0, 0, 1.0)  # On optical axis, 1m away
        
        u, v = project_3d_to_2d(point, camera)
        
        assert math.isclose(u, camera.principal_x, abs_tol=1e-9)
        assert math.isclose(v, camera.principal_y, abs_tol=1e-9)
    
    def test_project_point_offset_x(self):
        """Point offset in X should project to right of center."""
        camera = CameraConfig(focal_length=500, image_width=1280, image_height=720)
        point = (0.1, 0, 1.0)  # 0.1m to the right, 1m away
        
        u, v = project_3d_to_2d(point, camera)
        
        # u = fx * X / Z + cx = 500 * 0.1 / 1.0 + 640 = 690
        expected_u = 500 * 0.1 / 1.0 + 640
        assert math.isclose(u, expected_u, abs_tol=1e-9)
        assert math.isclose(v, camera.principal_y, abs_tol=1e-9)
    
    def test_project_point_farther_distance(self):
        """Point farther away should project closer to center."""
        camera = CameraConfig(focal_length=500, image_width=1280, image_height=720)
        point_near = (0.1, 0, 1.0)
        point_far = (0.1, 0, 2.0)
        
        u_near, _ = project_3d_to_2d(point_near, camera)
        u_far, _ = project_3d_to_2d(point_far, camera)
        
        # Farther point should be closer to center
        assert abs(u_far - camera.principal_x) < abs(u_near - camera.principal_x)


class TestRotation:
    """Tests for point rotation around Y axis."""
    
    def test_rotate_zero_degrees(self):
        """Zero rotation should not change the point."""
        point = (1.0, 0.0, 0.0)
        rotated = rotate_point_around_y(point, 0.0)
        
        assert math.isclose(rotated[0], point[0], abs_tol=1e-9)
        assert math.isclose(rotated[1], point[1], abs_tol=1e-9)
        assert math.isclose(rotated[2], point[2], abs_tol=1e-9)
    
    def test_rotate_90_degrees(self):
        """90 degree rotation should swap X and Z."""
        point = (1.0, 0.0, 0.0)
        rotated = rotate_point_around_y(point, 90.0)
        
        # After 90 degree rotation: (1, 0, 0) -> (0, 0, -1)
        assert math.isclose(rotated[0], 0.0, abs_tol=1e-9)
        assert math.isclose(rotated[1], 0.0, abs_tol=1e-9)
        assert math.isclose(rotated[2], -1.0, abs_tol=1e-9)
    
    def test_rotate_preserves_distance_from_origin(self):
        """Rotation should preserve distance from origin."""
        point = (1.0, 0.5, 0.3)
        original_dist = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        
        rotated = rotate_point_around_y(point, 45.0)
        rotated_dist = math.sqrt(rotated[0]**2 + rotated[1]**2 + rotated[2]**2)
        
        assert math.isclose(original_dist, rotated_dist, abs_tol=1e-9)


# =============================================================================
# Tests for Synthetic Landmark Generation
# =============================================================================

class TestSyntheticLandmarkGeneration:
    """Tests for the main synthetic landmark generator."""
    
    def test_generate_frontal_pose(self):
        """Test generating landmarks for frontal pose (yaw=0)."""
        result = generate_synthetic_landmarks(
            shoulder_width_3d=0.40,
            distance=2.0,
            yaw_angle=0.0,
            focal_length=500.0
        )
        
        # 3D shoulder width should match input
        assert math.isclose(result.shoulder_width_3d, 0.40, abs_tol=1e-6)
        
        # Body yaw should be 0
        assert result.body_yaw == 0.0
        
        # Distance should match input
        assert result.distance == 2.0
        
        # Should have landmarks
        assert len(result.landmarks_3d) == 33
        assert len(result.landmarks_2d) == 33
    
    def test_generate_side_pose(self):
        """Test generating landmarks for side pose (yaw=90)."""
        result = generate_synthetic_landmarks(
            shoulder_width_3d=0.40,
            distance=2.0,
            yaw_angle=90.0,
            focal_length=500.0
        )
        
        # 3D shoulder width should still be 0.40 (constant)
        assert math.isclose(result.shoulder_width_3d, 0.40, abs_tol=1e-6)
        
        # 2D shoulder width should be near zero (side view)
        assert result.shoulder_width_2d_pixels < 10  # Very small
    
    def test_3d_width_constant_across_yaw(self):
        """3D shoulder width should remain constant regardless of yaw angle."""
        shoulder_width = 0.40
        
        for yaw in [0, 15, 30, 45, 60, 75, 90]:
            result = generate_synthetic_landmarks(
                shoulder_width_3d=shoulder_width,
                distance=2.0,
                yaw_angle=yaw,
                focal_length=500.0
            )
            
            assert math.isclose(result.shoulder_width_3d, shoulder_width, abs_tol=1e-6), \
                f"3D width should be constant at yaw={yaw}"
    
    def test_2d_width_decreases_with_yaw(self):
        """2D shoulder width should decrease as yaw increases (toward side view)."""
        results = []
        for yaw in [0, 30, 60, 90]:
            result = generate_synthetic_landmarks(
                shoulder_width_3d=0.40,
                distance=2.0,
                yaw_angle=yaw,
                focal_length=500.0
            )
            results.append(result)
        
        # Each subsequent result should have smaller 2D width
        for i in range(len(results) - 1):
            assert results[i].shoulder_width_2d_pixels > results[i+1].shoulder_width_2d_pixels, \
                f"2D width should decrease: yaw={results[i].body_yaw} vs yaw={results[i+1].body_yaw}"
    
    def test_2d_width_follows_cosine_law(self):
        """2D width should follow: 2D = 3D × |cos(yaw)| × f / distance."""
        shoulder_width = 0.40
        distance = 2.0
        focal_length = 500.0
        
        for yaw in [0, 30, 45, 60]:
            result = generate_synthetic_landmarks(
                shoulder_width_3d=shoulder_width,
                distance=distance,
                yaw_angle=yaw,
                focal_length=focal_length
            )
            
            expected_2d = compute_expected_2d_width(
                shoulder_width, distance, yaw, focal_length
            )
            
            # Allow some tolerance due to projection geometry
            assert math.isclose(result.shoulder_width_2d_pixels, expected_2d, rel_tol=0.01), \
                f"2D width mismatch at yaw={yaw}: got {result.shoulder_width_2d_pixels}, expected {expected_2d}"


# =============================================================================
# Tests for Rotation Sequence Generation
# =============================================================================

class TestRotationSequence:
    """Tests for rotation sequence generation."""
    
    def test_generate_rotation_sequence_length(self):
        """Sequence should have correct number of frames."""
        sequence = generate_rotation_sequence(
            shoulder_width_3d=0.40,
            distance=2.0,
            start_yaw=0.0,
            end_yaw=90.0,
            num_steps=10
        )
        
        assert len(sequence) == 10
    
    def test_rotation_sequence_yaw_values(self):
        """Sequence should have correct yaw values."""
        sequence = generate_rotation_sequence(
            shoulder_width_3d=0.40,
            distance=2.0,
            start_yaw=0.0,
            end_yaw=90.0,
            num_steps=10
        )
        
        # First frame should be at start_yaw
        assert math.isclose(sequence[0].body_yaw, 0.0, abs_tol=1e-9)
        
        # Last frame should be at end_yaw
        assert math.isclose(sequence[-1].body_yaw, 90.0, abs_tol=1e-9)
    
    def test_rotation_sequence_3d_width_constant(self):
        """3D shoulder width should remain constant throughout rotation."""
        sequence = generate_rotation_sequence(
            shoulder_width_3d=0.40,
            distance=2.0,
            start_yaw=0.0,
            end_yaw=90.0,
            num_steps=10
        )
        
        is_constant, max_deviation = verify_3d_shoulder_width_constant(sequence)
        
        assert is_constant, f"3D width should be constant, max deviation: {max_deviation}"
    
    def test_rotation_sequence_2d_width_decreases(self):
        """2D shoulder width should decrease during rotation from frontal to side."""
        sequence = generate_rotation_sequence(
            shoulder_width_3d=0.40,
            distance=2.0,
            start_yaw=0.0,
            end_yaw=90.0,
            num_steps=10
        )
        
        # 2D width should generally decrease (with some tolerance for small variations)
        first_2d = sequence[0].shoulder_width_2d_pixels
        last_2d = sequence[-1].shoulder_width_2d_pixels
        
        assert first_2d > last_2d, \
            f"2D width should decrease: first={first_2d}, last={last_2d}"


# =============================================================================
# Tests for Distance Sequence Generation
# =============================================================================

class TestDistanceSequence:
    """Tests for distance sequence generation."""
    
    def test_generate_distance_sequence_length(self):
        """Sequence should have correct number of frames."""
        sequence = generate_distance_sequence(
            shoulder_width_3d=0.40,
            start_distance=1.0,
            end_distance=3.0,
            num_steps=10
        )
        
        assert len(sequence) == 10
    
    def test_distance_sequence_values(self):
        """Sequence should have correct distance values."""
        sequence = generate_distance_sequence(
            shoulder_width_3d=0.40,
            start_distance=1.0,
            end_distance=3.0,
            num_steps=3
        )
        
        assert math.isclose(sequence[0].distance, 1.0, abs_tol=1e-9)
        assert math.isclose(sequence[1].distance, 2.0, abs_tol=1e-9)
        assert math.isclose(sequence[2].distance, 3.0, abs_tol=1e-9)
    
    def test_distance_sequence_2d_width_decreases(self):
        """2D width should decrease as distance increases."""
        sequence = generate_distance_sequence(
            shoulder_width_3d=0.40,
            start_distance=1.0,
            end_distance=3.0,
            yaw_angle=0.0,
            num_steps=10
        )
        
        # Each subsequent frame should have smaller 2D width
        for i in range(len(sequence) - 1):
            assert sequence[i].shoulder_width_2d_pixels > sequence[i+1].shoulder_width_2d_pixels


# =============================================================================
# Property-Based Tests
# =============================================================================

@given(
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    yaw_angle=st.floats(min_value=-89.0, max_value=89.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_3d_width_invariant_to_yaw(shoulder_width, distance, yaw_angle):
    """
    **Feature: pose-corrected-distance, Property: 3D Width Invariance**
    
    *For any* shoulder width, distance, and yaw angle, the 3D shoulder width
    in the generated landmarks SHALL equal the input shoulder width.
    
    **Validates: Requirements 8.1, 8.4**
    """
    result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width,
        distance=distance,
        yaw_angle=yaw_angle
    )
    
    assert math.isclose(result.shoulder_width_3d, shoulder_width, abs_tol=1e-6), \
        f"3D width should be {shoulder_width}, got {result.shoulder_width_3d}"


@given(
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    yaw_angle=st.floats(min_value=-60.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    focal_length=st.floats(min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_2d_width_follows_projection_model(shoulder_width, distance, yaw_angle, focal_length):
    """
    **Feature: pose-corrected-distance, Property: 2D Width Projection**
    
    *For any* shoulder width, distance, yaw angle, and focal length, the 2D shoulder
    width SHALL approximately follow: 2D = 3D × |cos(yaw)| × f / distance.
    
    Note: The simplified formula is an approximation. The actual projection accounts
    for the fact that when the body rotates, the shoulders are at different depths.
    At close distances and large angles, this difference becomes more pronounced.
    We use a 5% tolerance to account for this geometric difference.
    
    **Validates: Requirements 8.1**
    """
    result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width,
        distance=distance,
        yaw_angle=yaw_angle,
        focal_length=focal_length
    )
    
    expected_2d = compute_expected_2d_width(
        shoulder_width, distance, yaw_angle, focal_length
    )
    
    # Allow 5% tolerance for projection geometry differences
    # The simplified formula doesn't account for depth differences between shoulders
    # when the body is rotated, which becomes more significant at close distances
    assert math.isclose(result.shoulder_width_2d_pixels, expected_2d, rel_tol=0.05), \
        f"2D width mismatch: got {result.shoulder_width_2d_pixels}, expected {expected_2d}"


@given(
    shoulder_width=st.floats(min_value=0.2, max_value=0.6, allow_nan=False, allow_infinity=False),
    distance=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    num_steps=st.integers(min_value=2, max_value=20)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_rotation_sequence_3d_constant(shoulder_width, distance, num_steps):
    """
    **Feature: pose-corrected-distance, Property: Rotation Sequence 3D Constancy**
    
    *For any* rotation sequence from 0° to 90°, the 3D shoulder width SHALL
    remain constant throughout the sequence.
    
    **Validates: Requirements 8.4**
    """
    sequence = generate_rotation_sequence(
        shoulder_width_3d=shoulder_width,
        distance=distance,
        start_yaw=0.0,
        end_yaw=90.0,
        num_steps=num_steps
    )
    
    is_constant, max_deviation = verify_3d_shoulder_width_constant(sequence)
    
    assert is_constant, \
        f"3D width should be constant, max deviation: {max_deviation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
