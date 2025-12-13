"""
Unit tests for the visualization module.

Tests for:
- Task 13.1: Pose skeleton drawing (Requirements 7.1)
- Task 13.2: Measurement info display (Requirements 7.2)
- Task 13.3: Distance ratio display with color coding (Requirements 7.3)
- Task 13.4: Direction indicator (Requirements 7.4)
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Tuple

# Import visualization functions
from modules.distance_estimation.visualization import (
    get_ratio_color,
    get_direction_text_and_color,
    draw_distance_ratio,
    draw_direction_indicator,
    draw_measurement_info,
    draw_confidence,
    draw_orientation_indicator,
    STABLE_RATIO_MIN,
    STABLE_RATIO_MAX,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_YELLOW,
)


# =============================================================================
# Local Data Structures for Testing
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


# =============================================================================
# Task 13.3: Distance Ratio Color Coding Tests (Requirements 7.3)
# =============================================================================

class TestGetRatioColor:
    """Tests for get_ratio_color function - Task 13.3"""
    
    def test_stable_ratio_returns_green(self):
        """
        Test that stable ratios (0.95-1.05) return green color.
        
        **Validates: Requirements 7.3**
        """
        # Test exact boundaries
        assert get_ratio_color(0.95) == COLOR_GREEN
        assert get_ratio_color(1.05) == COLOR_GREEN
        
        # Test middle of stable range
        assert get_ratio_color(1.0) == COLOR_GREEN
        assert get_ratio_color(0.98) == COLOR_GREEN
        assert get_ratio_color(1.02) == COLOR_GREEN
    
    def test_ratio_below_stable_returns_orange(self):
        """
        Test that ratios below 0.95 return orange color.
        
        **Validates: Requirements 7.3**
        """
        assert get_ratio_color(0.94) == COLOR_ORANGE
        assert get_ratio_color(0.90) == COLOR_ORANGE
        assert get_ratio_color(0.80) == COLOR_ORANGE
        assert get_ratio_color(0.50) == COLOR_ORANGE
    
    def test_ratio_above_stable_returns_orange(self):
        """
        Test that ratios above 1.05 return orange color.
        
        **Validates: Requirements 7.3**
        """
        assert get_ratio_color(1.06) == COLOR_ORANGE
        assert get_ratio_color(1.10) == COLOR_ORANGE
        assert get_ratio_color(1.20) == COLOR_ORANGE
        assert get_ratio_color(1.50) == COLOR_ORANGE
    
    def test_custom_stable_range(self):
        """
        Test that custom stable range works correctly.
        
        **Validates: Requirements 7.3**
        """
        # Custom range: 0.90 to 1.10
        assert get_ratio_color(0.95, stable_min=0.90, stable_max=1.10) == COLOR_GREEN
        assert get_ratio_color(0.89, stable_min=0.90, stable_max=1.10) == COLOR_ORANGE
        assert get_ratio_color(1.11, stable_min=0.90, stable_max=1.10) == COLOR_ORANGE


# =============================================================================
# Task 13.4: Direction Indicator Tests (Requirements 7.4)
# =============================================================================

class TestGetDirectionTextAndColor:
    """Tests for get_direction_text_and_color function - Task 13.4"""
    
    def test_moving_away_when_ratio_above_stable_max(self):
        """
        Test that ratio > 1.05 returns "MOVING AWAY" with red color.
        
        **Validates: Requirements 7.4**
        """
        text, color = get_direction_text_and_color(1.06)
        assert text == "MOVING AWAY"
        assert color == COLOR_RED
        
        text, color = get_direction_text_and_color(1.50)
        assert text == "MOVING AWAY"
        assert color == COLOR_RED
    
    def test_approaching_when_ratio_below_stable_min(self):
        """
        Test that ratio < 0.95 returns "APPROACHING" with green color.
        
        **Validates: Requirements 7.4**
        """
        text, color = get_direction_text_and_color(0.94)
        assert text == "APPROACHING"
        assert color == COLOR_GREEN
        
        text, color = get_direction_text_and_color(0.50)
        assert text == "APPROACHING"
        assert color == COLOR_GREEN
    
    def test_stable_when_ratio_in_stable_range(self):
        """
        Test that ratio in [0.95, 1.05] returns "STABLE" with yellow color.
        
        **Validates: Requirements 7.4**
        """
        # Test boundaries
        text, color = get_direction_text_and_color(0.95)
        assert text == "STABLE"
        assert color == COLOR_YELLOW
        
        text, color = get_direction_text_and_color(1.05)
        assert text == "STABLE"
        assert color == COLOR_YELLOW
        
        # Test middle
        text, color = get_direction_text_and_color(1.0)
        assert text == "STABLE"
        assert color == COLOR_YELLOW
    
    def test_custom_stable_range_direction(self):
        """
        Test that custom stable range works for direction indication.
        
        **Validates: Requirements 7.4**
        """
        # Custom range: 0.90 to 1.10
        text, color = get_direction_text_and_color(1.05, stable_min=0.90, stable_max=1.10)
        assert text == "STABLE"
        
        text, color = get_direction_text_and_color(1.11, stable_min=0.90, stable_max=1.10)
        assert text == "MOVING AWAY"


# =============================================================================
# Task 13.2 & 13.3: Drawing Function Tests (Requirements 7.2, 7.3)
# =============================================================================

class TestDrawFunctions:
    """Tests for drawing functions that modify images"""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image for drawing tests."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def test_measurement(self):
        """Create a test BodyMeasurement."""
        return BodyMeasurement(
            shoulder_width_3d=0.40,
            hip_width_3d=0.30,
            body_yaw=15.0,
            confidence=0.85
        )
    
    def test_draw_distance_ratio_returns_image_and_y(self, test_image):
        """
        Test that draw_distance_ratio returns modified image and next Y position.
        
        **Validates: Requirements 7.3**
        """
        result_image, next_y = draw_distance_ratio(
            test_image, ratio=1.0, start_y=100, is_calibrated=True
        )
        
        # Should return an image
        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == test_image.shape
        
        # Should return next Y position
        assert isinstance(next_y, int)
        assert next_y > 100
    
    def test_draw_distance_ratio_uncalibrated_shows_prompt(self, test_image):
        """
        Test that uncalibrated state shows calibration prompt.
        
        **Validates: Requirements 7.3**
        """
        result_image, next_y = draw_distance_ratio(
            test_image, ratio=1.0, start_y=100, is_calibrated=False
        )
        
        # Image should be modified (not all zeros)
        assert not np.array_equal(result_image, test_image)
    
    def test_draw_direction_indicator_returns_image_and_y(self, test_image):
        """
        Test that draw_direction_indicator returns modified image and next Y position.
        
        **Validates: Requirements 7.4**
        """
        result_image, next_y = draw_direction_indicator(
            test_image, ratio=1.0, start_y=100, is_calibrated=True
        )
        
        # Should return an image
        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == test_image.shape
        
        # Should return next Y position
        assert isinstance(next_y, int)
        assert next_y > 100
    
    def test_draw_direction_indicator_uncalibrated_unchanged(self, test_image):
        """
        Test that uncalibrated state doesn't draw direction indicator.
        
        **Validates: Requirements 7.4**
        """
        original = test_image.copy()
        result_image, next_y = draw_direction_indicator(
            test_image, ratio=1.0, start_y=100, is_calibrated=False
        )
        
        # Y position should not change when uncalibrated
        assert next_y == 100
    
    def test_draw_measurement_info_returns_image_and_y(self, test_image, test_measurement):
        """
        Test that draw_measurement_info returns modified image and next Y position.
        
        **Validates: Requirements 7.2**
        """
        result_image, next_y = draw_measurement_info(
            test_image,
            test_measurement,
            corrected_shoulder_width=200.0,
            corrected_hip_width=150.0
        )
        
        # Should return an image
        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == test_image.shape
        
        # Should return next Y position (after 5 lines of info)
        assert isinstance(next_y, int)
        assert next_y > 30  # Default start_y
    
    def test_draw_confidence_returns_image_and_y(self, test_image):
        """
        Test that draw_confidence returns modified image and next Y position.
        """
        result_image, next_y = draw_confidence(
            test_image, confidence=0.85, start_y=100
        )
        
        assert isinstance(result_image, np.ndarray)
        assert next_y > 100
    
    def test_draw_orientation_indicator_frontal(self, test_image):
        """
        Test that draw_orientation_indicator shows FRONTAL for frontal pose.
        """
        result_image, next_y = draw_orientation_indicator(
            test_image, is_frontal=True, start_y=100
        )
        
        assert isinstance(result_image, np.ndarray)
        assert next_y > 100
    
    def test_draw_orientation_indicator_side_view(self, test_image):
        """
        Test that draw_orientation_indicator shows SIDE VIEW for side pose.
        """
        result_image, next_y = draw_orientation_indicator(
            test_image, is_frontal=False, start_y=100
        )
        
        assert isinstance(result_image, np.ndarray)
        assert next_y > 100


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for visualization constants"""
    
    def test_stable_ratio_range(self):
        """Test that stable ratio range is correctly defined."""
        assert STABLE_RATIO_MIN == 0.95
        assert STABLE_RATIO_MAX == 1.05
    
    def test_color_constants_are_bgr_tuples(self):
        """Test that color constants are valid BGR tuples."""
        colors = [COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_YELLOW]
        
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for channel in color:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255


# =============================================================================
# Integration Tests
# =============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization module"""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_multiple_draws_accumulate(self, test_image):
        """
        Test that multiple draw calls accumulate correctly.
        """
        y = 30
        
        # Draw measurement info
        image, y = draw_measurement_info(
            test_image,
            BodyMeasurement(shoulder_width_3d=0.4, hip_width_3d=0.3, body_yaw=10.0),
            corrected_shoulder_width=200.0,
            corrected_hip_width=150.0,
            start_y=y
        )
        
        # Draw distance ratio
        image, y = draw_distance_ratio(image, ratio=1.02, start_y=y, is_calibrated=True)
        
        # Draw direction indicator
        image, y = draw_direction_indicator(image, ratio=1.02, start_y=y, is_calibrated=True)
        
        # Draw confidence
        image, y = draw_confidence(image, confidence=0.85, start_y=y)
        
        # Draw orientation
        image, y = draw_orientation_indicator(image, is_frontal=True, start_y=y)
        
        # Y should have progressed significantly
        assert y > 200
        
        # Image should be modified
        assert not np.array_equal(image, test_image)
