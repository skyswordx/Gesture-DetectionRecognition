# -*- coding: utf-8 -*-
"""
Tests for PatternGenerator class
"""

import numpy as np
import pytest
import os
import tempfile
import cv2

from pattern_generator import PatternGenerator


class TestPatternGenerator:
    """Unit tests for PatternGenerator"""
    
    def test_checkerboard_shape(self):
        """Test checkerboard returns correct shape"""
        width, height = 200, 150
        result = PatternGenerator.checkerboard((width, height), 25)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
    
    def test_checkerboard_pattern_alternation(self):
        """Test checkerboard pattern alternates correctly"""
        square_size = 10
        result = PatternGenerator.checkerboard((100, 100), square_size)
        
        # Top-left square (0,0) should be white
        assert result[0, 0, 0] == 255
        
        # Next square horizontally (0,1) should be black
        assert result[0, square_size, 0] == 0
        
        # Next square vertically (1,0) should be black
        assert result[square_size, 0, 0] == 0
        
        # Diagonal square (1,1) should be white
        assert result[square_size, square_size, 0] == 255
    
    def test_grid_lines_shape(self):
        """Test grid_lines returns correct shape"""
        width, height = 200, 150
        result = PatternGenerator.grid_lines((width, height), 30, 2)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
    
    def test_grid_lines_has_lines(self):
        """Test grid_lines contains black lines on white background"""
        result = PatternGenerator.grid_lines((100, 100), 20, 1)
        
        # Background should be white
        assert result[5, 5, 0] == 255  # Between lines
        
        # Lines should be black
        assert result[0, 0, 0] == 0  # First horizontal line
        assert result[20, 0, 0] == 0  # Second horizontal line
    
    def test_radial_pattern_shape(self):
        """Test radial_pattern returns correct shape"""
        width, height = 200, 150
        result = PatternGenerator.radial_pattern((width, height), 8)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
    
    def test_radial_pattern_has_center(self):
        """Test radial pattern has lines through center"""
        result = PatternGenerator.radial_pattern((100, 100), 5)
        center_x, center_y = 50, 50
        
        # Center should have a line (black pixel)
        assert result[center_y, center_x, 0] == 0
    
    def test_load_image_file_not_found(self):
        """Test load_image raises FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            PatternGenerator.load_image('nonexistent_file.png')
    
    def test_load_image_with_resize(self):
        """Test load_image can resize images"""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create and save a test image
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(temp_path, test_img)
            
            # Load with resize
            result = PatternGenerator.load_image(temp_path, (50, 50))
            assert result.shape == (50, 50, 3)
        finally:
            os.unlink(temp_path)
    
    def test_load_image_without_resize(self):
        """Test load_image preserves original size when no target specified"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            test_img = np.ones((100, 150, 3), dtype=np.uint8) * 128
            cv2.imwrite(temp_path, test_img)
            
            result = PatternGenerator.load_image(temp_path)
            assert result.shape == (100, 150, 3)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    # Run tests directly
    test = TestPatternGenerator()
    test.test_checkerboard_shape()
    print('test_checkerboard_shape: PASSED')
    test.test_checkerboard_pattern_alternation()
    print('test_checkerboard_pattern_alternation: PASSED')
    test.test_grid_lines_shape()
    print('test_grid_lines_shape: PASSED')
    test.test_grid_lines_has_lines()
    print('test_grid_lines_has_lines: PASSED')
    test.test_radial_pattern_shape()
    print('test_radial_pattern_shape: PASSED')
    test.test_radial_pattern_has_center()
    print('test_radial_pattern_has_center: PASSED')
    test.test_load_image_file_not_found()
    print('test_load_image_file_not_found: PASSED')
    test.test_load_image_with_resize()
    print('test_load_image_with_resize: PASSED')
    test.test_load_image_without_resize()
    print('test_load_image_without_resize: PASSED')
    print('\nAll PatternGenerator tests PASSED!')
