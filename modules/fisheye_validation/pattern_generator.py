# -*- coding: utf-8 -*-
"""
Pattern Generator for Fisheye Calibration Validation

This module provides various test pattern generators for validating
fisheye camera calibration parameters.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class PatternGenerator:
    """
    Generator for various test patterns used in calibration validation.
    
    All methods are static and return numpy arrays representing grayscale
    or color images suitable for distortion testing.
    """
    
    @staticmethod
    def checkerboard(size: Tuple[int, int], square_size: int = 50) -> np.ndarray:
        """
        Generate a checkerboard pattern.
        
        Args:
            size: Output image size as (width, height)
            square_size: Size of each square in pixels (default: 50)
            
        Returns:
            Grayscale image with checkerboard pattern (uint8)
            
        Requirements: 4.1
        """
        width, height = size
        image = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                # Determine which square this pixel belongs to
                square_row = i // square_size
                square_col = j // square_size
                # Alternate colors based on sum of row and column indices
                if (square_row + square_col) % 2 == 0:
                    image[i, j] = 255  # White
                else:
                    image[i, j] = 0    # Black
        
        # Convert to 3-channel for consistency with other patterns
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    
    @staticmethod
    def grid_lines(size: Tuple[int, int], spacing: int = 50, thickness: int = 1) -> np.ndarray:
        """
        Generate a grid line pattern for visual distortion assessment.
        
        Args:
            size: Output image size as (width, height)
            spacing: Distance between grid lines in pixels (default: 50)
            thickness: Line thickness in pixels (default: 1)
            
        Returns:
            BGR image with grid line pattern (uint8)
            
        Requirements: 4.2
        """
        width, height = size
        # Start with white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw horizontal lines
        for y in range(0, height, spacing):
            cv2.line(image, (0, y), (width - 1, y), (0, 0, 0), thickness)
        
        # Draw vertical lines
        for x in range(0, width, spacing):
            cv2.line(image, (x, 0), (x, height - 1), (0, 0, 0), thickness)
        
        return image
    
    @staticmethod
    def radial_pattern(size: Tuple[int, int], num_rings: int = 10) -> np.ndarray:
        """
        Generate a radial concentric circle pattern to highlight barrel/pincushion distortion.
        
        Args:
            size: Output image size as (width, height)
            num_rings: Number of concentric rings (default: 10)
            
        Returns:
            BGR image with radial pattern (uint8)
            
        Requirements: 4.3
        """
        width, height = size
        # Start with white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate center and maximum radius
        center_x = width // 2
        center_y = height // 2
        max_radius = int(np.sqrt(center_x**2 + center_y**2))
        
        # Calculate spacing between rings
        ring_spacing = max_radius // num_rings if num_rings > 0 else max_radius
        
        # Draw concentric circles
        for i in range(1, num_rings + 1):
            radius = i * ring_spacing
            cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 1)
        
        # Draw radial lines (8 lines at 45-degree intervals)
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            end_x = int(center_x + max_radius * np.cos(rad))
            end_y = int(center_y + max_radius * np.sin(rad))
            cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 0, 0), 1)
        
        return image
    
    @staticmethod
    def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load a custom image as the test source.
        
        Args:
            path: Path to the image file
            target_size: Optional target size as (width, height) to resize to
            
        Returns:
            BGR image (uint8)
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be loaded
            
        Requirements: 4.4
        """
        image = cv2.imread(path)
        
        if image is None:
            # Check if file exists to provide better error message
            import os
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            else:
                raise ValueError(f"Failed to load image: {path}")
        
        if target_size is not None:
            width, height = target_size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return image
