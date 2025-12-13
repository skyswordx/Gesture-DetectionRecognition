# -*- coding: utf-8 -*-
"""
DistortionSimulator - Apply fisheye distortion to undistorted images

Implements forward projection using the OpenCV fisheye equidistant model:
θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)

This is the inverse operation of undistortion, used for round-trip validation.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from .calibration_loader import ParsedCalibration
except ImportError:
    from calibration_loader import ParsedCalibration


class DistortionSimulator:
    """Forward fisheye distortion simulator using equidistant projection model."""
    
    def __init__(self, K: np.ndarray, D: np.ndarray):
        """
        Initialize the distortion simulator.
        
        Args:
            K: 3x3 camera intrinsic matrix
            D: 4x1 or (4,) distortion coefficients [k1, k2, k3, k4]
        """
        self.K = K.astype(np.float64)
        self.D = D.flatten().astype(np.float64)
        if len(self.D) != 4:
            raise ValueError(f"dist_coeffs must have 4 elements, got {len(self.D)}")
        
        # Cache for distortion maps
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._cached_shape: Optional[Tuple[int, int]] = None
    
    @classmethod
    def from_calibration(cls, calib: ParsedCalibration) -> 'DistortionSimulator':
        """Create a DistortionSimulator from a ParsedCalibration object."""
        return cls(calib.camera_matrix, calib.dist_coeffs)
    
    def _fisheye_distort_point(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """
        Apply fisheye distortion to a single normalized point.
        
        OpenCV fisheye equidistant projection model:
        r = sqrt(x² + y²)
        θ = atan(r)
        θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
        x' = (θd/r) * x
        y' = (θd/r) * y
        
        Args:
            x_norm: Normalized x coordinate (x / z in camera frame)
            y_norm: Normalized y coordinate (y / z in camera frame)
            
        Returns:
            Tuple of distorted normalized coordinates (x', y')
        """
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # Handle center point (avoid division by zero)
        if r < 1e-8:
            return x_norm, y_norm
        
        theta = np.arctan(r)
        theta2 = theta * theta
        
        k1, k2, k3, k4 = self.D
        
        # Equidistant projection with radial distortion
        theta_d = theta * (1 + k1*theta2 + k2*theta2**2 + k3*theta2**3 + k4*theta2**4)
        
        scale = theta_d / r
        return x_norm * scale, y_norm * scale

    def _build_distortion_map(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build forward distortion mapping tables for efficient remapping.
        
        For forward distortion (undistorted -> distorted), we need to find
        for each pixel in the OUTPUT (distorted) image, where to sample from
        the INPUT (undistorted) image.
        
        This is done by:
        1. For each output pixel (u_d, v_d), compute its normalized coords
        2. Apply inverse distortion to get undistorted normalized coords
        3. Convert back to pixel coords to get the source location
        
        However, OpenCV's fisheye model doesn't have a closed-form inverse.
        Instead, we use the forward mapping approach:
        1. For each input pixel (u, v), compute where it maps to in distorted space
        2. Build an inverse lookup by iterating over output pixels
        
        For efficiency, we use OpenCV's fisheye.distortPoints which handles
        the forward mapping, then build the remap tables.
        
        Args:
            shape: (height, width) of the image
            
        Returns:
            Tuple of (map_x, map_y) for cv2.remap
        """
        h, w = shape
        
        # Extract camera parameters
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # Create grid of output (distorted) pixel coordinates
        u_d = np.arange(w, dtype=np.float32)
        v_d = np.arange(h, dtype=np.float32)
        u_d_grid, v_d_grid = np.meshgrid(u_d, v_d)
        
        # For forward distortion mapping, we need to find for each distorted pixel
        # where to sample from the undistorted image.
        # 
        # The approach: for each distorted pixel, we need to "undistort" it to find
        # the corresponding undistorted pixel. This is the inverse of what we want
        # to simulate, but it gives us the correct remap tables.
        #
        # Actually, for forward distortion (adding distortion), we want:
        # distorted_image[v_d, u_d] = undistorted_image[v_u, u_u]
        # where (u_u, v_u) = undistort(u_d, v_d)
        #
        # So we use cv2.fisheye.undistortPoints to find source coordinates
        
        # Reshape for OpenCV
        points_distorted = np.stack([u_d_grid.ravel(), v_d_grid.ravel()], axis=-1)
        points_distorted = points_distorted.reshape(-1, 1, 2).astype(np.float64)
        
        # Undistort the distorted coordinates to get source coordinates
        # This tells us where to sample from the undistorted image
        D_reshaped = self.D.reshape(4, 1)
        points_undistorted = cv2.fisheye.undistortPoints(
            points_distorted, 
            self.K, 
            D_reshaped,
            P=self.K  # Use same K to get pixel coordinates
        )
        
        # Reshape back to image dimensions
        points_undistorted = points_undistorted.reshape(h, w, 2)
        
        map_x = points_undistorted[:, :, 0].astype(np.float32)
        map_y = points_undistorted[:, :, 1].astype(np.float32)
        
        return map_x, map_y
    
    def apply_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Apply fisheye distortion to an undistorted image.
        
        Args:
            image: Input undistorted image (H, W) or (H, W, C)
            
        Returns:
            Distorted image with same shape as input
        """
        if image.size == 0:
            return image.copy()
        
        # Get image shape
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        shape = (h, w)
        
        # Build or reuse cached maps
        if self._map_x is None or self._cached_shape != shape:
            self._map_x, self._map_y = self._build_distortion_map(shape)
            self._cached_shape = shape
        
        # Apply remapping with border handling
        distorted = cv2.remap(
            image,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return distorted
    
    def clear_cache(self) -> None:
        """Clear cached distortion maps."""
        self._map_x = None
        self._map_y = None
        self._cached_shape = None
