# -*- coding: utf-8 -*-
"""
ValidationEngine - Round-trip validation for fisheye calibration parameters

Performs distort -> undistort round-trip testing and computes quality metrics
(PSNR, SSIM) to verify mathematical consistency of calibration parameters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from .distortion_simulator import DistortionSimulator
except ImportError:
    from distortion_simulator import DistortionSimulator


@dataclass
class ValidationResult:
    """Results from round-trip validation."""
    psnr: float                    # Peak Signal-to-Noise Ratio (dB)
    ssim: float                    # Structural Similarity Index
    max_pixel_error: float         # Maximum pixel-level error
    mean_pixel_error: float        # Mean pixel-level error
    is_consistent: bool            # True if PSNR > 30 and SSIM > 0.95
    original_image: np.ndarray     # Original test image
    distorted_image: np.ndarray    # Image after applying distortion
    recovered_image: np.ndarray    # Image after undistortion (round-trip)
    difference_map: np.ndarray     # Difference heatmap (0-255)


class ValidationEngine:
    """
    Engine for round-trip validation of fisheye calibration parameters.
    
    Validates calibration by:
    1. Applying forward distortion to an undistorted image
    2. Applying undistortion to recover the original
    3. Computing quality metrics between original and recovered images
    """
    
    # Thresholds for consistency classification
    PSNR_THRESHOLD = 30.0  # dB
    SSIM_THRESHOLD = 0.95
    
    def __init__(self, simulator: DistortionSimulator, K: np.ndarray, D: np.ndarray):
        """
        Initialize the validation engine.
        
        Args:
            simulator: DistortionSimulator for applying forward distortion
            K: 3x3 camera intrinsic matrix for undistortion
            D: 4x1 or (4,) distortion coefficients for undistortion
        """
        self.simulator = simulator
        self.K = K.astype(np.float64)
        self.D = D.flatten().astype(np.float64).reshape(4, 1)

    
    @classmethod
    def from_calibration(cls, calib, simulator: Optional[DistortionSimulator] = None) -> 'ValidationEngine':
        """
        Create a ValidationEngine from a ParsedCalibration object.
        
        Args:
            calib: ParsedCalibration object with camera_matrix and dist_coeffs
            simulator: Optional DistortionSimulator (created from calib if not provided)
            
        Returns:
            ValidationEngine instance
        """
        if simulator is None:
            simulator = DistortionSimulator(calib.camera_matrix, calib.dist_coeffs)
        return cls(simulator, calib.camera_matrix, calib.dist_coeffs)
    
    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two images.
        
        PSNR = 10 * log10(MAX^2 / MSE)
        
        Args:
            img1: First image
            img2: Second image (same shape as img1)
            mask: Optional mask for valid pixels (non-zero = valid)
            
        Returns:
            PSNR value in dB. Returns infinity for identical images.
        """
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
        
        # Convert to float for computation
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        
        if mask is not None:
            # Apply mask - only consider valid pixels
            if len(img1.shape) == 3:
                mask_3d = np.stack([mask] * img1.shape[2], axis=-1)
                diff = (img1_f - img2_f) * (mask_3d > 0)
                n_pixels = np.sum(mask > 0) * img1.shape[2]
            else:
                diff = (img1_f - img2_f) * (mask > 0)
                n_pixels = np.sum(mask > 0)
            
            if n_pixels == 0:
                return 0.0
            mse = np.sum(diff ** 2) / n_pixels
        else:
            mse = np.mean((img1_f - img2_f) ** 2)
        
        if mse < 1e-10:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index between two images.
        
        Uses the standard SSIM formula with default constants:
        - C1 = (K1 * L)^2 where K1 = 0.01, L = 255
        - C2 = (K2 * L)^2 where K2 = 0.03, L = 255
        
        Args:
            img1: First image
            img2: Second image (same shape as img1)
            
        Returns:
            SSIM value in range [0, 1]. Returns 1.0 for identical images.
        """
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
        
        # Convert to grayscale if color
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Convert to float
        img1_f = img1_gray.astype(np.float64)
        img2_f = img2_gray.astype(np.float64)
        
        # SSIM constants
        L = 255.0
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        # Window size for local statistics
        window_size = 11
        
        # Compute local means using Gaussian blur
        mu1 = cv2.GaussianBlur(img1_f, (window_size, window_size), 1.5)
        mu2 = cv2.GaussianBlur(img2_f, (window_size, window_size), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (window_size, window_size), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (window_size, window_size), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1_f * img2_f, (window_size, window_size), 1.5) - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        
        # Return mean SSIM
        return float(np.mean(ssim_map))

    
    def _compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute a difference heatmap between two images.
        
        The heatmap shows pixel-level discrepancies, scaled to 0-255 range.
        
        Args:
            img1: First image
            img2: Second image (same shape as img1)
            
        Returns:
            Difference heatmap as uint8 image (0-255)
        """
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
        
        # Convert to float for computation
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        
        # Compute absolute difference
        if len(img1.shape) == 3:
            # For color images, compute per-channel difference and take max
            diff = np.abs(img1_f - img2_f)
            diff_map = np.max(diff, axis=2)
        else:
            diff_map = np.abs(img1_f - img2_f)
        
        # Scale to 0-255 range
        max_diff = np.max(diff_map)
        if max_diff > 0:
            diff_map = (diff_map / max_diff) * 255.0
        
        return diff_map.astype(np.uint8)
    
    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply fisheye undistortion to an image.
        
        Args:
            image: Distorted input image
            
        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]
        
        # Compute undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K, (w, h), cv2.CV_32FC1
        )
        
        # Apply remapping
        undistorted = cv2.remap(
            image, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return undistorted
    
    def run_round_trip(self, image: np.ndarray) -> ValidationResult:
        """
        Execute round-trip validation: distort -> undistort -> compare.
        
        Args:
            image: Undistorted test image
            
        Returns:
            ValidationResult with metrics and images
        """
        # Step 1: Apply forward distortion
        distorted = self.simulator.apply_distortion(image)
        
        # Step 2: Apply undistortion to recover original
        recovered = self._undistort_image(distorted)
        
        # Step 3: Compute quality metrics
        # Create a mask for valid pixels (non-black regions in recovered image)
        if len(recovered.shape) == 3:
            mask = np.any(recovered > 0, axis=2).astype(np.uint8)
        else:
            mask = (recovered > 0).astype(np.uint8)
        
        # Compute PSNR with mask for valid region
        psnr = self._compute_psnr(image, recovered, mask)
        
        # Compute SSIM (uses full image)
        ssim = self._compute_ssim(image, recovered)
        
        # Compute pixel errors
        diff = np.abs(image.astype(np.float64) - recovered.astype(np.float64))
        max_pixel_error = float(np.max(diff))
        mean_pixel_error = float(np.mean(diff))
        
        # Compute difference heatmap
        difference_map = self._compute_difference_map(image, recovered)
        
        # Determine consistency
        is_consistent = (psnr > self.PSNR_THRESHOLD) and (ssim > self.SSIM_THRESHOLD)
        
        # Check for all-black recovered image (potential parameter issues)
        if np.sum(recovered) == 0:
            is_consistent = False
        
        return ValidationResult(
            psnr=psnr,
            ssim=ssim,
            max_pixel_error=max_pixel_error,
            mean_pixel_error=mean_pixel_error,
            is_consistent=is_consistent,
            original_image=image,
            distorted_image=distorted,
            recovered_image=recovered,
            difference_map=difference_map
        )
