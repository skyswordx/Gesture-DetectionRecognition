"""
Distortion Corrector
Provides real-time fisheye image distortion correction functionality
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from .fisheye_calibrator import CalibrationResult

logger = logging.getLogger(__name__)

class DistortionCorrector:
    """Real-time fisheye distortion correction processor"""
    
    def __init__(self, calibration_result: Optional[CalibrationResult] = None):
        """
        Initialize distortion corrector
        
        Args:
            calibration_result: Camera calibration data
        """
        self.calibration_result = calibration_result
        self.is_initialized = False
        
        # Undistortion maps for performance optimization
        self.map1 = None
        self.map2 = None
        
        # Correction parameters
        self.balance = 0.0  # 0.0 = retain all original image, 1.0 = retain all pixels
        self.fov_scale = 1.0  # Field of view scaling factor
        self.crop_factor = 1.0  # Crop factor for output image
        
        # Performance monitoring
        self.frame_count = 0
        self.total_process_time = 0.0
        
        if calibration_result is not None:
            self.initialize_correction()
    
    def set_calibration(self, calibration_result: CalibrationResult) -> bool:
        """
        Set calibration data and initialize correction
        
        Args:
            calibration_result: Camera calibration data
            
        Returns:
            True if initialization successful
        """
        self.calibration_result = calibration_result
        return self.initialize_correction()
    
    def initialize_correction(self) -> bool:
        """
        Initialize undistortion maps for fast processing
        
        Returns:
            True if initialization successful
        """
        if self.calibration_result is None:
            logger.error("No calibration result available")
            return False
        
        if not self.calibration_result.is_valid:
            logger.error("Invalid calibration result")
            return False
        
        try:
            # Get image dimensions
            w, h = self.calibration_result.image_size
            
            # Get optimal new camera matrix for fisheye
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coeffs,
                (w, h),
                np.eye(3),
                balance=self.balance,
                new_size=(w, h),
                fov_scale=self.fov_scale
            )
            
            # Generate undistortion maps
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coeffs,
                np.eye(3),
                new_K,
                (w, h),
                cv2.CV_16SC2
            )
            
            self.new_camera_matrix = new_K
            self.is_initialized = True
            
            logger.info("Distortion correction initialized successfully")
            logger.info(f"Original image size: {w}x{h}")
            logger.info(f"Balance: {self.balance}, FOV scale: {self.fov_scale}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distortion correction: {e}")
            self.is_initialized = False
            return False
    
    def correct_distortion(self, image: np.ndarray, fast_mode: bool = True) -> Optional[np.ndarray]:
        """
        Correct fisheye distortion in image
        
        Args:
            image: Input fisheye image
            fast_mode: Use pre-computed maps for faster processing
            
        Returns:
            Corrected image or None if correction fails
        """
        if not self.is_initialized:
            logger.error("Distortion corrector not initialized")
            return None
        
        if image is None or image.size == 0:
            logger.error("Invalid input image")
            return None
        
        try:
            import time
            start_time = time.time()
            
            if fast_mode and self.map1 is not None and self.map2 is not None:
                # Fast correction using pre-computed maps
                corrected = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
            else:
                # Direct undistortion (slower but more flexible)
                corrected = cv2.fisheye.undistortImage(
                    image,
                    self.calibration_result.camera_matrix,
                    self.calibration_result.distortion_coeffs,
                    None,
                    self.new_camera_matrix
                )
            
            # Apply cropping if specified
            if self.crop_factor < 1.0:
                corrected = self._apply_crop(corrected)
            
            # Update performance statistics
            process_time = time.time() - start_time
            self.frame_count += 1
            self.total_process_time += process_time
            
            return corrected
            
        except Exception as e:
            logger.error(f"Distortion correction failed: {e}")
            return None
    
    def _apply_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply center crop to corrected image"""
        h, w = image.shape[:2]
        new_h = int(h * self.crop_factor)
        new_w = int(w * self.crop_factor)
        
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        return image[start_y:start_y + new_h, start_x:start_x + new_w]
    
    def set_correction_parameters(self, 
                                balance: Optional[float] = None,
                                fov_scale: Optional[float] = None,
                                crop_factor: Optional[float] = None) -> bool:
        """
        Update correction parameters and reinitialize
        
        Args:
            balance: Balance parameter (0.0-1.0)
            fov_scale: FOV scaling factor (0.1-2.0)
            crop_factor: Crop factor (0.1-1.0)
            
        Returns:
            True if update successful
        """
        updated = False
        
        if balance is not None and 0.0 <= balance <= 1.0:
            self.balance = balance
            updated = True
        
        if fov_scale is not None and 0.1 <= fov_scale <= 2.0:
            self.fov_scale = fov_scale
            updated = True
        
        if crop_factor is not None and 0.1 <= crop_factor <= 1.0:
            self.crop_factor = crop_factor
            updated = True
        
        if updated and self.calibration_result is not None:
            return self.initialize_correction()
        
        return updated
    
    def get_correction_parameters(self) -> Dict[str, Any]:
        """Get current correction parameters"""
        return {
            "balance": self.balance,
            "fov_scale": self.fov_scale,
            "crop_factor": self.crop_factor,
            "is_initialized": self.is_initialized,
            "has_calibration": self.calibration_result is not None
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.frame_count == 0:
            return {"status": "No frames processed"}
        
        avg_time = self.total_process_time / self.frame_count
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "frames_processed": self.frame_count,
            "total_time": self.total_process_time,
            "average_process_time": avg_time,
            "average_fps": avg_fps
        }
    
    def create_comparison_view(self, original: np.ndarray, corrected: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of original and corrected images
        
        Args:
            original: Original fisheye image
            corrected: Corrected image
            
        Returns:
            Comparison image
        """
        if original is None or corrected is None:
            return None
        
        # Resize images to same height
        h_orig, w_orig = original.shape[:2]
        h_corr, w_corr = corrected.shape[:2]
        
        target_height = min(h_orig, h_corr)
        scale_orig = target_height / h_orig
        scale_corr = target_height / h_corr
        
        orig_resized = cv2.resize(original, (int(w_orig * scale_orig), target_height))
        corr_resized = cv2.resize(corrected, (int(w_corr * scale_corr), target_height))
        
        # Create comparison image
        comparison = np.hstack([orig_resized, corr_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original (Fisheye)", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Corrected", (orig_resized.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        
        return comparison
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.total_process_time = 0.0
        logger.info("Performance statistics reset")

class AdaptiveDistortionCorrector(DistortionCorrector):
    """Adaptive distortion corrector with auto-parameter adjustment"""
    
    def __init__(self, calibration_result: Optional[CalibrationResult] = None):
        super().__init__(calibration_result)
        
        # Adaptive parameters
        self.auto_balance = True
        self.target_fov_coverage = 0.85  # Target FOV coverage ratio
        self.min_valid_pixels_ratio = 0.7  # Minimum valid pixels ratio
        
    def auto_adjust_parameters(self, sample_image: np.ndarray) -> bool:
        """
        Automatically adjust correction parameters based on sample image
        
        Args:
            sample_image: Sample image for parameter optimization
            
        Returns:
            True if adjustment successful
        """
        if not self.is_initialized:
            return False
        
        logger.info("Starting auto-adjustment of correction parameters")
        
        best_balance = self.balance
        best_score = 0.0
        
        # Test different balance values
        test_balances = np.linspace(0.0, 1.0, 11)
        
        for balance in test_balances:
            self.set_correction_parameters(balance=balance)
            corrected = self.correct_distortion(sample_image)
            
            if corrected is not None:
                score = self._evaluate_correction_quality(corrected)
                if score > best_score:
                    best_score = score
                    best_balance = balance
        
        # Apply best parameters
        self.set_correction_parameters(balance=best_balance)
        
        logger.info(f"Auto-adjustment completed. Best balance: {best_balance:.2f}, Score: {best_score:.3f}")
        return True
    
    def _evaluate_correction_quality(self, corrected_image: np.ndarray) -> float:
        """
        Evaluate quality of corrected image
        
        Args:
            corrected_image: Corrected image to evaluate
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Convert to grayscale if needed
        if len(corrected_image.shape) == 3:
            gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = corrected_image
        
        # Calculate valid pixels ratio (non-black pixels)
        valid_pixels = np.sum(gray > 10)
        total_pixels = gray.size
        valid_ratio = valid_pixels / total_pixels
        
        # Calculate image sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize sharpness score
        sharpness_score = min(sharpness / 1000.0, 1.0)
        
        # Combined score
        quality_score = (valid_ratio * 0.6 + sharpness_score * 0.4)
        
        return quality_score
