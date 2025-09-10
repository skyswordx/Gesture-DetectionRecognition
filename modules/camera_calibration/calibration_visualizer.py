"""
Calibration Visualizer
Provides visualization tools for camera calibration and distortion correction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import logging
from .fisheye_calibrator import CalibrationResult, FisheyeCalibrator
from .distortion_corrector import DistortionCorrector

logger = logging.getLogger(__name__)

class CalibrationVisualizer:
    """Visualization tools for camera calibration process"""
    
    def __init__(self):
        """Initialize calibration visualizer"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
    
    def draw_chessboard_corners(self, 
                               image: np.ndarray, 
                               corners: np.ndarray, 
                               chessboard_size: Tuple[int, int],
                               found: bool = True) -> np.ndarray:
        """
        Draw detected chessboard corners on image
        
        Args:
            image: Input image
            corners: Detected corner points
            chessboard_size: Chessboard pattern size
            found: Whether corners were found successfully
            
        Returns:
            Image with drawn corners
        """
        result = image.copy()
        
        if found and corners is not None:
            # Draw corners
            cv2.drawChessboardCorners(result, chessboard_size, corners, found)
            
            # Add status text
            status_text = f"Chessboard Found: {len(corners)} corners"
            cv2.putText(result, status_text, (10, 30), self.font, 0.8, (0, 255, 0), 2)
            
            # Draw corner numbers for first few corners
            for i, corner in enumerate(corners[:20]):  # Show only first 20 for clarity
                x, y = corner.ravel().astype(int)
                cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(result, str(i), (x + 5, y), self.font, 0.3, (255, 255, 255), 1)
        else:
            # No corners found
            cv2.putText(result, "Chessboard Not Found", (10, 30), self.font, 0.8, (0, 0, 255), 2)
            cv2.putText(result, "Ensure good lighting and clear pattern", (10, 60), 
                       self.font, 0.5, (0, 0, 255), 1)
        
        return result
    
    def draw_calibration_progress(self, 
                                calibrator: FisheyeCalibrator,
                                target_images: int = 20) -> np.ndarray:
        """
        Create calibration progress visualization
        
        Args:
            calibrator: Fisheye calibrator instance
            target_images: Target number of calibration images
            
        Returns:
            Progress visualization image
        """
        # Create progress panel
        panel_width = 400
        panel_height = 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Current progress
        current_images = len(calibrator.image_points)
        progress_ratio = min(current_images / target_images, 1.0)
        
        # Draw progress bar
        bar_x, bar_y = 50, 80
        bar_width, bar_height = 300, 30
        
        # Background bar
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress_ratio)
        if progress_width > 0:
            color = (0, 255, 0) if current_images >= calibrator.min_images else (0, 255, 255)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                         color, -1)
        
        # Title and text
        cv2.putText(panel, "Calibration Progress", (10, 30), self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"Images: {current_images}/{target_images}", (10, 60), 
                   self.font, 0.6, (255, 255, 255), 1)
        
        # Progress percentage
        cv2.putText(panel, f"{progress_ratio*100:.1f}%", (bar_x + bar_width + 10, bar_y + 20), 
                   self.font, 0.6, (255, 255, 255), 1)
        
        # Instructions
        if current_images < calibrator.min_images:
            cv2.putText(panel, "Need more images for calibration", (10, 140), 
                       self.font, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(panel, "Ready for calibration", (10, 140), 
                       self.font, 0.5, (0, 255, 0), 1)
        
        cv2.putText(panel, "Capture from different angles and distances", (10, 160), 
                   self.font, 0.4, (200, 200, 200), 1)
        
        return panel
    
    def visualize_calibration_result(self, calibration_result: CalibrationResult) -> np.ndarray:
        """
        Create calibration result visualization
        
        Args:
            calibration_result: Calibration result to visualize
            
        Returns:
            Visualization image
        """
        panel_width = 500
        panel_height = 400
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "Calibration Results", (10, 30), self.font, 0.8, (255, 255, 255), 2)
        
        # RMS Error with color coding
        rms_color = (0, 255, 0) if calibration_result.rms_error < 1.0 else (0, 255, 255) if calibration_result.rms_error < 2.0 else (0, 0, 255)
        cv2.putText(panel, f"RMS Error: {calibration_result.rms_error:.4f}", (10, 70), 
                   self.font, 0.6, rms_color, 2)
        
        # Quality assessment
        if calibration_result.rms_error < 0.5:
            quality = "Excellent"
            quality_color = (0, 255, 0)
        elif calibration_result.rms_error < 1.0:
            quality = "Good"
            quality_color = (0, 255, 255)
        elif calibration_result.rms_error < 2.0:
            quality = "Fair"
            quality_color = (0, 165, 255)
        else:
            quality = "Poor"
            quality_color = (0, 0, 255)
        
        cv2.putText(panel, f"Quality: {quality}", (10, 100), self.font, 0.6, quality_color, 1)
        
        # Image size
        cv2.putText(panel, f"Image Size: {calibration_result.image_size[0]}x{calibration_result.image_size[1]}", 
                   (10, 130), self.font, 0.5, (200, 200, 200), 1)
        
        # Calibration date
        cv2.putText(panel, f"Date: {calibration_result.calibration_date}", (10, 160), 
                   self.font, 0.5, (200, 200, 200), 1)
        
        # Camera matrix (simplified)
        cv2.putText(panel, "Camera Matrix (fx, fy, cx, cy):", (10, 200), 
                   self.font, 0.5, (255, 255, 255), 1)
        
        fx = calibration_result.camera_matrix[0, 0]
        fy = calibration_result.camera_matrix[1, 1]
        cx = calibration_result.camera_matrix[0, 2]
        cy = calibration_result.camera_matrix[1, 2]
        
        cv2.putText(panel, f"fx: {fx:.1f}, fy: {fy:.1f}", (10, 230), 
                   self.font, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, f"cx: {cx:.1f}, cy: {cy:.1f}", (10, 250), 
                   self.font, 0.4, (200, 200, 200), 1)
        
        # Distortion coefficients
        cv2.putText(panel, "Distortion Coefficients:", (10, 290), 
                   self.font, 0.5, (255, 255, 255), 1)
        
        dist_coeffs = calibration_result.distortion_coeffs.flatten()
        for i, coeff in enumerate(dist_coeffs):
            cv2.putText(panel, f"k{i+1}: {coeff:.6f}", (10, 320 + i * 20), 
                       self.font, 0.4, (200, 200, 200), 1)
        
        return panel
    
    def create_distortion_comparison(self, 
                                   original: np.ndarray, 
                                   corrected: np.ndarray,
                                   corrector: DistortionCorrector) -> np.ndarray:
        """
        Create detailed comparison between original and corrected images
        
        Args:
            original: Original fisheye image
            corrected: Corrected image
            corrector: Distortion corrector instance
            
        Returns:
            Detailed comparison image
        """
        # Basic comparison
        comparison = corrector.create_comparison_view(original, corrected)
        if comparison is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add parameter information
        params = corrector.get_correction_parameters()
        perf_stats = corrector.get_performance_stats()
        
        # Create info panel
        info_height = 100
        info_panel = np.zeros((info_height, comparison.shape[1], 3), dtype=np.uint8)
        
        # Parameter information
        y_offset = 20
        cv2.putText(info_panel, f"Balance: {params['balance']:.2f}", (10, y_offset), 
                   self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"FOV Scale: {params['fov_scale']:.2f}", (150, y_offset), 
                   self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Crop: {params['crop_factor']:.2f}", (300, y_offset), 
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Performance information
        if 'average_fps' in perf_stats:
            cv2.putText(info_panel, f"Avg FPS: {perf_stats['average_fps']:.1f}", (10, y_offset + 30), 
                       self.font, 0.5, (0, 255, 0), 1)
            cv2.putText(info_panel, f"Process Time: {perf_stats['average_process_time']*1000:.1f}ms", 
                       (150, y_offset + 30), self.font, 0.5, (0, 255, 0), 1)
        
        # Quality indicators
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
        
        # Calculate sharpness
        orig_sharpness = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        corr_sharpness = cv2.Laplacian(corrected_gray, cv2.CV_64F).var()
        
        cv2.putText(info_panel, f"Original Sharpness: {orig_sharpness:.0f}", (10, y_offset + 60), 
                   self.font, 0.4, (200, 200, 200), 1)
        cv2.putText(info_panel, f"Corrected Sharpness: {corr_sharpness:.0f}", (250, y_offset + 60), 
                   self.font, 0.4, (200, 200, 200), 1)
        
        # Combine images
        result = np.vstack([comparison, info_panel])
        return result
    
    def create_parameter_adjustment_panel(self, corrector: DistortionCorrector) -> np.ndarray:
        """
        Create interactive parameter adjustment panel
        
        Args:
            corrector: Distortion corrector instance
            
        Returns:
            Parameter adjustment panel
        """
        panel_width = 400
        panel_height = 300
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        params = corrector.get_correction_parameters()
        
        # Title
        cv2.putText(panel, "Correction Parameters", (10, 30), self.font, 0.7, (255, 255, 255), 2)
        
        # Balance parameter
        y_pos = 70
        cv2.putText(panel, f"Balance: {params['balance']:.2f}", (10, y_pos), 
                   self.font, 0.6, (255, 255, 255), 1)
        # Balance bar
        bar_x, bar_y = 150, y_pos - 10
        bar_width = 200
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        balance_pos = int(params['balance'] * bar_width)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + balance_pos, bar_y + 15), (0, 255, 0), -1)
        
        # FOV Scale parameter
        y_pos += 50
        cv2.putText(panel, f"FOV Scale: {params['fov_scale']:.2f}", (10, y_pos), 
                   self.font, 0.6, (255, 255, 255), 1)
        # FOV bar (normalized to 0-2 range)
        cv2.rectangle(panel, (bar_x, y_pos - 10), (bar_x + bar_width, y_pos + 5), (50, 50, 50), -1)
        fov_pos = int((params['fov_scale'] / 2.0) * bar_width)
        cv2.rectangle(panel, (bar_x, y_pos - 10), (bar_x + fov_pos, y_pos + 5), (255, 255, 0), -1)
        
        # Crop Factor parameter
        y_pos += 50
        cv2.putText(panel, f"Crop Factor: {params['crop_factor']:.2f}", (10, y_pos), 
                   self.font, 0.6, (255, 255, 255), 1)
        # Crop bar
        cv2.rectangle(panel, (bar_x, y_pos - 10), (bar_x + bar_width, y_pos + 5), (50, 50, 50), -1)
        crop_pos = int(params['crop_factor'] * bar_width)
        cv2.rectangle(panel, (bar_x, y_pos - 10), (bar_x + crop_pos, y_pos + 5), (0, 255, 255), -1)
        
        # Instructions
        cv2.putText(panel, "Keyboard Controls:", (10, y_pos + 50), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "1/2: Balance +/-", (10, y_pos + 75), self.font, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, "3/4: FOV Scale +/-", (10, y_pos + 95), self.font, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, "5/6: Crop +/-", (10, y_pos + 115), self.font, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, "r: Reset to defaults", (10, y_pos + 135), self.font, 0.4, (200, 200, 200), 1)
        
        return panel
    
    def draw_grid_overlay(self, image: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """
        Draw grid overlay to help visualize distortion correction
        
        Args:
            image: Input image
            grid_size: Grid cell size in pixels
            
        Returns:
            Image with grid overlay
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Draw vertical lines
        for x in range(0, w, grid_size):
            cv2.line(result, (x, 0), (x, h), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(result, (0, y), (w, y), (255, 255, 255), 1)
        
        return result
