"""
Fisheye Camera Calibrator
Provides fisheye camera calibration functionality using chessboard pattern
"""

import cv2
import numpy as np
import os
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Camera calibration result data structure"""
    camera_matrix: np.ndarray           # Camera intrinsic matrix
    distortion_coeffs: np.ndarray       # Distortion coefficients (k1,k2,k3,k4)
    rotation_vectors: List[np.ndarray]  # Rotation vectors for each calibration image
    translation_vectors: List[np.ndarray]  # Translation vectors for each calibration image
    rms_error: float                    # Root mean square reprojection error
    image_size: Tuple[int, int]         # Image dimensions (width, height)
    calibration_flags: int              # OpenCV calibration flags used
    fisheye_flags: int                  # OpenCV fisheye calibration flags
    is_valid: bool                      # Whether calibration is valid
    calibration_date: str               # Calibration timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration result to dictionary for JSON serialization"""
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'rms_error': float(self.rms_error),
            'image_size': list(self.image_size),
            'calibration_flags': int(self.calibration_flags),
            'fisheye_flags': int(self.fisheye_flags),
            'is_valid': bool(self.is_valid),
            'calibration_date': self.calibration_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """Create calibration result from dictionary"""
        return cls(
            camera_matrix=np.array(data['camera_matrix']),
            distortion_coeffs=np.array(data['distortion_coeffs']),
            rotation_vectors=[],  # Not serialized
            translation_vectors=[],  # Not serialized
            rms_error=data['rms_error'],
            image_size=tuple(data['image_size']),
            calibration_flags=data['calibration_flags'],
            fisheye_flags=data['fisheye_flags'],
            is_valid=data['is_valid'],
            calibration_date=data['calibration_date']
        )

class FisheyeCalibrator:
    """Fisheye camera calibration class"""
    
    def __init__(self, 
                 chessboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 1.0,
                 calibration_flags: int = None):
        """
        Initialize fisheye camera calibrator
        
        Args:
            chessboard_size: Number of inner corners per chessboard row and column
            square_size: Size of chessboard square in physical units (e.g., cm)
            calibration_flags: OpenCV fisheye calibration flags
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Default fisheye calibration flags
        if calibration_flags is None:
            self.calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
                                    cv2.fisheye.CALIB_CHECK_COND +
                                    cv2.fisheye.CALIB_FIX_SKEW)
        else:
            self.calibration_flags = calibration_flags
        
        # Prepare object points for chessboard
        self.object_points = self._prepare_object_points()
        
        # Storage for calibration data
        self.image_points = []
        self.object_points_list = []
        self.calibration_images = []
        self.image_size = None
        
        # Calibration result
        self.calibration_result: Optional[CalibrationResult] = None
        
        # Quality assessment
        self.min_images = 10
        self.max_rms_error = 1.0
        
        logger.info(f"Fisheye calibrator initialized with chessboard size {chessboard_size}")
    
    def _prepare_object_points(self) -> np.ndarray:
        """Prepare 3D object points for chessboard pattern"""
        object_points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 
                                       0:self.chessboard_size[1]].T.reshape(-1, 2)
        object_points *= self.square_size
        return object_points
    
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add a calibration image with chessboard pattern
        
        Args:
            image: Input image containing chessboard pattern
            
        Returns:
            True if chessboard was found and added successfully
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Set image size from first image
        if self.image_size is None:
            self.image_size = (gray.shape[1], gray.shape[0])
        
        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if found:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store calibration data
            self.image_points.append(refined_corners)
            self.object_points_list.append(self.object_points)
            self.calibration_images.append(image.copy())
            
            logger.info(f"Added calibration image {len(self.image_points)}, corners found: {len(refined_corners)}")
            return True
        else:
            logger.warning("Chessboard pattern not found in image")
            return False
    
    def calibrate(self) -> Optional[CalibrationResult]:
        """
        Perform fisheye camera calibration
        
        Returns:
            CalibrationResult if successful, None otherwise
        """
        if len(self.image_points) < self.min_images:
            logger.error(f"Need at least {self.min_images} calibration images, got {len(self.image_points)}")
            return None
        
        if self.image_size is None:
            logger.error("No image size information available")
            return None
        
        logger.info(f"Starting fisheye calibration with {len(self.image_points)} images")
        
        # Initialize camera matrix
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        # Convert lists to required format
        object_points_array = np.array(self.object_points_list, dtype=np.float32)
        image_points_array = np.array(self.image_points, dtype=np.float32)
        
        try:
            # Perform fisheye calibration
            rms_error, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points_array,
                image_points_array,
                self.image_size,
                K,
                D,
                flags=self.calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            
            # Assess calibration quality
            is_valid = (rms_error < self.max_rms_error and 
                       len(self.image_points) >= self.min_images)
            
            # Create calibration result
            self.calibration_result = CalibrationResult(
                camera_matrix=K,
                distortion_coeffs=D,
                rotation_vectors=rvecs,
                translation_vectors=tvecs,
                rms_error=rms_error,
                image_size=self.image_size,
                calibration_flags=self.calibration_flags,
                fisheye_flags=self.calibration_flags,
                is_valid=is_valid,
                calibration_date=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            logger.info(f"Fisheye calibration completed successfully")
            logger.info(f"RMS error: {rms_error:.4f}")
            logger.info(f"Camera matrix:\n{K}")
            logger.info(f"Distortion coefficients: {D.flatten()}")
            
            return self.calibration_result
            
        except Exception as e:
            logger.error(f"Fisheye calibration failed: {e}")
            return None
    
    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration result to file
        
        Args:
            filepath: Path to save calibration data
            
        Returns:
            True if saved successfully
        """
        if self.calibration_result is None:
            logger.error("No calibration result to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save calibration data as JSON
            with open(filepath, 'w') as f:
                json.dump(self.calibration_result.to_dict(), f, indent=2)
            
            logger.info(f"Calibration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration result from file
        
        Args:
            filepath: Path to calibration data file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.calibration_result = CalibrationResult.from_dict(data)
            self.image_size = self.calibration_result.image_size
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def get_calibration_quality_info(self) -> Dict[str, Any]:
        """Get calibration quality assessment information"""
        if self.calibration_result is None:
            return {"status": "No calibration available"}
        
        quality_info = {
            "num_images": len(self.image_points),
            "rms_error": self.calibration_result.rms_error,
            "is_valid": self.calibration_result.is_valid,
            "image_size": self.calibration_result.image_size,
            "calibration_date": self.calibration_result.calibration_date
        }
        
        # Quality assessment
        if self.calibration_result.rms_error < 0.5:
            quality_info["quality"] = "Excellent"
        elif self.calibration_result.rms_error < 1.0:
            quality_info["quality"] = "Good"
        elif self.calibration_result.rms_error < 2.0:
            quality_info["quality"] = "Fair"
        else:
            quality_info["quality"] = "Poor"
        
        return quality_info
    
    def reset(self):
        """Reset calibrator state for new calibration"""
        self.image_points.clear()
        self.object_points_list.clear()
        self.calibration_images.clear()
        self.image_size = None
        self.calibration_result = None
        logger.info("Calibrator state reset")
