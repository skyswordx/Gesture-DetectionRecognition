# -*- coding: utf-8 -*-
"""
CalibrationLoader - Load and parse fisheye calibration parameters

Supports both OpenCV and MATLAB-exported formats with automatic detection
and conversion.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np


@dataclass
class ParsedCalibration:
    """Parsed calibration parameters in OpenCV format."""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    was_converted: bool = False
    original_alpha: Optional[float] = None
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ParsedCalibration):
            return False
        return (
            np.allclose(self.camera_matrix, other.camera_matrix, rtol=1e-6, atol=1e-9) and
            np.allclose(self.dist_coeffs, other.dist_coeffs, rtol=1e-6, atol=1e-9) and
            self.image_size == other.image_size and
            self.was_converted == other.was_converted
        )


class CalibrationLoader:
    """Load and parse fisheye calibration parameters from JSON files."""
    
    MATLAB_ALPHA_THRESHOLD = 10.0
    MATLAB_FOCAL_THRESHOLD = 2.0
    
    def load(self, path: str) -> ParsedCalibration:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._parse_data(data)
    
    def load_from_dict(self, data: dict) -> ParsedCalibration:
        return self._parse_data(data)

    def _parse_data(self, data: dict) -> ParsedCalibration:
        if 'camera_matrix' not in data:
            raise ValueError("Missing required field: camera_matrix")
        if 'dist_coeffs' not in data:
            raise ValueError("Missing required field: dist_coeffs")
        
        K = np.array(data['camera_matrix'], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3, got {K.shape}")
        
        D = np.array(data['dist_coeffs'], dtype=np.float64).flatten()
        if len(D) != 4:
            raise ValueError(f"dist_coeffs must have 4 elements, got {len(D)}")
        D = D.reshape(4, 1)
        
        width = data.get('image_width', 1920)
        height = data.get('image_height', 1080)
        image_size = (int(width), int(height))
        
        was_converted = False
        original_alpha = None
        
        if self._detect_matlab_format(K, D):
            K, D, original_alpha = self._convert_matlab_to_opencv(K, D)
            was_converted = True
        
        return ParsedCalibration(
            camera_matrix=K,
            dist_coeffs=D,
            image_size=image_size,
            was_converted=was_converted,
            original_alpha=original_alpha
        )
    
    def _detect_matlab_format(self, K: np.ndarray, D: np.ndarray) -> bool:
        fx = K[0, 0]
        fy = K[1, 1]
        alpha = abs(D[0, 0])
        
        fx_normalized = abs(fx) < self.MATLAB_FOCAL_THRESHOLD
        fy_normalized = abs(fy) < self.MATLAB_FOCAL_THRESHOLD
        alpha_is_scale = alpha > self.MATLAB_ALPHA_THRESHOLD
        
        return fx_normalized and fy_normalized and alpha_is_scale
    
    def _convert_matlab_to_opencv(
        self, K: np.ndarray, D: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        alpha = D[0, 0]
        
        K_converted = K.copy()
        K_converted[0, 0] = K[0, 0] * alpha
        K_converted[1, 1] = K[1, 1] * alpha
        
        D_converted = np.array([
            D[1, 0], D[2, 0], D[3, 0], 0.0
        ], dtype=np.float64).reshape(4, 1)
        
        return K_converted, D_converted, float(alpha)
    
    def to_json(self, calib: ParsedCalibration) -> str:
        data = {
            'camera_matrix': calib.camera_matrix.tolist(),
            'dist_coeffs': calib.dist_coeffs.flatten().tolist(),
            'image_width': calib.image_size[0],
            'image_height': calib.image_size[1],
            'was_converted': calib.was_converted
        }
        if calib.original_alpha is not None:
            data['original_alpha'] = calib.original_alpha
        return json.dumps(data, indent=2)
    
    def pretty_print(self, calib: ParsedCalibration) -> str:
        K = calib.camera_matrix
        D = calib.dist_coeffs.flatten()
        
        lines = [
            "=" * 50,
            "Fisheye Calibration Parameters",
            "=" * 50,
            "",
            "Camera Matrix (K):",
            f"  fx = {K[0, 0]:.6f}",
            f"  fy = {K[1, 1]:.6f}",
            f"  cx = {K[0, 2]:.6f}",
            f"  cy = {K[1, 2]:.6f}",
            f"  skew = {K[0, 1]:.6e}",
            "",
            "Distortion Coefficients (D):",
            f"  k1 = {D[0]:.6e}",
            f"  k2 = {D[1]:.6e}",
            f"  k3 = {D[2]:.6e}",
            f"  k4 = {D[3]:.6e}",
            "",
            f"Image Size: {calib.image_size[0]} x {calib.image_size[1]}",
        ]
        
        if calib.was_converted:
            lines.extend([
                "",
                "Note: MATLAB-to-OpenCV conversion was applied",
                f"  Original alpha (scaling factor): {calib.original_alpha:.6f}"
            ])
        
        lines.append("=" * 50)
        return "\n".join(lines)
