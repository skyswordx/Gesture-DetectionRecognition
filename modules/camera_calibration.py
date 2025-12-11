# -*- coding: utf-8 -*-
"""
通用的鱼眼/针孔相机标定参数加载与去畸模块
- 统一主程序与测试脚本的导入: from camera_calibration import FisheyeCalibrator, DistortionCorrector, CalibrationVisualizer
- 支持从 OpenCV 风格与 MATLAB 导出的 JSON 读取
- 提供基于 OpenCV fisheye 的去畸矫正
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    rms_error: float = 0.0
    model: str = "fisheye"  # fisheye or pinhole
    flags: int = 0
    is_valid: bool = True


class FisheyeCalibrator:
    """
    这里不实现完整的采集标定流程，仅提供加载现有参数文件的能力，
    以便主程序/验证脚本统一读取。
    """
    def __init__(self, chessboard_size: Tuple[int, int] = (9, 6), square_size: float = 1.0):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.calibration_result: Optional[CalibrationResult] = None

    def load_calibration(self, path: str) -> bool:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 读取标定文件失败: {e}")
            return False

        try:
            parsed = self._parse_generic_json(data)
            self.calibration_result = parsed
            return True
        except Exception as e:
            print(f"❌ 解析标定文件失败: {e}")
            return False

    def _parse_generic_json(self, data: Dict[str, Any]) -> CalibrationResult:
        """
        同时兼容以下字段命名：
        - camera_matrix / K / intrinsic / intrinsics
        - dist_coeffs / distortion_coeffs / D
        - image_size / (image_width,image_height)
        - rms_error / rms_reprojection_error
        - model: 默认 fisheye
        """
        # 相机内参
        # 注意：某些MATLAB导出会把第一行写为 [sx, skew, cx]；第二行为 [0, sy, cy]
        # 这里统一按 3x3 读取后，不对数值做交换，必要时在下方做启发式转换。
        K = (data.get('camera_matrix') or data.get('K') or data.get('intrinsic') or data.get('intrinsics'))
        if K is None:
            # 一些MATLAB导出可能给的是3x3，某些示例给的是近似归一化参数，确保为3x3
            raise ValueError('缺少 camera_matrix/K/intrinsic 字段')
        K = np.array(K, dtype=np.float64)
        if K.shape != (3, 3):
            # 处理某些错误导出格式
            K = K.reshape(3, 3)

        # 畸变系数
        # MATLAB导出中常见 dist_coeffs = [k1, k2, k3, k4] 或含有多余项，OpenCV fisheye 需要前4项
        D = (data.get('dist_coeffs') or data.get('distortion_coeffs') or data.get('D'))
        if D is None:
            raise ValueError('缺少 dist_coeffs/distortion_coeffs/D 字段')
        D = np.array(D, dtype=np.float64).reshape(-1)
        # OpenCV fisheye 期望4系数 (k1, k2, k3, k4)，但也可接受长度>=4 取前4个
        if D.size >= 4:
            D = D[:4]
        else:
            # 若给出的是5参数(针孔k1,k2,p1,p2,k3)或其他，尝试降级到常规undistort
            # 在 fisheye 模式下至少需要4参数，这里填充0到4长度
            D = np.pad(D, (0, max(0, 4 - D.size)), mode='constant')

        # 启发式识别 MATLAB 特定导出：
        # 若 K 的 fx,fy 近似 1 且 D[0] 极大，则认为 D[0] 是一个缩放因子 alpha，
        # K 存储为 [sx, skew, cx; 0, sy, cy; 0,0,1]，最终像素级 K 应为：
        #   fx = alpha*sx, fy = alpha*sy, skew = alpha*skew，cx,cy 不缩放。
        if (K[0, 0] <= 2.5 and K[1, 1] <= 2.5 and D[0] > 10):
            alpha = float(D[0])
            sx = float(K[0, 0])
            skew = float(K[0, 1])
            cx = float(K[0, 2])
            sy = float(K[1, 1])
            cy = float(K[1, 2])
            fx = alpha * sx
            fy = alpha * sy
            s = alpha * skew
            K = np.array([[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            # 将剩余三项作为 fisheye k1,k2,k3，k4 置 0
            if D.size >= 4:
                D = np.array([D[1], D[2], D[3], 0.0], dtype=np.float64)

        # 图像尺寸
        if 'image_size' in data and data['image_size']:
            w, h = int(data['image_size'][0]), int(data['image_size'][1])
        else:
            w = int(data.get('image_width') or 0)
            h = int(data.get('image_height') or 0)
            if w <= 0 or h <= 0:
                # 若未提供，允许使用常见默认值，实际运行时需与输入帧一致
                w, h = 640, 480

        # 误差
        rms = float(data.get('rms_error') or data.get('rms_reprojection_error') or 0.0)

        # 模型
        model = data.get('model', 'fisheye')

        return CalibrationResult(
            camera_matrix=K,
            dist_coeffs=D.reshape(4, 1),
            image_size=(w, h),
            rms_error=rms,
            model=model,
            flags=int(data.get('calibration_flags') or data.get('fisheye_flags') or 0),
            is_valid=bool(data.get('is_valid', True))
        )


class DistortionCorrector:
    def __init__(self, calibration: Optional[CalibrationResult] = None):
        self.calibration = calibration
        self._maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._newK: Optional[np.ndarray] = None

    def set_calibration(self, calibration: CalibrationResult) -> bool:
        self.calibration = calibration
        self._maps = None
        self._newK = None
        return True

    def _ensure_maps(self, frame_shape: Tuple[int, int]) -> bool:
        if self.calibration is None:
            return False

        h, w = frame_shape[:2]
        K = self.calibration.camera_matrix.copy()
        D = self.calibration.dist_coeffs.copy()

        # 若当前帧尺寸与标定尺寸不一致，我们按比例缩放主点和焦距
        cw, ch = self.calibration.image_size
        if cw > 0 and ch > 0 and (w != cw or h != ch):
            sx, sy = w / float(cw), h / float(ch)
            # 对K进行尺度调整
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy

        # 使用 fisheye 矫正
        balance = 0.0
        self._newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), new_size=(w, h), balance=balance, fov_scale=1.0
        )

        # 部分MATLAB导出的参数在estimateNewCameraMatrix后会得到极小的焦距，导致画面被压缩成一点。
        # 检测到这种退化情况时，直接退回使用原始K作为新内参矩阵。
        if self._newK[0, 0] < 50 or self._newK[1, 1] < 50:
            self._newK = K.copy()

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), self._newK, (w, h), cv2.CV_16SC2
        )
        self._maps = (map1, map2)
        return True

    def correct_distortion(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self.calibration is None:
                return image
            if self._maps is None:
                ok = self._ensure_maps(image.shape)
                if not ok:
                    return image
            map1, map2 = self._maps
            undist = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # 检查是否接近全黑，必要时回退到使用原始K
            gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
            nonzero_ratio = float(np.count_nonzero(gray)) / (gray.shape[0] * gray.shape[1])
            if nonzero_ratio < 0.05:
                # fallback
                K = self.calibration.camera_matrix.copy()
                D = self.calibration.dist_coeffs.copy()
                h, w = image.shape[:2]
                cw, ch = self.calibration.image_size
                if cw > 0 and ch > 0 and (w != cw or h != ch):
                    sx, sy = w / float(cw), h / float(ch)
                    K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
                map1_fb, map2_fb = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
                )
                undist_fb = cv2.remap(image, map1_fb, map2_fb, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                return undist_fb

            return undist
        except Exception as e:
            print(f"去畸失败: {e}")
            return image


class CalibrationVisualizer:
    def draw_k_matrix(self, image: np.ndarray, K: np.ndarray) -> np.ndarray:
        out = image.copy()
        txt = [
            f"fx={K[0,0]:.1f}", f"fy={K[1,1]:.1f}", f"cx={K[0,2]:.1f}", f"cy={K[1,2]:.1f}"
        ]
        y = 30
        for t in txt:
            cv2.putText(out, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y += 24
        return out


__all__ = [
    'FisheyeCalibrator',
    'DistortionCorrector',
    'CalibrationVisualizer',
    'CalibrationResult',
]
