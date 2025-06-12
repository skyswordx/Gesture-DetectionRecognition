"""
距离估算模块
基于人体关键点进行距离测量和深度估算
"""

import numpy as np
import cv2
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistanceResult:
    """距离估算结果"""
    distance: float  # 估算距离(米)
    confidence: float  # 估算置信度
    method: str  # 使用的估算方法
    raw_measurements: Dict[str, float]  # 原始测量值
    timestamp: float  # 时间戳

class CameraCalibration:
    """相机标定参数"""
    
    def __init__(self, focal_length: float = 500.0, sensor_width_mm: float = 6.17):
        """
        初始化相机参数
        
        Args:
            focal_length: 焦距(像素)
            sensor_width_mm: 传感器宽度(毫米)
        """
        self.focal_length = focal_length
        self.sensor_width_mm = sensor_width_mm
        
        # 人体测量参数(米)
        self.average_shoulder_width = 0.45  # 平均肩宽
        self.average_head_height = 0.23     # 平均头高
        self.average_arm_span = 1.68        # 平均臂展
        self.average_leg_length = 0.87      # 平均腿长
        self.average_body_height = 1.70     # 平均身高
        
    def update_focal_length(self, image_width: int, fov_degrees: float = 60):
        """
        根据视野角度更新焦距
        
        Args:
            image_width: 图像宽度(像素)
            fov_degrees: 水平视野角度(度)
        """
        fov_radians = math.radians(fov_degrees)
        self.focal_length = (image_width / 2) / math.tan(fov_radians / 2)
        logger.info(f"更新焦距为: {self.focal_length:.1f} 像素")

class KalmanFilter:
    """卡尔曼滤波器用于距离平滑"""
    
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """
        初始化卡尔曼滤波器
        
        Args:
            process_variance: 过程噪声方差
            measurement_variance: 测量噪声方差
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # 状态变量 [距离, 速度]
        self.state = np.array([3.0, 0.0])  # 初始距离3米，速度0
        self.covariance = np.eye(2) * 1.0
        
        # 状态转移矩阵 (假设时间间隔dt=0.033s, 30fps)
        dt = 0.033
        self.transition_matrix = np.array([[1, dt], [0, 1]])
        
        # 测量矩阵 (只测量距离)
        self.measurement_matrix = np.array([[1, 0]])
        
        # 噪声矩阵
        self.process_noise = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * process_variance
        self.measurement_noise = np.array([[measurement_variance]])
        
    def predict(self):
        """预测步骤"""
        # 预测状态
        self.state = np.dot(self.transition_matrix, self.state)
        
        # 预测协方差
        self.covariance = (np.dot(np.dot(self.transition_matrix, self.covariance), 
                                 self.transition_matrix.T) + self.process_noise)
    
    def update(self, measurement: float):
        """更新步骤"""
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.measurement_matrix, self.covariance), 
                   self.measurement_matrix.T) + self.measurement_noise
        K = np.dot(np.dot(self.covariance, self.measurement_matrix.T), 
                   np.linalg.inv(S))
        
        # 更新状态
        y = measurement - np.dot(self.measurement_matrix, self.state)
        self.state = self.state + np.dot(K, y)
        
        # 更新协方差
        I_KH = np.eye(2) - np.dot(K, self.measurement_matrix)
        self.covariance = np.dot(I_KH, self.covariance)
    
    def get_distance(self) -> float:
        """获取滤波后的距离"""
        return float(self.state[0])
    
    def get_velocity(self) -> float:
        """获取估算速度"""
        return float(self.state[1])

class DistanceEstimator:
    """距离估算器"""
    
    def __init__(self, camera_calibration: CameraCalibration = None):
        """
        初始化距离估算器
        
        Args:
            camera_calibration: 相机标定参数
        """
        self.calibration = camera_calibration or CameraCalibration()
        self.kalman_filter = KalmanFilter()
          # 历史数据
        self.distance_history = deque()  # 保存30帧历史
        self.measurement_history = deque()  # 保存10次测量历史
        self.max_history_size = 30
        self.max_measurement_size = 10
        
        # 统计信息
        self.total_estimates = 0
        self.successful_estimates = 0
        
    def estimate_distance(self, landmarks, frame_width: int, frame_height: int) -> DistanceResult:
        """
        估算距离
        
        Args:
            landmarks: 姿势关键点
            frame_width: 图像宽度
            frame_height: 图像高度
            
        Returns:
            距离估算结果
        """
        start_time = time.time()
        self.total_estimates += 1
        
        if not landmarks or len(landmarks) < 33:
            return DistanceResult(
                distance=3.0,
                confidence=0.0,
                method="default",
                raw_measurements={},
                timestamp=start_time
            )
        
        # 多种方法估算距离
        measurements = {}
        
        # 方法1: 基于肩宽
        shoulder_distance = self._estimate_by_shoulder_width(landmarks, frame_width)
        if shoulder_distance > 0:
            measurements['shoulder'] = shoulder_distance
        
        # 方法2: 基于头部高度
        head_distance = self._estimate_by_head_height(landmarks, frame_height)
        if head_distance > 0:
            measurements['head'] = head_distance
        
        # 方法3: 基于身体高度
        body_distance = self._estimate_by_body_height(landmarks, frame_height)
        if body_distance > 0:
            measurements['body'] = body_distance
        
        # 方法4: 基于手臂跨度
        arm_distance = self._estimate_by_arm_span(landmarks, frame_width)
        if arm_distance > 0:
            measurements['arm_span'] = arm_distance
        
        # 融合多种测量结果
        if measurements:
            fused_distance, confidence, method = self._fuse_measurements(measurements)
            self.successful_estimates += 1
        else:
            fused_distance, confidence, method = 3.0, 0.0, "default"
        
        # 卡尔曼滤波平滑
        self.kalman_filter.predict()
        if confidence > 0.3:  # 只有置信度足够高才更新滤波器
            self.kalman_filter.update(fused_distance)
        
        smoothed_distance = self.kalman_filter.get_distance()
          # 添加到历史记录
        if len(self.distance_history) >= self.max_history_size:
            self.distance_history.popleft()
        self.distance_history.append(smoothed_distance)
        
        if len(self.measurement_history) >= self.max_measurement_size:
            self.measurement_history.popleft()
        self.measurement_history.append(measurements)
        
        return DistanceResult(
            distance=smoothed_distance,
            confidence=confidence,
            method=method,
            raw_measurements=measurements,
            timestamp=start_time
        )
    
    def _estimate_by_shoulder_width(self, landmarks, frame_width: int) -> float:
        """基于肩宽估算距离"""
        try:
            # 左右肩膀关键点 (索引11, 12)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # 检查可见性
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                return 0.0
            
            # 计算肩宽像素距离
            shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * frame_width
            
            if shoulder_width_px < 10:  # 太小的肩宽不可信
                return 0.0
            
            # 距离 = (真实尺寸 × 焦距) / 像素尺寸
            distance = (self.calibration.average_shoulder_width * self.calibration.focal_length) / shoulder_width_px
            
            return max(0.3, min(15.0, distance))  # 限制在合理范围内
            
        except Exception as e:
            logger.error(f"肩宽距离估算错误: {e}")
            return 0.0
    
    def _estimate_by_head_height(self, landmarks, frame_height: int) -> float:
        """基于头部高度估算距离"""
        try:
            # 使用鼻子和嘴角来估算头部高度
            nose = landmarks[0]  # 鼻子
            left_mouth = landmarks[9]   # 左嘴角
            right_mouth = landmarks[10]  # 右嘴角
            
            if nose.visibility < 0.5 or left_mouth.visibility < 0.5 or right_mouth.visibility < 0.5:
                return 0.0
            
            # 计算头部高度 (鼻子到嘴的距离的约3倍)
            mouth_y = (left_mouth.y + right_mouth.y) / 2
            head_height_px = abs(nose.y - mouth_y) * frame_height * 3
            
            if head_height_px < 5:
                return 0.0
            
            distance = (self.calibration.average_head_height * self.calibration.focal_length) / head_height_px
            
            return max(0.3, min(15.0, distance))
            
        except Exception as e:
            logger.error(f"头部距离估算错误: {e}")
            return 0.0
    
    def _estimate_by_body_height(self, landmarks, frame_height: int) -> float:
        """基于身体高度估算距离"""
        try:
            # 使用头顶到脚底的距离
            nose = landmarks[0]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            if (nose.visibility < 0.5 or 
                (left_ankle.visibility < 0.5 and right_ankle.visibility < 0.5)):
                return 0.0
            
            # 选择可见性更好的脚踝
            ankle = left_ankle if left_ankle.visibility > right_ankle.visibility else right_ankle
            
            body_height_px = abs(nose.y - ankle.y) * frame_height
            
            if body_height_px < 50:  # 身体高度太小不可信
                return 0.0
            
            distance = (self.calibration.average_body_height * self.calibration.focal_length) / body_height_px
            
            return max(0.3, min(15.0, distance))
            
        except Exception as e:
            logger.error(f"身体高度距离估算错误: {e}")
            return 0.0
    
    def _estimate_by_arm_span(self, landmarks, frame_width: int) -> float:
        """基于手臂跨度估算距离"""
        try:
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            if left_wrist.visibility < 0.5 or right_wrist.visibility < 0.5:
                return 0.0
            
            arm_span_px = abs(left_wrist.x - right_wrist.x) * frame_width
            
            if arm_span_px < 20:
                return 0.0
            
            distance = (self.calibration.average_arm_span * self.calibration.focal_length) / arm_span_px
            
            return max(0.3, min(15.0, distance))
            
        except Exception as e:
            logger.error(f"手臂跨度距离估算错误: {e}")
            return 0.0
    
    def _fuse_measurements(self, measurements: Dict[str, float]) -> Tuple[float, float, str]:
        """
        融合多种测量结果
        
        Args:
            measurements: 测量结果字典
            
        Returns:
            (融合距离, 置信度, 主要方法)
        """
        if not measurements:
            return 3.0, 0.0, "none"
        
        # 权重定义 (根据可靠性)
        weights = {
            'shoulder': 0.4,    # 肩宽最可靠
            'body': 0.3,        # 身体高度较可靠
            'head': 0.2,        # 头部高度一般
            'arm_span': 0.1     # 手臂跨度最不可靠
        }
        
        # 异常值检测和过滤
        valid_measurements = {}
        values = list(measurements.values())
        
        if len(values) > 1:
            median_distance = np.median(values)
            
            for method, distance in measurements.items():
                # 过滤与中位数差异过大的值
                if abs(distance - median_distance) / median_distance < 0.5:
                    valid_measurements[method] = distance
        else:
            valid_measurements = measurements
        
        if not valid_measurements:
            return 3.0, 0.0, "filtered_out"
        
        # 加权平均
        total_weight = 0
        weighted_sum = 0
        
        for method, distance in valid_measurements.items():
            weight = weights.get(method, 0.1)
            weighted_sum += distance * weight
            total_weight += weight
        
        fused_distance = weighted_sum / total_weight if total_weight > 0 else 3.0
        
        # 置信度计算
        confidence = min(1.0, len(valid_measurements) / 4.0)  # 方法数量越多越可信
        
        # 一致性检查
        if len(valid_measurements) > 1:
            std_dev = np.std(list(valid_measurements.values()))
            consistency = max(0, 1 - std_dev / fused_distance)
            confidence *= consistency
        
        # 主要方法 (权重最高的)
        main_method = max(valid_measurements.keys(), 
                         key=lambda x: weights.get(x, 0))
        
        return fused_distance, confidence, main_method
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        success_rate = (self.successful_estimates / self.total_estimates * 100 
                       if self.total_estimates > 0 else 0)
        
        recent_distances = list(self.distance_history)[-10:]  # 最近10个距离
        
        stats = {
            "total_estimates": self.total_estimates,
            "successful_estimates": self.successful_estimates,
            "success_rate": success_rate,
            "current_distance": recent_distances[-1] if recent_distances else 0,
            "average_distance": np.mean(recent_distances) if recent_distances else 0,
            "distance_std": np.std(recent_distances) if len(recent_distances) > 1 else 0,
            "velocity": self.kalman_filter.get_velocity()
        }
        
        return stats
    
    def reset_filter(self):
        """重置卡尔曼滤波器"""
        self.kalman_filter = KalmanFilter()
        self.distance_history.clear()
        self.measurement_history.clear()
        logger.info("距离估算器已重置")

class DistanceVisualizer:
    """距离可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.distance_history = deque()
        self.max_history_size = 100
        
    def draw_distance_info(self, image: np.ndarray, result: DistanceResult, 
                          landmarks=None, bbox=None) -> np.ndarray:
        """
        在图像上绘制距离信息
        
        Args:
            image: 输入图像
            result: 距离估算结果
            landmarks: 关键点 (可选)
            bbox: 边界框 (可选)
            
        Returns:
            绘制了信息的图像
        """
        if image is None:
            return None
        
        output = image.copy()
        
        # 绘制距离信息
        distance_text = f"Distance: {result.distance:.2f}m"
        confidence_text = f"Confidence: {result.confidence:.2f}"
        method_text = f"Method: {result.method}"
        
        # 主要信息
        cv2.putText(output, distance_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, confidence_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(output, method_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制测量详情
        y_offset = 110
        for method, distance in result.raw_measurements.items():
            text = f"{method}: {distance:.2f}m"
            cv2.putText(output, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        # 在人体边界框上显示距离
        if bbox:
            x1, y1, x2, y2 = bbox
            distance_label = f"{result.distance:.1f}m"
            cv2.putText(output, distance_label, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 绘制距离指示器
        self._draw_distance_indicator(output, result.distance)
        
        return output
    
    def _draw_distance_indicator(self, image: np.ndarray, distance: float):
        """绘制距离指示器"""
        height, width = image.shape[:2]
        
        # 在右侧绘制距离条
        bar_x = width - 50
        bar_y_start = 50
        bar_height = 200
        bar_width = 20
        
        # 背景条
        cv2.rectangle(image, (bar_x, bar_y_start), 
                     (bar_x + bar_width, bar_y_start + bar_height), 
                     (50, 50, 50), -1)
        
        # 距离映射到条的高度 (1-10米映射到条的长度)
        distance_ratio = max(0, min(1, (distance - 1) / 9))
        fill_height = int(bar_height * (1 - distance_ratio))
        
        # 根据距离选择颜色
        if distance < 2:
            color = (0, 0, 255)  # 红色 - 太近
        elif distance < 5:
            color = (0, 255, 255)  # 黄色 - 适中
        else:
            color = (0, 255, 0)  # 绿色 - 较远
        
        # 填充条
        cv2.rectangle(image, (bar_x, bar_y_start + fill_height), 
                     (bar_x + bar_width, bar_y_start + bar_height), 
                     color, -1)
        
        # 刻度标记
        for i in range(0, 11, 2):  # 0, 2, 4, 6, 8, 10米
            mark_y = bar_y_start + int(bar_height * i / 10)
            cv2.line(image, (bar_x - 5, mark_y), (bar_x, mark_y), (255, 255, 255), 1)
            cv2.putText(image, f"{10-i}", (bar_x - 25, mark_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def test_distance_estimation():
    """测试距离估算功能"""
    print("测试距离估算...")
    
    # 导入依赖模块
    import sys
    sys.path.append('..')
    from image_processing.image_processor import CameraCapture, ImageVisualizer
    from pose_detection.pose_detector import PoseDetector
    
    # 初始化组件
    capture = CameraCapture(camera_id=0)
    pose_detector = PoseDetector()
    distance_estimator = DistanceEstimator()
    distance_visualizer = DistanceVisualizer()
    img_visualizer = ImageVisualizer()
    
    if not capture.start():
        print("摄像头启动失败")
        return
    
    print("距离估算测试，按 'q' 退出, 's' 显示统计, 'r' 重置滤波器, 'c' 校准相机")
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 检测姿势
                pose_result = pose_detector.detect(frame)
                
                # 估算距离
                if pose_result.landmarks:
                    distance_result = distance_estimator.estimate_distance(
                        pose_result.landmarks, 
                        pose_result.frame_width, 
                        pose_result.frame_height
                    )
                    
                    # 可视化
                    output_frame = distance_visualizer.draw_distance_info(
                        frame, distance_result, 
                        pose_result.landmarks, 
                        pose_result.bbox
                    )
                else:
                    output_frame = frame
                
                key = img_visualizer.show_image(output_frame, "Distance Estimation Test")
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = distance_estimator.get_statistics()
                    print("\n=== 距离估算统计 ===")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif key == ord('r'):
                    distance_estimator.reset_filter()
                    print("滤波器已重置")
                elif key == ord('c'):
                    # 简单的相机校准
                    print("请输入你到摄像头的实际距离(米):")
                    try:
                        actual_distance = float(input())
                        if pose_result.landmarks and distance_result:
                            # 基于肩宽校准焦距
                            left_shoulder = pose_result.landmarks[11]
                            right_shoulder = pose_result.landmarks[12]
                            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                                shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * pose_result.frame_width
                                new_focal_length = (distance_estimator.calibration.average_shoulder_width * actual_distance) / (actual_distance * shoulder_width_px / distance_estimator.calibration.focal_length)
                                distance_estimator.calibration.focal_length = new_focal_length
                                print(f"焦距已校准为: {new_focal_length:.1f}")
                    except ValueError:
                        print("输入无效")
            else:
                time.sleep(0.01)
    
    finally:
        capture.stop()
        img_visualizer.close_all()

if __name__ == "__main__":
    test_distance_estimation()
