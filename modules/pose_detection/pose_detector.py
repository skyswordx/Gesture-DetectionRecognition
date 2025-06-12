"""
姿势检测模块
基于MediaPipe实现人体姿势检测和关键点提取
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PoseLandmark:
    """姿势关键点数据结构"""
    x: float  # 归一化x坐标 (0-1)
    y: float  # 归一化y坐标 (0-1)
    z: float  # 深度坐标 (相对)
    visibility: float  # 可见性 (0-1)

@dataclass
class PoseDetectionResult:
    """姿势检测结果"""
    landmarks: Optional[List[PoseLandmark]]  # 33个关键点
    bbox: Optional[Tuple[int, int, int, int]]  # 边界框 (x1, y1, x2, y2)
    confidence: float  # 检测置信度
    timestamp: float  # 时间戳
    frame_width: int  # 原图宽度
    frame_height: int  # 原图高度

class PoseDetector:
    """人体姿势检测器"""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_segmentation: bool = False):
        """
        初始化姿势检测器
        
        Args:
            model_complexity: 模型复杂度 (0-2, 越高越准确但越慢)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            enable_segmentation: 是否启用分割
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化姿势检测模型
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 性能统计
        self.total_detections = 0
        self.successful_detections = 0
        self.processing_times = []
        
        logger.info("姿势检测器初始化完成")
    
    def detect(self, image: np.ndarray) -> PoseDetectionResult:
        """
        检测人体姿势
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            姿势检测结果
        """
        start_time = time.time()
        
        if image is None:
            return PoseDetectionResult(None, None, 0.0, start_time, 0, 0)
        
        height, width = image.shape[:2]
        
        # 转换颜色空间 (BGR -> RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe检测
        results = self.pose.process(rgb_image)
        
        # 处理时间统计
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.total_detections += 1
        
        # 解析结果
        landmarks = None
        bbox = None
        confidence = 0.0
        
        if results.pose_landmarks:
            self.successful_detections += 1
            
            # 提取关键点
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append(PoseLandmark(lm.x, lm.y, lm.z, lm.visibility))
            
            # 计算边界框
            bbox = self._calculate_bbox(landmarks, width, height)
            
            # 计算置信度 (基于关键点可见性)
            confidence = self._calculate_confidence(landmarks)
        
        return PoseDetectionResult(
            landmarks=landmarks,
            bbox=bbox,
            confidence=confidence,
            timestamp=start_time,
            frame_width=width,
            frame_height=height
        )
    
    def _calculate_bbox(self, landmarks: List[PoseLandmark], 
                       width: int, height: int) -> Tuple[int, int, int, int]:
        """计算边界框"""
        if not landmarks:
            return None
        
        # 获取所有可见关键点的坐标
        visible_points = [(lm.x * width, lm.y * height) 
                         for lm in landmarks if lm.visibility > 0.5]
        
        if not visible_points:
            return None
        
        # 计算边界
        x_coords = [p[0] for p in visible_points]
        y_coords = [p[1] for p in visible_points]
        
        margin = 20  # 边界扩展
        x1 = max(0, int(min(x_coords)) - margin)
        y1 = max(0, int(min(y_coords)) - margin)
        x2 = min(width, int(max(x_coords)) + margin)
        y2 = min(height, int(max(y_coords)) + margin)
        
        return (x1, y1, x2, y2)
    
    def _calculate_confidence(self, landmarks: List[PoseLandmark]) -> float:
        """计算检测置信度"""
        if not landmarks:
            return 0.0
        
        # 重要关键点 (头部、躯干、四肢主要节点)
        important_indices = [0, 1, 2, 5, 6, 11, 12, 13, 14, 15, 16, 23, 24]
        
        total_confidence = 0.0
        valid_count = 0
        
        for i, landmark in enumerate(landmarks):
            if i in important_indices:
                total_confidence += landmark.visibility
                valid_count += 1
        
        return total_confidence / valid_count if valid_count > 0 else 0.0
    
    def get_statistics(self) -> Dict:
        """获取性能统计"""
        success_rate = (self.successful_detections / self.total_detections * 100 
                       if self.total_detections > 0 else 0)
        
        avg_processing_time = (np.mean(self.processing_times) 
                              if self.processing_times else 0)
        
        return {
            "total_detections": self.total_detections,
            "successful_detections": self.successful_detections,
            "success_rate": success_rate,
            "avg_processing_time_ms": avg_processing_time,
            "fps_estimate": 1000 / avg_processing_time if avg_processing_time > 0 else 0
        }

class PoseVisualizer:
    """姿势可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 绘图样式
        self.landmark_style = self.mp_drawing_styles.get_default_pose_landmarks_style()
        self.connection_style = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )
    
    def draw_pose(self, image: np.ndarray, result: PoseDetectionResult, 
                  draw_landmarks: bool = True, draw_bbox: bool = True,
                  draw_info: bool = True) -> np.ndarray:
        """
        在图像上绘制姿势信息
        
        Args:
            image: 输入图像
            result: 姿势检测结果
            draw_landmarks: 是否绘制关键点
            draw_bbox: 是否绘制边界框
            draw_info: 是否绘制信息文本
            
        Returns:
            绘制后的图像
        """
        if image is None or result is None:
            return image
        
        output_image = image.copy()
        
        # 绘制关键点和连接线
        if draw_landmarks and result.landmarks:
            # 转换为MediaPipe格式
            mp_landmarks = self._convert_to_mp_landmarks(result.landmarks)
            
            # 绘制
            self.mp_drawing.draw_landmarks(
                output_image,
                mp_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_style
            )
        
        # 绘制边界框
        if draw_bbox and result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 边界框标签
            label = f"Person ({result.confidence:.2f})"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制信息文本
        if draw_info:
            info_lines = [
                f"Confidence: {result.confidence:.3f}",
                f"Landmarks: {len(result.landmarks) if result.landmarks else 0}",
                f"Time: {time.time() - result.timestamp:.3f}s ago"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(output_image, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_image
    
    def _convert_to_mp_landmarks(self, landmarks: List[PoseLandmark]):
        """转换为MediaPipe landmarks格式"""
        from mediapipe.framework.formats import landmark_pb2
        
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        
        for lm in landmarks:
            landmark = landmark_list.landmark.add()
            landmark.x = lm.x
            landmark.y = lm.y
            landmark.z = lm.z
            landmark.visibility = lm.visibility
        
        return landmark_list
    
    def draw_skeleton_simple(self, image: np.ndarray, landmarks: List[PoseLandmark]) -> np.ndarray:
        """绘制简化骨架"""
        if not landmarks:
            return image
        
        output = image.copy()
        height, width = image.shape[:2]
        
        # 主要连接关系 (简化版)
        connections = [
            # 头部
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # 躯干
            (9, 10), (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            # 腿部
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]
        
        # 绘制连接线
        for start_idx, end_idx in connections:
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5):
                
                start_point = (int(landmarks[start_idx].x * width),
                              int(landmarks[start_idx].y * height))
                end_point = (int(landmarks[end_idx].x * width),
                            int(landmarks[end_idx].y * height))
                
                cv2.line(output, start_point, end_point, (0, 255, 0), 2)
        
        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            if landmark.visibility > 0.5:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(output, (x, y), 4, (0, 0, 255), -1)
        
        return output

class PoseAnalyzer:
    """姿势分析器"""
    
    def __init__(self):
        """初始化姿势分析器"""
        self.mp_pose = mp.solutions.pose
        
    def calculate_angle(self, point1: PoseLandmark, point2: PoseLandmark, 
                       point3: PoseLandmark) -> float:
        """
        计算三点构成的角度
        
        Args:
            point1, point2, point3: 三个关键点
            
        Returns:
            角度 (度)
        """
        # 转换为向量
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        # 计算向量
        ba = a - b
        bc = c - b
        
        # 计算角度
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_body_angles(self, landmarks: List[PoseLandmark]) -> Dict[str, float]:
        """
        获取身体主要关节角度
        
        Args:
            landmarks: 关键点列表
            
        Returns:
            角度字典
        """
        if not landmarks or len(landmarks) < 33:
            return {}
        
        angles = {}
        
        try:
            # 左臂角度 (肩膀-肘部-手腕)
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            if all(lm.visibility > 0.5 for lm in [left_shoulder, left_elbow, left_wrist]):
                angles['left_arm'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # 右臂角度
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            if all(lm.visibility > 0.5 for lm in [right_shoulder, right_elbow, right_wrist]):
                angles['right_arm'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # 左腿角度 (髋部-膝盖-踝部)
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            if all(lm.visibility > 0.5 for lm in [left_hip, left_knee, left_ankle]):
                angles['left_leg'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # 右腿角度
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            if all(lm.visibility > 0.5 for lm in [right_hip, right_knee, right_ankle]):
                angles['right_leg'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            
        except Exception as e:
            logger.error(f"角度计算错误: {e}")
        
        return angles
    
    def get_body_position(self, landmarks: List[PoseLandmark]) -> Dict[str, float]:
        """
        获取身体位置信息
        
        Args:
            landmarks: 关键点列表
            
        Returns:
            位置信息字典
        """
        if not landmarks:
            return {}
        
        position = {}
        
        try:
            # 头部位置
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            if nose.visibility > 0.5:
                position['head_x'] = nose.x
                position['head_y'] = nose.y
            
            # 肩膀中心
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                position['shoulder_center_x'] = (left_shoulder.x + right_shoulder.x) / 2
                position['shoulder_center_y'] = (left_shoulder.y + right_shoulder.y) / 2
                position['shoulder_width'] = abs(left_shoulder.x - right_shoulder.x)
            
            # 髋部中心
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                position['hip_center_x'] = (left_hip.x + right_hip.x) / 2
                position['hip_center_y'] = (left_hip.y + right_hip.y) / 2
                position['hip_width'] = abs(left_hip.x - right_hip.x)
            
            # 身体倾斜度
            if 'shoulder_center_x' in position and 'hip_center_x' in position:
                lean = position['shoulder_center_x'] - position['hip_center_x']
                position['body_lean'] = lean
        
        except Exception as e:
            logger.error(f"位置计算错误: {e}")
        
        return position

def test_pose_detection():
    """测试姿势检测功能"""
    print("测试姿势检测...")
    
    # 导入图像处理模块
    import sys
    sys.path.append('..')
    from image_processing.image_processor import CameraCapture, ImageVisualizer
    
    # 初始化组件
    capture = CameraCapture(camera_id=0)
    detector = PoseDetector()
    visualizer = PoseVisualizer()
    analyzer = PoseAnalyzer()
    img_visualizer = ImageVisualizer()
    
    if not capture.start():
        print("摄像头启动失败")
        return
    
    print("姿势检测测试，按 'q' 退出, 's' 显示统计信息, 'a' 显示角度信息")
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 检测姿势
                result = detector.detect(frame)
                
                # 可视化
                output_frame = visualizer.draw_pose(frame, result)
                
                # 显示角度信息
                if result.landmarks:
                    angles = analyzer.get_body_angles(result.landmarks)
                    position = analyzer.get_body_position(result.landmarks)
                    
                    # 在图像上绘制角度信息
                    y_offset = output_frame.shape[0] - 120
                    for name, angle in angles.items():
                        text = f"{name}: {angle:.1f}°"
                        cv2.putText(output_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 20
                
                key = img_visualizer.show_image(output_frame, "Pose Detection Test")
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = detector.get_statistics()
                    print("\n=== 检测统计 ===")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif key == ord('a') and result.landmarks:
                    angles = analyzer.get_body_angles(result.landmarks)
                    position = analyzer.get_body_position(result.landmarks)
                    print("\n=== 角度信息 ===")
                    for name, angle in angles.items():
                        print(f"{name}: {angle:.1f}°")
                    print("\n=== 位置信息 ===")
                    for name, pos in position.items():
                        print(f"{name}: {pos:.3f}")
            else:
                time.sleep(0.01)
    
    finally:
        capture.stop()
        img_visualizer.close_all()

if __name__ == "__main__":
    test_pose_detection()
