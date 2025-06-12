"""
无人机手势控制系统 - 核心实现
基于MediaPipe的实时手势识别与无人机控制
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import deque
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DroneCommand:
    """无人机控制指令数据结构"""
    action: str  # takeoff, land, move, rotate, hover
    params: Dict[str, float]  # 参数字典
    confidence: float  # 识别置信度
    timestamp: float  # 时间戳
    priority: int = 1  # 优先级 (1-5, 5最高)

@dataclass
class PersonState:
    """人物状态信息"""
    distance: float  # 距离(米)
    position: Tuple[float, float]  # 在画面中的位置 (x, y)
    gesture: str  # 当前手势
    confidence: float  # 置信度
    bbox: Tuple[int, int, int, int]  # 边界框
    landmarks: Optional[any] = None  # MediaPipe关键点

class AdvancedGestureRecognizer:
    """高级手势识别器"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 手势历史缓冲区
        self.gesture_history = deque(maxsize=10)
        self.position_history = deque(maxsize=5)
        
        # 姿势阈值参数
        self.angle_threshold = 15  # 角度阈值(度)
        self.position_threshold = 0.1  # 位置阈值
        self.confidence_threshold = 0.8  # 置信度阈值
        
    def calculate_angle(self, point1, point2, point3):
        """计算三点构成的角度"""
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def detect_takeoff_gesture(self, landmarks) -> float:
        """检测起飞手势: 双手高举过头"""
        try:
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # 检查双手是否高于头部
            hands_above_head = (left_wrist.y < nose.y and right_wrist.y < nose.y)
            
            # 检查双手是否对称举起
            hand_symmetry = abs(left_wrist.y - right_wrist.y) < 0.1
            
            # 检查双手是否高于肩膀足够高度
            left_height = left_shoulder.y - left_wrist.y
            right_height = right_shoulder.y - right_wrist.y
            sufficient_height = (left_height > 0.15 and right_height > 0.15)
            
            if hands_above_head and hand_symmetry and sufficient_height:
                return 0.9
            elif hands_above_head and sufficient_height:
                return 0.7
            elif hands_above_head:
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"起飞手势检测错误: {e}")
            return 0.0
    
    def detect_land_gesture(self, landmarks) -> float:
        """检测降落手势: 双手向下压"""
        try:
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # 检查双手是否低于腰部
            hands_below_waist = (left_wrist.y > left_hip.y and right_wrist.y > right_hip.y)
            
            # 检查双手是否水平向下
            hands_horizontal = abs(left_wrist.y - right_wrist.y) < 0.08
            
            # 检查手掌朝向(基于手腕和肘部角度)
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            
            # 手臂向下伸直
            left_straight = abs(left_elbow.y - left_wrist.y) < 0.05
            right_straight = abs(right_elbow.y - right_wrist.y) < 0.05
            
            if hands_below_waist and hands_horizontal and left_straight and right_straight:
                return 0.9
            elif hands_below_waist and hands_horizontal:
                return 0.7
            elif hands_below_waist:
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"降落手势检测错误: {e}")
            return 0.0
    
    def detect_direction_gesture(self, landmarks, direction: str) -> float:
        """检测方向手势: 单手指向特定方向"""
        try:
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            if direction == "forward":
                # 右手前伸
                hand_forward = right_wrist.z < nose.z - 0.1
                hand_level = abs(right_wrist.y - right_shoulder.y) < 0.1
                left_relaxed = abs(left_wrist.x - left_shoulder.x) < 0.15
                
                if hand_forward and hand_level and left_relaxed:
                    return 0.9
                elif hand_forward and hand_level:
                    return 0.7
                else:
                    return 0.0
            
            elif direction == "left":
                # 左手向左伸展
                hand_extended = left_wrist.x < left_shoulder.x - 0.2
                hand_level = abs(left_wrist.y - left_shoulder.y) < 0.1
                right_relaxed = abs(right_wrist.x - right_shoulder.x) < 0.15
                
                if hand_extended and hand_level and right_relaxed:
                    return 0.9
                elif hand_extended and hand_level:
                    return 0.7
                else:
                    return 0.0
            
            elif direction == "right":
                # 右手向右伸展
                hand_extended = right_wrist.x > right_shoulder.x + 0.2
                hand_level = abs(right_wrist.y - right_shoulder.y) < 0.1
                left_relaxed = abs(left_wrist.x - left_shoulder.x) < 0.15
                
                if hand_extended and hand_level and left_relaxed:
                    return 0.9
                elif hand_extended and hand_level:
                    return 0.7
                else:
                    return 0.0
            
            elif direction == "up":
                # 双手向上推
                both_hands_up = (left_wrist.y < left_shoulder.y - 0.1 and 
                                right_wrist.y < right_shoulder.y - 0.1)
                hands_separated = abs(left_wrist.x - right_wrist.x) > 0.3
                
                if both_hands_up and hands_separated:
                    return 0.9
                elif both_hands_up:
                    return 0.7
                else:
                    return 0.0
            
            elif direction == "down":
                # 双手向下压
                both_hands_down = (left_wrist.y > left_shoulder.y + 0.1 and 
                                  right_wrist.y > right_shoulder.y + 0.1)
                hands_separated = abs(left_wrist.x - right_wrist.x) > 0.3
                
                if both_hands_down and hands_separated:
                    return 0.9
                elif both_hands_down:
                    return 0.7
                else:
                    return 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"{direction}方向手势检测错误: {e}")
            return 0.0
    
    def detect_stop_gesture(self, landmarks) -> float:
        """检测停止手势: 双手胸前交叉或握拳"""
        try:
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # 双手在胸前
            chest_level = (abs(left_wrist.y - left_shoulder.y) < 0.15 and 
                          abs(right_wrist.y - right_shoulder.y) < 0.15)
            
            # 双手靠近身体中心
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            hands_centered = (abs(left_wrist.x - center_x) < 0.2 and 
                             abs(right_wrist.x - center_x) < 0.2)
            
            # 双手距离较近(交叉状态)
            hands_close = abs(left_wrist.x - right_wrist.x) < 0.15
            
            if chest_level and hands_centered and hands_close:
                return 0.9
            elif chest_level and hands_centered:
                return 0.7
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"停止手势检测错误: {e}")
            return 0.0
    
    def recognize_gesture(self, landmarks) -> Tuple[str, float]:
        """综合手势识别"""
        if not landmarks:
            return "none", 0.0
        
        # 检测各种手势
        gestures = {
            "takeoff": self.detect_takeoff_gesture(landmarks),
            "land": self.detect_land_gesture(landmarks),
            "forward": self.detect_direction_gesture(landmarks, "forward"),
            "left": self.detect_direction_gesture(landmarks, "left"),
            "right": self.detect_direction_gesture(landmarks, "right"),
            "up": self.detect_direction_gesture(landmarks, "up"),
            "down": self.detect_direction_gesture(landmarks, "down"),
            "stop": self.detect_stop_gesture(landmarks)
        }
        
        # 找到置信度最高的手势
        best_gesture = max(gestures.items(), key=lambda x: x[1])
        gesture_name, confidence = best_gesture
        
        # 添加到历史记录
        self.gesture_history.append((gesture_name, confidence))
        
        # 时序平滑: 如果连续3帧都是同一手势且置信度>阈值，则确认
        if len(self.gesture_history) >= 3:
            recent_gestures = list(self.gesture_history)[-3:]
            if all(g[0] == gesture_name and g[1] > self.confidence_threshold 
                   for g in recent_gestures):
                return gesture_name, confidence
        
        # 如果置信度很高，可以直接返回
        if confidence > 0.9:
            return gesture_name, confidence
        
        return "none", 0.0

class DistanceEstimator:
    """距离估算器"""
    
    def __init__(self):
        # 相机参数 (需要根据实际相机标定)
        self.focal_length = 500  # 焦距(像素)
        self.real_shoulder_width = 0.45  # 平均肩宽(米)
        self.real_head_height = 0.23  # 平均头高(米)
        
        # 卡尔曼滤波器
        self.kf = self._init_kalman_filter()
        self.distance_history = deque(maxsize=10)
        
    def _init_kalman_filter(self):
        """初始化卡尔曼滤波器"""
        kf = cv2.KalmanFilter(2, 1)
        kf.measurementMatrix = np.array([[1, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kf.processNoiseCov = 0.03 * np.eye(2, dtype=np.float32)
        kf.measurementNoiseCov = np.array([[0.1]], np.float32)
        kf.errorCovPost = np.eye(2, dtype=np.float32)
        kf.statePost = np.array([3.0, 0], dtype=np.float32)  # 初始距离3米
        return kf
    
    def estimate_by_shoulder_width(self, landmarks, frame_width: int) -> float:
        """基于肩宽估算距离"""
        try:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * frame_width
            
            if shoulder_width_px > 0:
                distance = (self.real_shoulder_width * self.focal_length) / shoulder_width_px
                return max(0.5, min(10.0, distance))  # 限制在0.5-10米范围
            
        except Exception as e:
            logger.error(f"肩宽距离估算错误: {e}")
        
        return 3.0  # 默认距离
    
    def estimate_by_head_size(self, landmarks, frame_height: int) -> float:
        """基于头部大小估算距离"""
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # 计算头部高度
            head_top = min(left_ear.y, right_ear.y, nose.y)
            mouth = landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT]
            head_height_px = abs(mouth.y - head_top) * frame_height
            
            if head_height_px > 0:
                distance = (self.real_head_height * self.focal_length) / head_height_px
                return max(0.5, min(10.0, distance))
            
        except Exception as e:
            logger.error(f"头部距离估算错误: {e}")
        
        return 3.0
    
    def estimate_distance(self, landmarks, frame_width: int, frame_height: int) -> float:
        """多特征融合距离估算"""
        if not landmarks:
            return 3.0
        
        # 获取多种估算结果
        shoulder_dist = self.estimate_by_shoulder_width(landmarks, frame_width)
        head_dist = self.estimate_by_head_size(landmarks, frame_height)
        
        # 加权融合
        weights = [0.7, 0.3]  # 肩宽更可靠
        distances = [shoulder_dist, head_dist]
        
        # 过滤异常值
        valid_distances = [d for d in distances if 0.5 <= d <= 10.0]
        if not valid_distances:
            return 3.0
        
        # 计算加权平均
        if len(valid_distances) == len(distances):
            fused_distance = np.average(distances, weights=weights)
        else:
            fused_distance = np.mean(valid_distances)
        
        # 卡尔曼滤波平滑
        self.kf.correct(np.array([[fused_distance]], dtype=np.float32))
        prediction = self.kf.predict()
        smoothed_distance = prediction[0][0]
        
        # 添加到历史记录
        self.distance_history.append(smoothed_distance)
        
        return float(smoothed_distance)

class DroneCommandGenerator:
    """无人机指令生成器"""
    
    def __init__(self):
        self.gesture_to_action = {
            "takeoff": "takeoff",
            "land": "land", 
            "forward": "move",
            "left": "move",
            "right": "move",
            "up": "move",
            "down": "move",
            "stop": "hover"
        }
        
        self.base_speed = 1.0  # 基础移动速度 (m/s)
        self.max_speed = 3.0   # 最大移动速度
        self.safe_distance = 2.0  # 安全距离
        
    def calculate_speed_factor(self, distance: float, gesture: str) -> float:
        """根据距离计算速度因子"""
        if gesture in ["takeoff", "land", "stop"]:
            return 1.0
        
        # 距离越近，速度越慢
        if distance < 1.0:
            return 0.3
        elif distance < 2.0:
            return 0.6
        elif distance < 4.0:
            return 1.0
        else:
            return 1.5  # 距离远时可以更快
    
    def generate_command(self, gesture: str, confidence: float, distance: float, 
                        person_position: Tuple[float, float]) -> Optional[DroneCommand]:
        """生成无人机控制指令"""
        
        if confidence < 0.8:
            return None
        
        action = self.gesture_to_action.get(gesture)
        if not action:
            return None
        
        # 计算速度因子
        speed_factor = self.calculate_speed_factor(distance, gesture)
        
        # 生成参数
        params = {}
        priority = 1
        
        if action == "takeoff":
            params = {"altitude": 2.0}
            priority = 5  # 最高优先级
            
        elif action == "land":
            params = {}
            priority = 5  # 最高优先级
            
        elif action == "move":
            base_velocity = self.base_speed * speed_factor
            
            if gesture == "forward":
                params = {"vx": base_velocity, "vy": 0, "vz": 0}
            elif gesture == "left":
                params = {"vx": 0, "vy": -base_velocity, "vz": 0}
            elif gesture == "right":
                params = {"vx": 0, "vy": base_velocity, "vz": 0}
            elif gesture == "up":
                params = {"vx": 0, "vy": 0, "vz": base_velocity * 0.5}  # 垂直速度减半
            elif gesture == "down":
                params = {"vx": 0, "vy": 0, "vz": -base_velocity * 0.5}
            
            priority = 3
            
        elif action == "hover":
            params = {"vx": 0, "vy": 0, "vz": 0}
            priority = 4
        
        # 安全检查
        if distance < 0.8 and gesture in ["forward"]:
            # 距离太近时禁止前进
            return None
        
        return DroneCommand(
            action=action,
            params=params,
            confidence=confidence,
            timestamp=time.time(),
            priority=priority
        )

class DroneGestureControlSystem:
    """无人机手势控制主系统"""
    
    def __init__(self):
        # 初始化组件
        self.gesture_recognizer = AdvancedGestureRecognizer()
        self.distance_estimator = DistanceEstimator()
        self.command_generator = DroneCommandGenerator()
        
        # MediaPipe绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 摄像头
        self.cap = None
        self.is_running = False
        
        # 数据队列
        self.command_queue = queue.PriorityQueue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=3)
        
        # 状态信息
        self.current_person_state = None
        self.last_command_time = 0
        self.command_cooldown = 0.5  # 命令冷却时间
        
        # 线程
        self.detection_thread = None
        self.command_thread = None
        
        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def init_camera(self, camera_id: int = 0) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error("无法打开摄像头")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("摄像头初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """图像预处理"""
        # 镜像翻转
        frame = cv2.flip(frame, 1)
        
        # 图像增强
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        # 噪声过滤
        frame = cv2.bilateralFilter(frame, 5, 80, 80)
        
        return frame
    
    def detect_person_bbox(self, landmarks, frame_shape) -> Optional[Tuple[int, int, int, int]]:
        """检测人体边界框"""
        if not landmarks:
            return None
        
        h, w = frame_shape[:2]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        margin = 20
        x_min = max(0, int(min(x_coords)) - margin)
        y_min = max(0, int(min(y_coords)) - margin)
        x_max = min(w, int(max(x_coords)) + margin)
        y_max = min(h, int(max(y_coords)) + margin)
        
        return (x_min, y_min, x_max, y_max)
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 预处理
        processed_frame = self.preprocess_frame(frame)
        
        # MediaPipe检测
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.gesture_recognizer.pose.process(rgb_frame)
        
        person_state = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            
            # 估算距离
            distance = self.distance_estimator.estimate_distance(
                landmarks, frame.shape[1], frame.shape[0]
            )
            
            # 识别手势
            gesture, confidence = self.gesture_recognizer.recognize_gesture(landmarks)
            
            # 检测边界框
            bbox = self.detect_person_bbox(landmarks, frame.shape)
            
            # 计算人物在画面中的位置
            if bbox:
                center_x = (bbox[0] + bbox[2]) / 2 / frame.shape[1]
                center_y = (bbox[1] + bbox[3]) / 2 / frame.shape[0]
                position = (center_x, center_y)
            else:
                position = (0.5, 0.5)
            
            # 创建人物状态
            person_state = PersonState(
                distance=distance,
                position=position,
                gesture=gesture,
                confidence=confidence,
                bbox=bbox,
                landmarks=landmarks
            )
            
            # 生成控制指令
            if gesture != "none" and time.time() - self.last_command_time > self.command_cooldown:
                command = self.command_generator.generate_command(
                    gesture, confidence, distance, position
                )
                
                if command:
                    # 添加到指令队列 (优先级队列会自动排序)
                    try:
                        self.command_queue.put_nowait((5 - command.priority, command))
                        self.last_command_time = time.time()
                        logger.info(f"生成指令: {command.action} - {command.params} (置信度: {confidence:.2f})")
                    except queue.Full:
                        logger.warning("指令队列已满")
        
        self.current_person_state = person_state
        return self.draw_annotations(processed_frame, person_state)
    
    def draw_annotations(self, frame, person_state: Optional[PersonState]):
        """绘制标注信息"""
        annotated_frame = frame.copy()
        
        if person_state and person_state.landmarks:
            # 绘制骨骼点
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                person_state.landmarks,
                self.gesture_recognizer.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # 绘制边界框
            if person_state.bbox:
                x1, y1, x2, y2 = person_state.bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示距离信息
                distance_text = f"距离: {person_state.distance:.1f}m"
                cv2.putText(annotated_frame, distance_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示手势信息
            gesture_text = f"手势: {person_state.gesture} ({person_state.confidence:.2f})"
            cv2.putText(annotated_frame, gesture_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 显示FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示队列状态
        queue_text = f"指令队列: {self.command_queue.qsize()}"
        cv2.putText(annotated_frame, queue_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def detection_loop(self):
        """检测线程主循环"""
        logger.info("检测线程启动")
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("无法读取摄像头帧")
                continue
            
            try:
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 更新FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # 显示结果
                cv2.imshow('无人机手势控制', processed_frame)
                
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    cv2.imwrite(f'gesture_frame_{int(time.time())}.jpg', processed_frame)
                    logger.info("保存当前帧")
                
            except Exception as e:
                logger.error(f"检测循环错误: {e}")
                continue
        
        logger.info("检测线程结束")
    
    def command_processing_loop(self):
        """指令处理线程"""
        logger.info("指令处理线程启动")
        
        while self.is_running:
            try:
                # 等待指令 (timeout避免无限阻塞)
                priority, command = self.command_queue.get(timeout=1.0)
                
                # 处理指令 (这里应该连接到实际的无人机)
                self.execute_drone_command(command)
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"指令处理错误: {e}")
        
        logger.info("指令处理线程结束")
    
    def execute_drone_command(self, command: DroneCommand):
        """执行无人机指令 (模拟实现)"""
        # 这里应该实现真实的无人机控制逻辑
        # 例如使用MAVLink协议发送指令
        
        logger.info(f"执行无人机指令: {command.action}")
        logger.info(f"参数: {command.params}")
        logger.info(f"置信度: {command.confidence:.2f}")
        logger.info(f"优先级: {command.priority}")
        
        # 模拟指令执行时间
        time.sleep(0.1)
    
    def start(self, camera_id: int = 0):
        """启动系统"""
        logger.info("启动无人机手势控制系统")
        
        # 初始化摄像头
        if not self.init_camera(camera_id):
            return False
        
        self.is_running = True
        
        # 启动线程
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.command_thread = threading.Thread(target=self.command_processing_loop)
        
        self.detection_thread.start()
        self.command_thread.start()
        
        logger.info("系统启动成功")
        return True
    
    def stop(self):
        """停止系统"""
        logger.info("停止无人机手势控制系统")
        
        self.is_running = False
        
        # 等待线程结束
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        if self.command_thread:
            self.command_thread.join(timeout=2.0)
        
        # 释放资源
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        logger.info("系统已停止")

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建系统实例
    system = DroneGestureControlSystem()
    
    try:
        # 启动系统
        if system.start():
            print("系统运行中... 按'q'退出, 按's'保存帧")
            
            # 等待检测线程结束
            if system.detection_thread:
                system.detection_thread.join()
        else:
            print("系统启动失败")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        logger.error(f"系统运行错误: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
