"""
手势识别模块
基于人体关键点进行手势识别和动作分类
"""

import numpy as np
import cv2
import time
import logging
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GestureResult:
    """手势识别结果"""
    gesture: str  # 识别的手势名称
    confidence: float  # 识别置信度
    details: Dict  # 详细信息 (角度、位置等)
    timestamp: float  # 时间戳
    duration: float  # 手势持续时间

class GestureClassifier:
    """手势分类器基类"""
    
    def __init__(self, name: str):
        """
        初始化分类器
        
        Args:
            name: 分类器名称
        """
        self.name = name
        self.confidence_threshold = 0.7
        
    def classify(self, landmarks, frame_info: Dict = None) -> Tuple[float, Dict]:
        """
        分类手势
        
        Args:
            landmarks: 人体关键点
            frame_info: 帧信息 (宽度、高度等)
            
        Returns:
            (置信度, 详细信息)
        """
        raise NotImplementedError("子类必须实现classify方法")

class TakeoffGestureClassifier(GestureClassifier):
    """起飞手势分类器 - 双手高举过头"""
    
    def __init__(self):
        super().__init__("takeoff")
        self.confidence_threshold = 0.8
    
    def classify(self, landmarks, frame_info: Dict = None) -> Tuple[float, Dict]:
        """识别起飞手势"""
        if not landmarks or len(landmarks) < 33:
            return 0.0, {}
        
        try:
            # 关键点索引
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            nose = landmarks[0]
            
            # 检查关键点可见性
            required_points = [left_wrist, right_wrist, left_shoulder, right_shoulder, nose]
            if any(point.visibility < 0.5 for point in required_points):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            confidence = 0.0
            
            # 1. 检查双手是否高于头部
            hands_above_head = (left_wrist.y < nose.y and right_wrist.y < nose.y)
            details['hands_above_head'] = hands_above_head
            
            if not hands_above_head:
                return 0.0, details
            
            # 2. 检查双手高度对称性
            hand_symmetry = abs(left_wrist.y - right_wrist.y)
            details['hand_symmetry'] = hand_symmetry
            symmetry_score = max(0, 1 - hand_symmetry / 0.1)  # 对称性得分
            
            # 3. 检查双手是否足够高于肩膀
            left_height = left_shoulder.y - left_wrist.y
            right_height = right_shoulder.y - right_wrist.y
            details['left_arm_height'] = left_height
            details['right_arm_height'] = right_height
            
            min_height = 0.15  # 最小高度阈值
            height_score = min(1.0, (left_height + right_height) / (2 * min_height))
            
            # 4. 检查手臂张开程度
            hand_spread = abs(left_wrist.x - right_wrist.x)
            details['hand_spread'] = hand_spread
            spread_score = min(1.0, hand_spread / 0.4)  # 期望至少0.4的归一化距离
            
            # 5. 综合计算置信度
            if left_height > min_height and right_height > min_height:
                confidence = (symmetry_score * 0.3 + height_score * 0.4 + 
                            spread_score * 0.3)
                
                # 额外奖励：完美对称和足够高度
                if hand_symmetry < 0.05 and min(left_height, right_height) > 0.2:
                    confidence = min(1.0, confidence * 1.2)
            
            details['final_confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            logger.error(f"起飞手势识别错误: {e}")
            return 0.0, {"error": str(e)}

class LandingGestureClassifier(GestureClassifier):
    """降落手势分类器 - 双手向下压"""
    
    def __init__(self):
        super().__init__("landing")
        self.confidence_threshold = 0.8
    
    def classify(self, landmarks, frame_info: Dict = None) -> Tuple[float, Dict]:
        """识别降落手势"""
        if not landmarks or len(landmarks) < 33:
            return 0.0, {}
        
        try:
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            required_points = [left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip]
            if any(point.visibility < 0.5 for point in required_points):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 1. 检查双手是否低于腰部
            hip_y = (left_hip.y + right_hip.y) / 2
            hands_below_waist = (left_wrist.y > hip_y and right_wrist.y > hip_y)
            details['hands_below_waist'] = hands_below_waist
            
            if not hands_below_waist:
                return 0.0, details
            
            # 2. 检查双手水平程度
            hand_level_diff = abs(left_wrist.y - right_wrist.y)
            details['hand_level_diff'] = hand_level_diff
            level_score = max(0, 1 - hand_level_diff / 0.08)
            
            # 3. 检查手臂向下伸直程度
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            
            # 手腕应该低于肘部
            left_arm_down = left_wrist.y > left_elbow.y
            right_arm_down = right_wrist.y > right_elbow.y
            details['arms_pointing_down'] = left_arm_down and right_arm_down
            
            # 4. 检查手势强度 (越低越好)
            depth_below_waist = min(left_wrist.y - hip_y, right_wrist.y - hip_y)
            details['depth_below_waist'] = depth_below_waist
            depth_score = min(1.0, depth_below_waist / 0.15)
            
            # 5. 计算置信度
            confidence = 0.0
            if hands_below_waist and left_arm_down and right_arm_down:
                confidence = level_score * 0.4 + depth_score * 0.6
                
                # 额外奖励：手掌朝下姿势 (通过手腕低于肘部判断)
                if (left_wrist.y > left_elbow.y + 0.05 and 
                    right_wrist.y > right_elbow.y + 0.05):
                    confidence = min(1.0, confidence * 1.2)
            
            details['final_confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            logger.error(f"降落手势识别错误: {e}")
            return 0.0, {"error": str(e)}

class DirectionGestureClassifier(GestureClassifier):
    """方向手势分类器 - 前后左右移动"""
    
    def __init__(self):
        super().__init__("direction")
        self.confidence_threshold = 0.7
    
    def classify(self, landmarks, frame_info: Dict = None) -> Tuple[float, Dict]:
        """识别方向手势"""
        if not landmarks or len(landmarks) < 33:
            return 0.0, {}
        
        # 检测各个方向
        directions = {
            'forward': self._detect_forward(landmarks),
            'left': self._detect_left(landmarks),
            'right': self._detect_right(landmarks),
            'up': self._detect_up(landmarks),
            'down': self._detect_down(landmarks)
        }
        
        # 找到最高置信度的方向
        best_direction = max(directions.items(), key=lambda x: x[1][0])
        direction_name, (confidence, details) = best_direction
        
        if confidence > self.confidence_threshold:
            details['detected_direction'] = direction_name
            return confidence, details
        
        return 0.0, {'all_directions': directions}
    
    def _detect_forward(self, landmarks) -> Tuple[float, Dict]:
        """检测前进手势 - 右手前推"""
        try:
            right_wrist = landmarks[16]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            left_shoulder = landmarks[11]
            nose = landmarks[0]
            
            if any(p.visibility < 0.5 for p in [right_wrist, right_shoulder, left_wrist, left_shoulder, nose]):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 右手前伸 (z坐标小于鼻子)
            hand_forward = right_wrist.z < nose.z - 0.05
            details['hand_forward'] = hand_forward
            
            if not hand_forward:
                return 0.0, details
            
            # 右手与肩膀水平
            hand_level = abs(right_wrist.y - right_shoulder.y) < 0.1
            details['hand_level'] = hand_level
            
            # 左手相对放松 (靠近身体)
            left_relaxed = abs(left_wrist.x - left_shoulder.x) < 0.15
            details['left_relaxed'] = left_relaxed
            
            # 计算置信度
            confidence = 0.0
            if hand_forward:
                confidence = 0.6  # 基础分
                if hand_level:
                    confidence += 0.3
                if left_relaxed:
                    confidence += 0.1
            
            details['confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            return 0.0, {"error": str(e)}
    
    def _detect_left(self, landmarks) -> Tuple[float, Dict]:
        """检测左移手势 - 左手指向左侧"""
        try:
            left_wrist = landmarks[15]
            left_shoulder = landmarks[11]
            right_wrist = landmarks[16]
            right_shoulder = landmarks[12]
            
            if any(p.visibility < 0.5 for p in [left_wrist, left_shoulder, right_wrist, right_shoulder]):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 左手向左伸展
            hand_extended = left_wrist.x < left_shoulder.x - 0.15
            details['hand_extended'] = hand_extended
            
            if not hand_extended:
                return 0.0, details
            
            # 左手水平
            hand_level = abs(left_wrist.y - left_shoulder.y) < 0.1
            details['hand_level'] = hand_level
            
            # 右手相对放松
            right_relaxed = abs(right_wrist.x - right_shoulder.x) < 0.15
            details['right_relaxed'] = right_relaxed
            
            # 计算置信度
            confidence = 0.0
            if hand_extended:
                confidence = 0.6
                if hand_level:
                    confidence += 0.3
                if right_relaxed:
                    confidence += 0.1
            
            details['confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            return 0.0, {"error": str(e)}
    
    def _detect_right(self, landmarks) -> Tuple[float, Dict]:
        """检测右移手势 - 右手指向右侧"""
        try:
            right_wrist = landmarks[16]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            left_shoulder = landmarks[11]
            
            if any(p.visibility < 0.5 for p in [right_wrist, right_shoulder, left_wrist, left_shoulder]):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 右手向右伸展
            hand_extended = right_wrist.x > right_shoulder.x + 0.15
            details['hand_extended'] = hand_extended
            
            if not hand_extended:
                return 0.0, details
            
            # 右手水平
            hand_level = abs(right_wrist.y - right_shoulder.y) < 0.1
            details['hand_level'] = hand_level
            
            # 左手相对放松
            left_relaxed = abs(left_wrist.x - left_shoulder.x) < 0.15
            details['left_relaxed'] = left_relaxed
            
            # 计算置信度
            confidence = 0.0
            if hand_extended:
                confidence = 0.6
                if hand_level:
                    confidence += 0.3
                if left_relaxed:
                    confidence += 0.1
            
            details['confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            return 0.0, {"error": str(e)}
    
    def _detect_up(self, landmarks) -> Tuple[float, Dict]:
        """检测上升手势 - 双手向上推"""
        try:
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            if any(p.visibility < 0.5 for p in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 双手都高于肩膀
            both_hands_up = (left_wrist.y < left_shoulder.y - 0.05 and 
                            right_wrist.y < right_shoulder.y - 0.05)
            details['both_hands_up'] = both_hands_up
            
            if not both_hands_up:
                return 0.0, details
            
            # 双手分开 (推举姿势)
            hands_separated = abs(left_wrist.x - right_wrist.x) > 0.2
            details['hands_separated'] = hands_separated
            
            # 双手高度相近
            hands_level = abs(left_wrist.y - right_wrist.y) < 0.1
            details['hands_level'] = hands_level
            
            # 计算置信度
            confidence = 0.0
            if both_hands_up:
                confidence = 0.5
                if hands_separated:
                    confidence += 0.3
                if hands_level:
                    confidence += 0.2
            
            details['confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            return 0.0, {"error": str(e)}
    
    def _detect_down(self, landmarks) -> Tuple[float, Dict]:
        """检测下降手势 - 双手向下压"""
        try:
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            if any(p.visibility < 0.5 for p in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 双手都低于肩膀
            both_hands_down = (left_wrist.y > left_shoulder.y + 0.05 and 
                              right_wrist.y > right_shoulder.y + 0.05)
            details['both_hands_down'] = both_hands_down
            
            if not both_hands_down:
                return 0.0, details
            
            # 双手分开
            hands_separated = abs(left_wrist.x - right_wrist.x) > 0.2
            details['hands_separated'] = hands_separated
            
            # 双手高度相近
            hands_level = abs(left_wrist.y - right_wrist.y) < 0.1
            details['hands_level'] = hands_level
            
            # 计算置信度
            confidence = 0.0
            if both_hands_down:
                confidence = 0.5
                if hands_separated:
                    confidence += 0.3
                if hands_level:
                    confidence += 0.2
            
            details['confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            return 0.0, {"error": str(e)}

class StopGestureClassifier(GestureClassifier):
    """停止手势分类器 - 双手胸前交叉或举起"""
    
    def __init__(self):
        super().__init__("stop")
        self.confidence_threshold = 0.7
    
    def classify(self, landmarks, frame_info: Dict = None) -> Tuple[float, Dict]:
        """识别停止手势"""
        if not landmarks or len(landmarks) < 33:
            return 0.0, {}
        
        try:
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            if any(p.visibility < 0.5 for p in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return 0.0, {"reason": "关键点不可见"}
            
            details = {}
            
            # 双手在胸前高度
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            chest_level = (abs(left_wrist.y - shoulder_y) < 0.15 and 
                          abs(right_wrist.y - shoulder_y) < 0.15)
            details['chest_level'] = chest_level
            
            if not chest_level:
                return 0.0, details
            
            # 双手靠近身体中心
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            hands_centered = (abs(left_wrist.x - center_x) < 0.25 and 
                             abs(right_wrist.x - center_x) < 0.25)
            details['hands_centered'] = hands_centered
            
            # 双手距离较近 (交叉或并拢)
            hands_close = abs(left_wrist.x - right_wrist.x) < 0.15
            details['hands_close'] = hands_close
            
            # 计算置信度
            confidence = 0.0
            if chest_level:
                confidence = 0.4
                if hands_centered:
                    confidence += 0.3
                if hands_close:
                    confidence += 0.3
            
            details['confidence'] = confidence
            return confidence, details
            
        except Exception as e:
            logger.error(f"停止手势识别错误: {e}")
            return 0.0, {"error": str(e)}

class GestureRecognizer:
    """手势识别器主类"""
    
    def __init__(self):
        """初始化手势识别器"""
        # 初始化各种手势分类器
        self.classifiers = {
            'takeoff': TakeoffGestureClassifier(),
            'landing': LandingGestureClassifier(),
            'direction': DirectionGestureClassifier(),
            'stop': StopGestureClassifier()
        }
          # 手势历史和状态管理
        self.gesture_history = deque()
        self.max_history_size = 10
        self.current_gesture = "none"
        self.gesture_start_time = 0
        self.min_gesture_duration = 0.5  # 最小手势持续时间(秒)
        self.max_gesture_duration = 5.0  # 最大手势持续时间(秒)
        
        # 统计信息
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.gesture_counts = {}
        
        logger.info("手势识别器初始化完成")
    
    def recognize(self, landmarks, frame_info: Dict = None) -> GestureResult:
        """
        识别手势
        
        Args:
            landmarks: 人体关键点
            frame_info: 帧信息
            
        Returns:
            手势识别结果
        """
        current_time = time.time()
        self.total_recognitions += 1
        
        if not landmarks:
            return self._create_result("none", 0.0, {}, current_time)
        
        # 对所有分类器进行识别
        classification_results = {}
        for name, classifier in self.classifiers.items():
            try:
                confidence, details = classifier.classify(landmarks, frame_info)
                classification_results[name] = {
                    'confidence': confidence,
                    'details': details
                }
            except Exception as e:
                logger.error(f"分类器 {name} 错误: {e}")
                classification_results[name] = {
                    'confidence': 0.0,
                    'details': {'error': str(e)}
                }
        
        # 处理方向手势的特殊情况
        direction_result = classification_results.get('direction', {})
        if (direction_result.get('confidence', 0) > 0.7 and 
            'detected_direction' in direction_result.get('details', {})):
            
            direction_name = direction_result['details']['detected_direction']
            classification_results[direction_name] = direction_result
            classification_results.pop('direction', None)
        
        # 找到最高置信度的手势
        best_gesture = "none"
        best_confidence = 0.0
        best_details = {}
        
        for gesture_name, result in classification_results.items():
            confidence = result['confidence']
            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name
                best_details = result['details']
        
        # 应用时序一致性检查
        final_gesture, final_confidence = self._apply_temporal_consistency(
            best_gesture, best_confidence, current_time
        )
        
        # 更新统计
        if final_confidence > 0.7:
            self.successful_recognitions += 1
            self.gesture_counts[final_gesture] = self.gesture_counts.get(final_gesture, 0) + 1
        
        return self._create_result(final_gesture, final_confidence, best_details, current_time)
    
    def _apply_temporal_consistency(self, gesture: str, confidence: float, 
                                  current_time: float) -> Tuple[str, float]:
        """应用时序一致性检查"""
        
        # 如果置信度太低，返回none
        if confidence < 0.5:
            self._reset_current_gesture(current_time)
            return "none", 0.0
        
        # 如果是新手势
        if gesture != self.current_gesture:
            # 检查是否需要重置
            if self.current_gesture != "none":
                gesture_duration = current_time - self.gesture_start_time
                
                # 如果当前手势持续时间太短，可能是误识别
                if gesture_duration < self.min_gesture_duration:
                    return self.current_gesture, confidence * 0.8  # 降低置信度
            
            # 开始新手势
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            
            # 新手势需要更高的置信度
            if confidence < 0.7:
                return "none", 0.0
        
        else:
            # 继续当前手势
            gesture_duration = current_time - self.gesture_start_time
            
            # 检查手势是否持续过长
            if gesture_duration > self.max_gesture_duration:
                self._reset_current_gesture(current_time)
                return "none", 0.0
            
            # 持续手势的置信度可以稍低
            if confidence < 0.6:
                self._reset_current_gesture(current_time)
                return "none", 0.0
          # 添加到历史
        if len(self.gesture_history) >= self.max_history_size:
            self.gesture_history.popleft()
        self.gesture_history.append((gesture, confidence, current_time))
        
        return gesture, confidence
    
    def _reset_current_gesture(self, current_time: float):
        """重置当前手势"""
        self.current_gesture = "none"
        self.gesture_start_time = current_time
    
    def _create_result(self, gesture: str, confidence: float, 
                      details: Dict, timestamp: float) -> GestureResult:
        """创建手势识别结果"""
        duration = 0.0
        if gesture != "none" and self.gesture_start_time > 0:
            duration = timestamp - self.gesture_start_time
        
        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            details=details,
            timestamp=timestamp,
            duration=duration
        )
    
    def get_statistics(self) -> Dict:
        """获取识别统计信息"""
        success_rate = (self.successful_recognitions / self.total_recognitions * 100 
                       if self.total_recognitions > 0 else 0)
        
        return {
            "total_recognitions": self.total_recognitions,
            "successful_recognitions": self.successful_recognitions,
            "success_rate": success_rate,
            "current_gesture": self.current_gesture,
            "gesture_duration": time.time() - self.gesture_start_time if self.gesture_start_time > 0 else 0,
            "gesture_counts": self.gesture_counts.copy(),
            "recent_history": list(self.gesture_history)[-5:]  # 最近5个结果
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.gesture_counts.clear()
        self.gesture_history.clear()
        self._reset_current_gesture(time.time())
        logger.info("手势识别统计已重置")

class GestureVisualizer:
    """手势可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.gesture_colors = {
            'takeoff': (0, 255, 0),      # 绿色
            'landing': (0, 0, 255),      # 红色
            'forward': (255, 0, 0),      # 蓝色
            'left': (255, 255, 0),       # 青色
            'right': (255, 0, 255),      # 洋红色
            'up': (0, 255, 255),         # 黄色
            'down': (128, 0, 128),       # 紫色
            'stop': (255, 255, 255),     # 白色
            'none': (128, 128, 128)      # 灰色
        }
    
    def draw_gesture_info(self, image: np.ndarray, result: GestureResult, 
                         landmarks=None) -> np.ndarray:
        """
        在图像上绘制手势信息
        
        Args:
            image: 输入图像
            result: 手势识别结果
            landmarks: 关键点 (可选)
            
        Returns:
            绘制了信息的图像
        """
        if image is None:
            return None
        
        output = image.copy()
        color = self.gesture_colors.get(result.gesture, (255, 255, 255))
        
        # 获取图像尺寸，计算右下角位置
        height, width = output.shape[:2]
        
        # 主要手势信息
        gesture_text = f"Gesture: {result.gesture.upper()}"
        confidence_text = f"Confidence: {result.confidence:.2f}"
        duration_text = f"Duration: {result.duration:.1f}s"
        
        # 计算文本宽度以便右对齐
        font = cv2.FONT_HERSHEY_SIMPLEX
        gesture_size = cv2.getTextSize(gesture_text, font, 1.0, 2)[0]
        confidence_size = cv2.getTextSize(confidence_text, font, 0.7, 2)[0]
        duration_size = cv2.getTextSize(duration_text, font, 0.6, 2)[0]
        
        # 计算最大文本宽度
        max_width = max(gesture_size[0], confidence_size[0], duration_size[0])
        
        # 设置右下角位置 (留出边距)
        margin = 20
        gesture_x = width - max_width - margin
        gesture_y = height - 120
        confidence_x = width - confidence_size[0] - margin  
        confidence_y = height - 85
        duration_x = width - duration_size[0] - margin
        duration_y = height - 50
        
        # 绘制文本
        cv2.putText(output, gesture_text, (gesture_x, gesture_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(output, confidence_text, (confidence_x, confidence_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(output, duration_text, (duration_x, duration_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制置信度条
        self._draw_confidence_bar(output, result.confidence, color)
        
        # 如果有详细信息，绘制
        if result.details and isinstance(result.details, dict):
            self._draw_details(output, result.details)
        
        # 高亮相关关键点
        if landmarks and result.gesture != "none":
            self._highlight_relevant_landmarks(output, landmarks, result.gesture)
        
        return output
    def _draw_confidence_bar(self, image: np.ndarray, confidence: float, color: Tuple[int, int, int]):
        """绘制置信度条"""
        height, width = image.shape[:2]
        
        # 置信度条尺寸
        bar_width = 200
        bar_height = 15
        margin = 20
        
        # 在右下角位置 (文本下方)
        bar_x = width - bar_width - margin
        bar_y = height - 25  # 在duration文本下方一点
        
        # 背景条
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # 置信度条
        fill_width = int(bar_width * confidence)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # 边框
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 1)
    def _draw_details(self, image: np.ndarray, details: Dict):
        """绘制详细信息"""
        height, width = image.shape[:2]
        
        # 计算详细信息显示位置 (在主要信息的左侧)
        margin = 20
        detail_x = width - 450  # 在主要信息左侧显示
        detail_y_start = height - 140
        
        y_offset = detail_y_start
        for key, value in details.items():
            if key == 'error':
                continue
            
            if isinstance(value, bool):
                text = f"{key}: {'✓' if value else '✗'}"
                color = (0, 255, 0) if value else (0, 0, 255)
            elif isinstance(value, (int, float)):
                text = f"{key}: {value:.3f}"
                color = (255, 255, 255)
            else:
                text = f"{key}: {str(value)}"
                color = (200, 200, 200)
            
            # 确保不超出图像边界
            if detail_x > 0 and y_offset < height - 20:
                cv2.putText(image, text, (detail_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 20
    
    def _highlight_relevant_landmarks(self, image: np.ndarray, landmarks, gesture: str):
        """高亮相关关键点"""
        height, width = image.shape[:2]
        color = self.gesture_colors.get(gesture, (255, 255, 255))
        
        # 根据手势类型高亮不同的关键点
        highlight_indices = []
        
        if gesture == 'takeoff':
            highlight_indices = [11, 12, 15, 16]  # 肩膀和手腕
        elif gesture == 'landing':
            highlight_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # 上身和髋部
        elif gesture in ['forward', 'left', 'right']:
            if gesture == 'forward':
                highlight_indices = [12, 14, 16]  # 右臂
            elif gesture == 'left':
                highlight_indices = [11, 13, 15]  # 左臂
            elif gesture == 'right':
                highlight_indices = [12, 14, 16]  # 右臂
        elif gesture in ['up', 'down']:
            highlight_indices = [11, 12, 13, 14, 15, 16]  # 双臂
        elif gesture == 'stop':
            highlight_indices = [11, 12, 15, 16]  # 肩膀和手腕
        
        # 绘制高亮圆圈
        for idx in highlight_indices:
            if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                x = int(landmarks[idx].x * width)
                y = int(landmarks[idx].y * height)
                cv2.circle(image, (x, y), 8, color, 3)

def test_gesture_recognition():
    """测试手势识别功能"""
    print("测试手势识别...")
    
    # 导入依赖模块
    import sys
    sys.path.append('..')
    from image_processing.image_processor import CameraCapture, ImageVisualizer
    from pose_detection.pose_detector import PoseDetector
    
    # 初始化组件
    capture = CameraCapture(camera_id=0)
    pose_detector = PoseDetector()
    gesture_recognizer = GestureRecognizer()
    gesture_visualizer = GestureVisualizer()
    img_visualizer = ImageVisualizer()
    
    if not capture.start():
        print("摄像头启动失败")
        return
    
    print("手势识别测试")
    print("支持的手势:")
    print("- 起飞: 双手高举过头")
    print("- 降落: 双手向下压")
    print("- 前进: 右手前推") 
    print("- 左移: 左手指向左")
    print("- 右移: 右手指向右")
    print("- 上升: 双手向上推")
    print("- 下降: 双手向下压")
    print("- 停止: 双手胸前交叉")
    print("按 'q' 退出, 's' 显示统计, 'r' 重置统计")
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 检测姿势
                pose_result = pose_detector.detect(frame)
                
                # 识别手势
                if pose_result.landmarks:
                    frame_info = {
                        'width': pose_result.frame_width,
                        'height': pose_result.frame_height
                    }
                    gesture_result = gesture_recognizer.recognize(
                        pose_result.landmarks, frame_info
                    )
                    
                    # 可视化
                    output_frame = gesture_visualizer.draw_gesture_info(
                        frame, gesture_result, pose_result.landmarks
                    )
                else:
                    # 没有检测到人体
                    gesture_result = gesture_recognizer.recognize(None)
                    output_frame = gesture_visualizer.draw_gesture_info(frame, gesture_result)
                
                key = img_visualizer.show_image(output_frame, "Gesture Recognition Test")
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = gesture_recognizer.get_statistics()
                    print("\n=== 手势识别统计 ===")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif key == ord('r'):
                    gesture_recognizer.reset_statistics()
                    print("统计信息已重置")
                    
            else:
                time.sleep(0.01)
    
    finally:
        capture.stop()
        img_visualizer.close_all()

if __name__ == "__main__":
    test_gesture_recognition()
