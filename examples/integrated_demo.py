"""
综合演示程序 - 完整的手势识别控制系统
展示所有模块的集成使用和实时手势控制功能
"""

import cv2
import time
import numpy as np
import sys
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
综合演示程序 - 完整的手势识别控制系统
展示所有模块的集成使用和实时手势控制功能
"""

import cv2
import time
import numpy as np
import sys
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    from image_processing.image_processor import CameraCapture, ImageProcessor, ImageQualityAssessment, ImageVisualizer
    from pose_detection.pose_detector import PoseDetector, PoseVisualizer, PoseAnalyzer
    from distance_estimation.distance_estimator import DistanceEstimator, DistanceVisualizer
    from gesture_recognition.gesture_recognizer import GestureRecognizer, GestureVisualizer
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有模块都在正确的位置")
    sys.exit(1)

class GestureControlSystem:
    """手势控制系统 - 集成所有模块"""
    
    def __init__(self, camera_id=0):
        """初始化系统"""
        print("初始化手势控制系统...")
        
        # 图像处理模块
        self.camera_capture = CameraCapture(camera_id=camera_id, width=640, height=480)
        self.image_processor = ImageProcessor()
        self.quality_assessor = ImageQualityAssessment()
        self.image_visualizer = ImageVisualizer()
        
        # 姿势检测模块
        self.pose_detector = PoseDetector(model_complexity=1)
        self.pose_visualizer = PoseVisualizer()
        self.pose_analyzer = PoseAnalyzer()
        
        # 距离估算模块
        self.distance_estimator = DistanceEstimator()
        self.distance_visualizer = DistanceVisualizer()
        
        # 手势识别模块
        self.gesture_recognizer = GestureRecognizer()
        self.gesture_visualizer = GestureVisualizer()
          # 系统状态
        self.is_running = False
        self.current_command = "none"
        self.last_command_time = 0
        self.display_mode = "full"  # full, pose_only, distance_only, gesture_only
        self.show_debug = True
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        
        print("手势控制系统初始化完成")
        print("支持的手势:")
        print("  🙌 起飞: 双手高举过头")
        print("  👇 降落: 双手向下压")
        print("  👉 前进: 右手前推")
        print("  👈 左移: 左手指向左侧")
        print("  👉 右移: 右手指向右侧")
        print("  ☝️ 上升: 双手向上推举")
        print("  👇 下降: 双手向下压")
        print("  ✋ 停止: 双手胸前交叉")
    
    def start(self):
        """启动系统"""
        print("启动摄像头...")
        if not self.camera_capture.start():
            print("❌ 摄像头启动失败")
            return False
        
        self.is_running = True
        print("✅ 系统启动成功")
        return True
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        self.camera_capture.stop()
        self.image_visualizer.close_all()
        print("系统已停止")
    
    def process_frame(self):
        """处理单帧图像"""
        # 1. 获取图像
        frame = self.camera_capture.get_frame()
        if frame is None:
            return None
        
        # 2. 图像预处理
        processed_frame = self.image_processor.preprocess(frame)
        
        # 3. 图像质量评估
        quality = self.quality_assessor.assess_quality(processed_frame)
        
        if not quality['valid']:
            # 显示质量警告
            warning_frame = self._draw_quality_warning(processed_frame, quality)
            return warning_frame
        
        # 4. 姿势检测
        pose_result = self.pose_detector.detect(processed_frame)
        
        if not pose_result.landmarks:
            # 显示"未检测到人体"
            no_person_frame = self._draw_no_person_warning(processed_frame)
            return no_person_frame
        
        # 5. 距离估算
        distance_result = self.distance_estimator.estimate_distance(
            pose_result.landmarks,
            pose_result.frame_width,
            pose_result.frame_height
        )
        
        # 6. 手势识别
        frame_info = {
            'width': pose_result.frame_width,
            'height': pose_result.frame_height
        }
        gesture_result = self.gesture_recognizer.recognize(
            pose_result.landmarks, frame_info
        )
        
        # 7. 处理控制指令
        self._process_control_command(gesture_result, distance_result)
        
        # 8. 可视化结果
        output_frame = self._create_visualization(
            processed_frame, pose_result, distance_result, gesture_result, quality
        )
        
        return output_frame
    
    def _process_control_command(self, gesture_result, distance_result):
        """处理控制指令"""
        current_time = time.time()
        
        # 只有高置信度的手势才作为指令
        if gesture_result.confidence > 0.8 and gesture_result.gesture != "none":
            # 避免指令重复触发
            if (gesture_result.gesture != self.current_command or 
                current_time - self.last_command_time > 2.0):
                
                self.current_command = gesture_result.gesture
                self.last_command_time = current_time
                
                # 执行指令处理
                self._execute_command(gesture_result, distance_result)
    
    def _execute_command(self, gesture_result, distance_result):
        """执行控制指令"""
        gesture = gesture_result.gesture
        distance = distance_result.distance
        confidence = gesture_result.confidence
        
        print(f"\n🎯 检测到指令: {gesture.upper()}")
        print(f"   置信度: {confidence:.2f}")
        print(f"   距离: {distance:.2f}m")
        
        # 这里可以添加实际的无人机控制代码
        if gesture == "takeoff":
            print("   🚁 执行起飞指令")
            # drone.takeoff()
        elif gesture == "landing":
            print("   🛬 执行降落指令")
            # drone.land()
        elif gesture == "forward":
            print("   ⬆️ 执行前进指令")
            # drone.move_forward(speed=self._calculate_speed(distance))
        elif gesture == "left":
            print("   ⬅️ 执行左移指令")
            # drone.move_left(speed=self._calculate_speed(distance))
        elif gesture == "right":
            print("   ➡️ 执行右移指令")
            # drone.move_right(speed=self._calculate_speed(distance))
        elif gesture == "up":
            print("   ⬆️ 执行上升指令")
            # drone.move_up(speed=self._calculate_speed(distance))
        elif gesture == "down":
            print("   ⬇️ 执行下降指令")
            # drone.move_down(speed=self._calculate_speed(distance))
        elif gesture == "stop":
            print("   ⏹️ 执行停止指令")
            # drone.hover()
    
    def _calculate_speed(self, distance):
        """根据距离计算控制速度"""
        if distance < 2.0:
            return 0.3  # 慢速
        elif distance < 4.0:
            return 0.6  # 中速
        else:
            return 1.0  # 快速
    
    def _create_visualization(self, frame, pose_result, distance_result, gesture_result, quality):
        """创建综合可视化"""
        output = frame.copy()
        
        # 绘制姿势
        if pose_result.landmarks:
            output = self.pose_visualizer.draw_pose(output, pose_result, draw_info=False)
        
        # 绘制距离信息
        output = self.distance_visualizer.draw_distance_info(
            output, distance_result, pose_result.landmarks, pose_result.bbox
        )
        
        # 绘制手势信息
        output = self.gesture_visualizer.draw_gesture_info(
            output, gesture_result, pose_result.landmarks
        )
        
        # 绘制系统状态
        self._draw_system_status(output, quality)
        
        return output
    
    def _draw_system_status(self, image, quality):
        """绘制系统状态信息"""
        # FPS信息
        fps = self.camera_capture.get_fps()
        cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 图像质量
        quality_color = (0, 255, 0) if quality['valid'] else (0, 0, 255)
        cv2.putText(image, f"Quality: {quality['quality']}", (image.shape[1] - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        # 当前指令
        if self.current_command != "none":
            cv2.putText(image, f"Last CMD: {self.current_command.upper()}", 
                       (image.shape[1] - 200, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _draw_quality_warning(self, frame, quality):
        """绘制图像质量警告"""
        output = frame.copy()
        warning_text = f"图像质量差: {quality['reason']}"
        
        # 添加半透明背景
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        cv2.putText(output, warning_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output
    
    def _draw_no_person_warning(self, frame):
        """绘制未检测到人体警告"""
        output = frame.copy()
        warning_text = "未检测到人体"
        
        # 添加半透明背景
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        cv2.putText(output, warning_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output
    
    def get_system_statistics(self):
        """获取系统统计信息"""
        pose_stats = self.pose_detector.get_statistics()
        distance_stats = self.distance_estimator.get_statistics()
        gesture_stats = self.gesture_recognizer.get_statistics()
        
        return {
            "pose_detection": pose_stats,
            "distance_estimation": distance_stats,
            "gesture_recognition": gesture_stats,
            "camera_fps": self.camera_capture.get_fps()
        }

def main():
    """主函数"""
    print("=" * 50)
    print("  手势控制系统 - 综合演示")
    print("=" * 50)
    print("支持的手势:")
    print("  🙌 起飞: 双手高举过头")
    print("  👇 降落: 双手向下压")
    print("  👉 方向: 手指指向移动方向")
    print("  ✋ 停止: 双手胸前交叉")
    print("  📊 统计: 按 's' 查看系统统计")
    print("  🚪 退出: 按 'q' 退出程序")
    print("=" * 50)
    
    # 初始化系统
    system = GestureControlSystem(camera_id=0)
    
    if not system.start():
        return
    
    try:
        print("系统运行中，开始手势控制...")
        
        while system.is_running:
            # 处理帧
            output_frame = system.process_frame()
            
            if output_frame is not None:
                # 显示结果
                key = system.image_visualizer.show_image(output_frame, "手势控制系统")
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 显示统计信息
                    stats = system.get_system_statistics()
                    print("\n" + "=" * 30)
                    print("系统统计信息:")
                    print("=" * 30)
                    
                    print(f"摄像头FPS: {stats['camera_fps']:.1f}")
                    print(f"姿势检测成功率: {stats['pose_detection']['success_rate']:.1f}%")
                    print(f"距离估算成功率: {stats['distance_estimation']['success_rate']:.1f}%")
                    print(f"手势识别成功率: {stats['gesture_recognition']['success_rate']:.1f}%")
                    print(f"当前手势: {stats['gesture_recognition']['current_gesture']}")
                    print("=" * 30)
                elif key == ord('r'):
                    # 重置统计
                    system.gesture_recognizer.reset_statistics()
                    print("统计信息已重置")
            else:
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"系统错误: {e}")
    finally:
        system.stop()
        print("系统已安全退出")

if __name__ == "__main__":
    main()
