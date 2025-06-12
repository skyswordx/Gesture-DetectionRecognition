"""
完整手势控制无人机演示
整合所有模块，实现真正的手势控制无人机功能
"""

import cv2
import time
import numpy as np
import sys
import os
import logging
import threading
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    # 图像处理模块
    from image_processing.image_processor import CameraCapture, ImageProcessor, ImageQualityAssessment, ImageVisualizer
    # 姿势检测模块
    from pose_detection.pose_detector import PoseDetector, PoseVisualizer, PoseAnalyzer
    # 距离估算模块
    from distance_estimation.distance_estimator import DistanceEstimator, DistanceVisualizer
    # 手势识别模块
    from gesture_recognition.gesture_recognizer import GestureRecognizer, GestureVisualizer
    # 无人机接口模块
    from drone_interface.drone_interface import SimulatedDroneInterface, TelloDroneInterface, DroneControlManager, DroneState
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有模块都在正确的位置")
    sys.exit(1)

class CompleteGestureDroneSystem:
    """完整的手势控制无人机系统"""
    
    def __init__(self, use_real_drone: bool = False, camera_id: int = 0):
        """
        初始化系统
        
        Args:
            use_real_drone: 是否使用真实无人机（Tello），False为模拟模式
            camera_id: 摄像头ID
        """
        self.use_real_drone = use_real_drone
        print(f"初始化完整手势控制无人机系统 ({'真实无人机' if use_real_drone else '模拟模式'})")
        
        # 初始化各个模块
        self._init_vision_modules(camera_id)
        self._init_drone_modules()
        
        # 系统状态
        self.is_running = False
        self.system_ready = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0  # 手势指令冷却时间(秒)
        
        # 统计信息
        self.frame_count = 0
        self.successful_commands = 0
        self.start_time = time.time()
        
        print("✅ 手势控制无人机系统初始化完成")
    
    def _init_vision_modules(self, camera_id: int):
        """初始化视觉处理模块"""
        logger.info("初始化视觉处理模块...")
        
        # 图像处理
        self.camera_capture = CameraCapture(camera_id=camera_id, width=640, height=480)
        self.image_processor = ImageProcessor()
        self.quality_assessor = ImageQualityAssessment()
        self.image_visualizer = ImageVisualizer()
        
        # 姿势检测
        self.pose_detector = PoseDetector(model_complexity=1)
        self.pose_visualizer = PoseVisualizer()
        self.pose_analyzer = PoseAnalyzer()
        
        # 距离估算
        self.distance_estimator = DistanceEstimator()
        self.distance_visualizer = DistanceVisualizer()
        
        # 手势识别
        self.gesture_recognizer = GestureRecognizer()
        self.gesture_visualizer = GestureVisualizer()
        
        logger.info("✅ 视觉处理模块初始化完成")
    
    def _init_drone_modules(self):
        """初始化无人机模块"""
        logger.info("初始化无人机模块...")
        
        try:
            # 选择无人机接口
            if self.use_real_drone:
                logger.info("尝试连接真实无人机 (Tello)...")
                self.drone_interface = TelloDroneInterface()
            else:
                logger.info("使用模拟无人机接口...")
                self.drone_interface = SimulatedDroneInterface()
            
            # 创建控制管理器
            self.drone_manager = DroneControlManager(self.drone_interface)
            
            # 配置安全参数
            self.drone_manager.safety_enabled = True
            self.drone_manager.max_altitude = 2.5  # 限制高度2.5米
            self.drone_manager.min_battery = 15.0  # 最低电量15%
            
            logger.info("✅ 无人机模块初始化完成")
            
        except Exception as e:
            logger.error(f"无人机模块初始化失败: {e}")
            logger.info("将使用模拟模式...")
            self.drone_interface = SimulatedDroneInterface()
            self.drone_manager = DroneControlManager(self.drone_interface)
            self.use_real_drone = False
    
    def start_system(self) -> bool:
        """启动系统"""
        logger.info("启动手势控制无人机系统...")
        
        # 启动摄像头
        if not self.camera_capture.start():
            logger.error("❌ 摄像头启动失败")
            return False
        
        # 连接无人机
        if not self.drone_interface.connect():
            logger.error("❌ 无人机连接失败")
            return False
        
        self.is_running = True
        self.system_ready = True
        self.start_time = time.time()
        
        logger.info("✅ 系统启动成功")
        return True
    
    def stop_system(self):
        """停止系统"""
        logger.info("停止手势控制无人机系统...")
        
        self.is_running = False
        self.system_ready = False
        
        # 确保无人机安全降落
        try:
            status = self.drone_interface.get_status()
            if status.state in [DroneState.FLYING, DroneState.HOVERING]:
                logger.info("无人机仍在飞行，执行安全降落...")
                self.drone_interface.land()
        except:
            pass
        
        # 停止各模块
        self.camera_capture.stop()
        self.drone_interface.disconnect()
        self.image_visualizer.close_all()
        
        logger.info("✅ 系统已安全停止")
    
    def run_main_loop(self):
        """运行主循环"""
        if not self.start_system():
            return
        
        self._show_startup_info()
        
        try:
            while self.is_running:
                self.frame_count += 1
                
                # 处理一帧
                success = self._process_single_frame()
                
                if not success:
                    time.sleep(0.01)
                    continue
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        except Exception as e:
            logger.error(f"系统运行错误: {e}")
        finally:
            self.stop_system()
            self._show_final_statistics()
    
    def _process_single_frame(self) -> bool:
        """处理单帧图像"""
        # 1. 获取摄像头图像
        frame = self.camera_capture.get_frame()
        if frame is None:
            return False
        
        # 2. 图像预处理
        processed_frame = self.image_processor.preprocess(frame)
        
        # 3. 图像质量评估
        quality_result = self.quality_assessor.assess_quality(processed_frame)
        
        if not quality_result['valid']:
            # 显示质量警告
            warning_frame = self._create_quality_warning_display(processed_frame, quality_result)
            self.image_visualizer.show_image(warning_frame, "Gesture Drone Control - Quality Warning")
            return True
        
        # 4. 人体姿势检测
        pose_result = self.pose_detector.detect(processed_frame)
        
        if not pose_result.landmarks:
            # 显示未检测到人体
            no_person_frame = self._create_no_person_display(processed_frame)
            self.image_visualizer.show_image(no_person_frame, "Gesture Drone Control - No Person")
            return True
        
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
        
        # 7. 处理无人机控制指令
        self._process_drone_command(gesture_result, distance_result)
        
        # 8. 创建综合显示
        display_frame = self._create_complete_display(
            processed_frame, pose_result, distance_result, 
            gesture_result, quality_result
        )
        
        # 9. 显示结果
        self.image_visualizer.show_image(display_frame, "Complete Gesture Drone Control System")
        
        return True
    
    def _process_drone_command(self, gesture_result, distance_result):
        """处理无人机控制指令"""
        current_time = time.time()
        
        # 检查手势置信度和冷却时间
        if (gesture_result.confidence > 0.8 and 
            gesture_result.gesture != "none" and
            current_time - self.last_gesture_time > self.gesture_cooldown):
            
            # 执行指令
            success = self.drone_manager.execute_gesture_command(
                gesture_result.gesture,
                gesture_result.confidence,
                distance_result.distance
            )
            
            if success:
                self.successful_commands += 1
                self.last_gesture_time = current_time
                
                # 在控制台显示指令信息
                self._display_command_info(gesture_result, distance_result)
    
    def _display_command_info(self, gesture_result, distance_result):
        """显示指令信息"""
        gesture_name = {
            'takeoff': '🚁 起飞',
            'landing': '🛬 降落', 
            'forward': '⬆️ 前进',
            'left': '⬅️ 左移',
            'right': '➡️ 右移',
            'up': '🔼 上升',
            'down': '🔽 下降',
            'stop': '⏹️ 停止'
        }.get(gesture_result.gesture, f"🎮 {gesture_result.gesture}")
        
        print(f"\n{'='*50}")
        print(f"  {gesture_name}")
        print(f"{'='*50}")
        print(f"置信度: {gesture_result.confidence:.2f}")
        print(f"距离: {distance_result.distance:.2f}m")
        print(f"时间: {time.strftime('%H:%M:%S')}")
        print(f"{'='*50}")
    
    def _create_complete_display(self, frame, pose_result, distance_result, gesture_result, quality_result):
        """创建完整的显示界面"""
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
        
        # 绘制无人机状态
        self._draw_drone_status(output)
        
        # 绘制系统信息
        self._draw_system_info(output, quality_result)
        
        return output
    
    def _draw_drone_status(self, image):
        """绘制无人机状态信息"""
        try:
            drone_status = self.drone_interface.get_status()
            
            # 无人机状态面板
            panel_y = 10
            panel_x = 10
            
            # 状态背景
            status_color = {
                DroneState.DISCONNECTED: (0, 0, 255),    # 红色
                DroneState.CONNECTED: (0, 255, 255),      # 黄色
                DroneState.FLYING: (0, 255, 0),           # 绿色
                DroneState.HOVERING: (0, 255, 0),         # 绿色
                DroneState.LANDING: (0, 165, 255),        # 橙色
                DroneState.EMERGENCY: (255, 0, 255)       # 紫色
            }.get(drone_status.state, (128, 128, 128))
            
            # 绘制状态面板
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + 250, panel_y + 120), (40, 40, 40), -1)
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + 250, panel_y + 120), status_color, 2)
            
            # 状态文本
            text_color = (255, 255, 255)
            cv2.putText(image, f"Drone: {drone_status.state.value.upper()}", 
                       (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            cv2.putText(image, f"Battery: {drone_status.battery_level:.0f}%", 
                       (panel_x + 10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            cv2.putText(image, f"Altitude: {drone_status.altitude:.1f}m", 
                       (panel_x + 10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            cv2.putText(image, f"Mode: {'Real' if self.use_real_drone else 'Sim'}", 
                       (panel_x + 10, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
        except Exception as e:
            logger.error(f"绘制无人机状态失败: {e}")
    
    def _draw_system_info(self, image, quality_result):
        """绘制系统信息"""
        # 右上角信息面板
        panel_x = image.shape[1] - 200
        panel_y = 10
        
        # 系统信息背景
        cv2.rectangle(image, (panel_x, panel_y), (panel_x + 190, panel_y + 100), (40, 40, 40), -1)
        cv2.rectangle(image, (panel_x, panel_y), (panel_x + 190, panel_y + 100), (100, 100, 100), 1)
        
        # 系统信息文本
        text_color = (255, 255, 255)
        
        # FPS
        fps = self.camera_capture.get_fps()
        cv2.putText(image, f"FPS: {fps:.1f}", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # 帧计数
        cv2.putText(image, f"Frames: {self.frame_count}", (panel_x + 10, panel_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # 成功指令数
        cv2.putText(image, f"Commands: {self.successful_commands}", (panel_x + 10, panel_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # 质量状态
        quality_color = (0, 255, 0) if quality_result['valid'] else (0, 0, 255)
        cv2.putText(image, f"Quality: OK" if quality_result['valid'] else "Poor", 
                   (panel_x + 10, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        # 底部控制提示
        help_text = "Press 'q' to quit | Gesture to control drone"
        cv2.putText(image, help_text, (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def _create_quality_warning_display(self, frame, quality_result):
        """创建图像质量警告显示"""
        output = frame.copy()
        
        # 半透明红色覆盖
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        
        # 警告文本
        warning_text = f"图像质量差 - {quality_result['reason']}"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (output.shape[1] - text_size[0]) // 2
        text_y = output.shape[0] // 2
        
        cv2.putText(output, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(output, "请改善光照条件", (text_x, text_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output
    
    def _create_no_person_display(self, frame):
        """创建未检测到人体显示"""
        output = frame.copy()
        
        # 半透明橙色覆盖
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        
        # 提示文本
        warning_text = "未检测到人体"
        instruction_text = "请站在摄像头前"
        
        text_size1 = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_size2 = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        text_x1 = (output.shape[1] - text_size1[0]) // 2
        text_x2 = (output.shape[1] - text_size2[0]) // 2
        text_y = output.shape[0] // 2
        
        cv2.putText(output, warning_text, (text_x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        cv2.putText(output, instruction_text, (text_x2, text_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output
    
    def _show_startup_info(self):
        """显示启动信息"""
        print("\n" + "=" * 70)
        print("  完整手势控制无人机系统")
        print("=" * 70)
        print(f"无人机模式: {'真实硬件 (Tello)' if self.use_real_drone else '模拟模式'}")
        print("支持的手势指令:")
        print("  🙌 双手高举     -> 起飞")
        print("  👇 双手向下     -> 降落")
        print("  👉 右手前推     -> 前进")
        print("  👈 左手指左     -> 左移")
        print("  👉 右手指右     -> 右移")
        print("  ☝️ 双手上举     -> 上升")
        print("  👇 双手下压     -> 下降")
        print("  ✋ 双手交叉     -> 停止/悬停")
        print("=" * 70)
        print("安全提示:")
        print("  • 确保周围环境安全")
        print("  • 保持适当距离 (1-4米)")
        print("  • 手势清晰可见")
        print("  • 随时准备紧急停止")
        print("=" * 70)
        print("控制:")
        print("  • 按 'q' 退出系统")
        print("  • 手势指令有2秒冷却时间")
        print("=" * 70)
        print("系统运行中... 开始手势控制")
        print()
    
    def _show_final_statistics(self):
        """显示最终统计信息"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 50)
        print("  系统运行统计")
        print("=" * 50)
        print(f"运行时间: {elapsed_time:.1f}秒")
        print(f"处理帧数: {self.frame_count}")
        print(f"平均FPS: {self.frame_count/elapsed_time:.1f}")
        print(f"成功指令: {self.successful_commands}")
        print(f"指令成功率: {(self.successful_commands*100/max(1, self.frame_count)):.1f}%")
        
        # 无人机统计
        try:
            drone_stats = self.drone_manager.get_statistics()
            if drone_stats:
                print("\n无人机控制统计:")
                print(f"  总指令数: {drone_stats.get('total_commands', 0)}")
                print(f"  平均置信度: {drone_stats.get('average_confidence', 0):.2f}")
                
                gesture_dist = drone_stats.get('gesture_distribution', {})
                if gesture_dist:
                    print("  指令分布:")
                    for gesture, count in gesture_dist.items():
                        print(f"    {gesture}: {count}")
        except:
            pass
        
        print("=" * 50)
        print("感谢使用手势控制无人机系统！")
        print("=" * 50)

def main():
    """主函数"""
    print("手势控制无人机系统")
    print("选择运行模式:")
    print("1. 模拟模式 (推荐用于测试)")
    print("2. 真实无人机模式 (需要DJI Tello)")
    
    while True:
        try:
            choice = input("\n请选择 (1 或 2): ").strip()
            if choice == "1":
                use_real_drone = False
                break
            elif choice == "2":
                use_real_drone = True
                print("注意: 请确保DJI Tello已连接并且安装了djitellopy库")
                confirm = input("继续? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
            else:
                print("无效选择，请输入1或2")
        except KeyboardInterrupt:
            print("\n程序已取消")
            return
    
    # 创建并运行系统
    try:
        system = CompleteGestureDroneSystem(use_real_drone=use_real_drone)
        system.run_main_loop()
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        print(f"错误: {e}")
        print("请检查:")
        print("  1. 摄像头是否正常连接")
        print("  2. Python依赖是否完整安装")
        print("  3. 无人机连接状态 (如果使用真实无人机)")

if __name__ == "__main__":
    main()
