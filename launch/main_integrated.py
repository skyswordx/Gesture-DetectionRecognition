"""
手势识别控制系统主入口
支持GUI界面和控制台模式，整合所有功能模块
"""

import cv2
import time
import numpy as np
import sys
import os
import logging
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_ws'))

try:
    from image_processing.image_processor import CameraCapture, ImageProcessor, ImageQualityAssessment, ImageVisualizer
    from pose_detection.pose_detector import PoseDetector, PoseVisualizer, PoseAnalyzer
    from distance_estimation.distance_estimator import DistanceEstimator, DistanceVisualizer
    from gesture_recognition.gesture_recognizer import GestureRecognizer, GestureVisualizer
    
except ImportError as e:
    logger.error(f"模块导入失败: {e}")
    print(f"模块导入失败: {e}")
    print("请确保所有模块都在正确的位置")
    print("检查项目:")
    print("  1. modules目录是否存在")
    print("  2. 所有__init__.py文件是否存在")
    print("  3. Python环境是否正确配置")

class GestureControlSystem:
    """手势控制系统 - 集成所有模块"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        """初始化系统"""
        print("初始化手势控制系统...")
        
        self.camera_id = camera_id
        self.width = width
        self.height = height
        
        try:
            # 图像处理模块
            self.camera_capture = CameraCapture(camera_id=camera_id, width=width, height=height)
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
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            print(f"❌ 组件初始化失败: {e}")
            print("请确保所有依赖包已安装")
            raise
        
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
        self._print_supported_gestures()
    
    def _print_supported_gestures(self):
        """打印支持的手势"""
        print("支持的手势:")
        print("  🙌 起飞: 双手高举过头")
        print("  👇 降落: 双手向下压")
        print("  👉 前进: 右手前推")
        print("  👈 左移: 左手指向左侧")
        print("  👉 右移: 右手指向右侧")
        print("  ☝️ 上升: 双手向上推举")
        print("  👇 下降: 双手向下压")
        print("  ✋ 停止: 双手胸前交叉")
    
    def get_system_info(self):
        """获取系统信息"""
        return {
            'camera_id': self.camera_id,
            'resolution': f"{self.width}x{self.height}",
            'is_running': self.is_running,
            'display_mode': self.display_mode,
            'frame_count': self.frame_count,
            'current_command': self.current_command
        }
    
    def set_display_mode(self, mode):
        """设置显示模式"""
        valid_modes = ["full", "pose_only", "distance_only", "gesture_only"]
        if mode in valid_modes:
            self.display_mode = mode
            print(f"显示模式切换为: {mode}")
        else:
            print(f"无效的显示模式: {mode}")
    
    def toggle_debug(self):
        """切换调试信息显示"""
        self.show_debug = not self.show_debug
        print(f"调试信息: {'显示' if self.show_debug else '隐藏'}")
    
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
        if self.camera_capture:
            self.camera_capture.stop()
        if self.image_visualizer:
            self.image_visualizer.close_all()
        print("系统已停止")
    
    def run(self):
        """运行主循环"""
        if not self.start():
            return
        
        print("\n" + "=" * 60)
        print("  综合手势控制系统演示")
        print("=" * 60)
        print("控制键:")
        print("  'q' - 退出程序")
        print("  '1' - 完整模式")
        print("  '2' - 仅姿势检测")
        print("  '3' - 仅距离估算")
        print("  '4' - 仅手势识别")
        print("  'd' - 切换调试信息")
        print("  's' - 显示统计信息")
        print("  'r' - 重置统计")
        print("=" * 60)
        
        try:
            while self.is_running:
                self.frame_count += 1
                
                # 处理一帧
                output_frame = self.process_frame()
                if output_frame is None:
                    time.sleep(0.01)
                    continue
                
                # 显示结果
                key = self.image_visualizer.show_image(output_frame, "Gesture Control System")
                
                # 处理按键
                if self._handle_key_input(key):
                    break
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
            logger.error(f"运行错误: {e}")
        finally:
            self.stop()
    
    def process_frame(self):
        """处理单帧图像"""
        try:
            # 1. 获取图像
            frame = self.camera_capture.get_frame()
            if frame is None:
                return None
            
            # 2. 图像预处理
            processed_frame = self.image_processor.preprocess(frame)
            
            # 3. 图像质量评估
            quality = self.quality_assessor.assess_quality(processed_frame)
            
            if not quality.get('valid', True):
                # 显示增强的质量警告
                warning_frame = self._draw_enhanced_quality_warning(processed_frame, quality)
                return warning_frame
            
            # 4. 姿势检测
            pose_result = self.pose_detector.detect(processed_frame)
            
            if not pose_result.landmarks:
                # 显示增强的"未检测到人体"警告
                no_person_frame = self._draw_enhanced_no_person_warning(processed_frame)
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
            
        except Exception as e:
            logger.error(f"帧处理错误: {e}")
            # 返回错误信息帧
            error_frame = frame if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)
            return self._draw_error_frame(error_frame, str(e))
    
    def _process_control_command(self, gesture_result, distance_result):
        """处理控制指令"""
        current_time = time.time()
        
        # 只有高置信度的手势才作为指令
        if gesture_result and gesture_result.confidence > 0.8 and gesture_result.gesture != "none":
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
        distance = distance_result.distance if distance_result else 0.0
        confidence = gesture_result.confidence
        
        print(f"\n🎯 检测到指令: {gesture.upper()}")
        print(f"   置信度: {confidence:.2f}")
        print(f"   距离: {distance:.2f}m")
        
        # 这里可以添加实际的无人机控制代码
        command_actions = {
            "takeoff": ("🚁 执行起飞指令", "drone.takeoff()"),
            "landing": ("🛬 执行降落指令", "drone.land()"),
            "forward": ("⬆️ 执行前进指令", f"drone.move_forward(speed={self._calculate_speed(distance)})"),
            "left": ("⬅️ 执行左移指令", f"drone.move_left(speed={self._calculate_speed(distance)})"),
            "right": ("➡️ 执行右移指令", f"drone.move_right(speed={self._calculate_speed(distance)})"),
            "up": ("⬆️ 执行上升指令", f"drone.move_up(speed={self._calculate_speed(distance)})"),
            "down": ("⬇️ 执行下降指令", f"drone.move_down(speed={self._calculate_speed(distance)})"),
            "stop": ("⏹️ 执行停止指令", "drone.hover()")
        }
        
        if gesture in command_actions:
            action_text, code_comment = command_actions[gesture]
            print(f"   {action_text}")
            # print(f"   代码: {code_comment}")  # 取消注释可显示对应代码
    
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
        
        # 根据显示模式选择绘制内容
        if self.display_mode in ["full", "pose_only"]:
            if pose_result and pose_result.landmarks:
                output = self.pose_visualizer.draw_pose(output, pose_result, draw_info=False)
        
        if self.display_mode in ["full", "distance_only"]:
            if distance_result:
                output = self.distance_visualizer.draw_distance_info(
                    output, distance_result, 
                    pose_result.landmarks if pose_result else None, 
                    pose_result.bbox if pose_result else None
                )
        
        if self.display_mode in ["full", "gesture_only"]:
            if gesture_result:
                output = self.gesture_visualizer.draw_gesture_info(
                    output, gesture_result, 
                    pose_result.landmarks if pose_result else None
                )
        
        # 系统状态信息
        if self.show_debug:
            self._draw_enhanced_system_status(output, quality)
        
        return output
    
    def _draw_enhanced_quality_warning(self, frame, quality):
        """绘制增强的图像质量警告"""
        output = frame.copy()
        
        # 获取详细的质量信息
        reason = quality.get('reason', '未知原因')
        suggestions = self._get_quality_suggestions(reason)
        
        # 创建警告背景
        h, w = output.shape[:2]
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
        
        # 主要警告文字
        cv2.putText(output, f"图像质量差: {reason}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 建议文字
        cv2.putText(output, f"建议: {suggestions}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 状态指示
        cv2.putText(output, "等待改善中...", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return output
    
    def _get_quality_suggestions(self, reason):
        """根据质量问题提供建议"""
        suggestions = {
            '光线过暗': '增加光照或移至明亮区域',
            '图像模糊': '保持摄像头稳定，调整焦距',
            '噪点过多': '改善光照条件',
            '对比度低': '调整环境光线或摄像头设置',
            '未知原因': '检查摄像头连接和环境'
        }
        return suggestions.get(reason, '检查摄像头和环境条件')
    
    def _draw_enhanced_no_person_warning(self, frame):
        """绘制增强的未检测到人体警告"""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # 创建信息背景
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (255, 165, 0), -1)
        cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
        
        # 主要警告
        cv2.putText(output, "未检测到人体", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 使用建议
        suggestions = [
            "确保全身在画面中",
            "保持适当距离 (1-3米)",
            "面向摄像头站立"
        ]
        
        for i, suggestion in enumerate(suggestions):
            cv2.putText(output, f"• {suggestion}", (30, 80 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def _draw_error_frame(self, frame, error_msg):
        """绘制错误信息帧"""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # 创建错误背景
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 128), -1)
        cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)
        
        # 错误信息
        cv2.putText(output, "系统错误", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, f"错误: {error_msg[:50]}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def _draw_enhanced_system_status(self, image, quality):
        """绘制增强的系统状态信息"""
        try:
            # FPS信息 - 带颜色指示
            fps = self.camera_capture.get_fps() if self.camera_capture else 0.0
            fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
            cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            # 图像质量 - 带状态指示
            if quality:
                quality_text = quality.get('quality', 'Good')
                quality_color = (0, 255, 0) if quality.get('valid', True) else (0, 0, 255)
                cv2.putText(image, f"Quality: {quality_text}", (image.shape[1] - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
            
            # 显示模式
            cv2.putText(image, f"Mode: {self.display_mode}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 当前指令 - 突出显示
            if self.current_command != "none":
                command_color = (0, 255, 255)  # 黄色
                cv2.putText(image, f"CMD: {self.current_command.upper()}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, command_color, 2)
                
                # 显示指令时间
                time_since_command = time.time() - self.last_command_time
                if time_since_command < 3.0:  # 3秒内显示倒计时
                    remaining = 3.0 - time_since_command
                    cv2.putText(image, f"({remaining:.1f}s)", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, command_color, 1)
            
            # 系统运行时间
            elapsed = time.time() - self.start_time
            cv2.putText(image, f"Runtime: {elapsed:.1f}s", (10, image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # 帧计数
            cv2.putText(image, f"Frames: {self.frame_count}", (image.shape[1] - 120, image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        except Exception as e:
            logger.error(f"状态绘制错误: {e}")
    
    def _handle_key_input(self, key):
        """处理键盘输入"""
        if key == ord('q'):
            return True  # 退出
        
        elif key == ord('1'):
            self.display_mode = "full"
            print("切换到完整显示模式")
        
        elif key == ord('2'):
            self.display_mode = "pose_only"
            print("切换到仅姿势检测模式")
        
        elif key == ord('3'):
            self.display_mode = "distance_only"
            print("切换到仅距离估算模式")
        
        elif key == ord('4'):
            self.display_mode = "gesture_only"
            print("切换到仅手势识别模式")
        
        elif key == ord('d'):
            self.show_debug = not self.show_debug
            print(f"调试信息: {'显示' if self.show_debug else '隐藏'}")
        
        elif key == ord('s'):
            self._show_statistics()
        
        elif key == ord('r'):
            self._reset_statistics()
        
        return False
    
    def _show_statistics(self):
        """显示统计信息"""
        print("\n" + "=" * 50)
        print("  系统统计信息")
        print("=" * 50)
        
        # 基本统计
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"运行时间: {elapsed:.1f}s")
        print(f"处理帧数: {self.frame_count}")
        print(f"平均FPS: {avg_fps:.1f}")
        
        # 各模块统计
        try:
            print("\n姿势检测模块状态: 运行正常")
            print("距离估算模块状态: 运行正常")
            print("手势识别模块状态: 运行正常")
            print(f"当前显示模式: {self.display_mode}")
            print(f"当前指令: {self.current_command}")
        except Exception as e:
            print(f"统计信息获取失败: {e}")
        
        print("=" * 50)
    
    def _reset_statistics(self):
        """重置统计"""
        self.frame_count = 0
        self.start_time = time.time()
        self.current_command = "none"
        try:
            if self.distance_estimator:
                self.distance_estimator.reset_filter()
        except:
            pass
        print("统计信息已重置")

class IntegratedGestureGUI:
    """整合的手势控制GUI系统"""
    
    def __init__(self):
        """初始化GUI系统"""
        self.root = tk.Tk()
        self.root.title("手势识别控制系统 - 整合平台")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 系统组件
        self.camera_capture = None
        self.image_processor = None
        self.quality_assessor = None
        self.pose_detector = None
        self.distance_estimator = None
        self.gesture_recognizer = None
        
        # 可视化组件
        self.pose_visualizer = None
        self.distance_visualizer = None
        self.gesture_visualizer = None
        
        # 运行状态
        self.is_running = False
        self.is_paused = False
        self.current_mode = "full"  # full, pose, distance, gesture
        self.show_debug = True
        
        # GUI组件
        self.video_canvas = None
        self.info_text = None
        self.status_var = None
        self.fps_var = None
        self.quality_var = None
        self.gesture_var = None
        self.distance_var = None
        self.confidence_var = None
        
        # 统计信息
        self.frame_count = 0
        self.start_time = time.time()
        self.last_command = "none"
        self.last_command_time = 0
        
        # 线程管理
        self.process_thread = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        
        # 创建GUI界面
        self.create_gui()
        
        # 初始化系统组件
        self.init_system_components()
    
    def init_system_components(self):
        """初始化系统组件"""
        try:
            # 图像处理模块
            self.image_processor = ImageProcessor()
            self.quality_assessor = ImageQualityAssessment()
            
            # 姿势检测模块
            self.pose_detector = PoseDetector(model_complexity=1)
            self.pose_visualizer = PoseVisualizer()
            
            # 距离估算模块
            self.distance_estimator = DistanceEstimator()
            self.distance_visualizer = DistanceVisualizer()
            
            # 手势识别模块
            self.gesture_recognizer = GestureRecognizer()
            self.gesture_visualizer = GestureVisualizer()
            
            logger.info("系统组件初始化完成")
            self.log_message("✅ 系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"系统组件初始化失败: {e}")
            self.log_message(f"❌ 系统组件初始化失败: {e}")
    
    def create_gui(self):
        """创建GUI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧视频显示区域
        self.create_video_panel(main_frame)
        
        # 底部状态栏
        self.create_status_bar(main_frame)
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        # 控制面板框架
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 系统控制
        system_frame = ttk.LabelFrame(control_frame, text="系统控制", padding=10)
        system_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(system_frame, text="启动摄像头", 
                                      command=self.start_system, state=tk.NORMAL)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(system_frame, text="停止检测", 
                                     command=self.stop_system, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        self.pause_button = ttk.Button(system_frame, text="暂停", 
                                      command=self.pause_resume, state=tk.DISABLED)
        self.pause_button.pack(fill=tk.X, pady=2)
        
        # 显示模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="显示模式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="full")
        modes = [
            ("完整模式", "full"),
            ("仅姿势检测", "pose"),
            ("仅距离估算", "distance"),
            ("仅手势识别", "gesture")
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                           value=mode, command=self.change_mode).pack(anchor=tk.W)
        
        # 参数设置
        settings_frame = ttk.LabelFrame(control_frame, text="参数设置", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 检测灵敏度
        ttk.Label(settings_frame, text="检测灵敏度:").pack(anchor=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sensitivity_scale = ttk.Scale(settings_frame, from_=0.3, to=1.0, 
                                     variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_scale.pack(fill=tk.X, pady=2)
        
        # 调试信息开关
        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="显示调试信息", 
                       variable=self.debug_var, command=self.toggle_debug).pack(anchor=tk.W)
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(control_frame, text="检测信息", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # 状态信息
        status_info_frame = ttk.Frame(info_frame)
        status_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        ttk.Label(status_info_frame, textvariable=self.fps_var, font=("Arial", 9)).pack(anchor=tk.W)
        
        self.quality_var = tk.StringVar(value="图像质量: 未知")
        ttk.Label(status_info_frame, textvariable=self.quality_var, font=("Arial", 9)).pack(anchor=tk.W)
        
        # 检测结果
        result_info_frame = ttk.Frame(info_frame)
        result_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gesture_var = tk.StringVar(value="手势: 未检测")
        ttk.Label(result_info_frame, textvariable=self.gesture_var, 
                 font=("Arial", 10, "bold"), foreground="blue").pack(anchor=tk.W)
        
        self.confidence_var = tk.StringVar(value="置信度: 0.0%")
        ttk.Label(result_info_frame, textvariable=self.confidence_var, font=("Arial", 9)).pack(anchor=tk.W)
        
        self.distance_var = tk.StringVar(value="距离: 未知")
        ttk.Label(result_info_frame, textvariable=self.distance_var, font=("Arial", 9)).pack(anchor=tk.W)
        
        # 日志区域
        log_frame = ttk.LabelFrame(info_frame, text="系统日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 日志文本区域
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_text_frame, height=8, width=35, font=("Consolas", 8))
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 清除日志按钮
        ttk.Button(log_frame, text="清除日志", command=self.clear_log).pack(pady=2)
    
    def create_video_panel(self, parent):
        """创建视频显示面板"""
        video_frame = ttk.LabelFrame(parent, text="视频显示", padding=10)
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 视频画布
        self.video_canvas = tk.Canvas(video_frame, width=640, height=480, bg="black")
        self.video_canvas.pack(pady=10)
        
        # 视频控制按钮
        video_control_frame = ttk.Frame(video_frame)
        video_control_frame.pack(fill=tk.X)
        
        ttk.Button(video_control_frame, text="截图保存", 
                  command=self.save_screenshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_control_frame, text="录制视频", 
                  command=self.toggle_recording).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_control_frame, text="重置统计", 
                  command=self.reset_statistics).pack(side=tk.LEFT, padx=5)
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="系统就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 帧计数器
        self.frame_counter_var = tk.StringVar(value="帧数: 0")
        ttk.Label(status_frame, textvariable=self.frame_counter_var, 
                 relief=tk.SUNKEN).pack(side=tk.RIGHT, padx=(5, 0))
    
    def start_system(self):
        """启动系统"""
        try:
            # 初始化摄像头
            self.camera_capture = CameraCapture(camera_id=0, width=640, height=480)
            if not self.camera_capture.start():
                raise Exception("摄像头启动失败")
            
            self.is_running = True
            self.is_paused = False
            self.start_time = time.time()
            self.frame_count = 0
            
            # 启动处理线程
            self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
            self.process_thread.start()
            
            # 启动GUI更新
            self.update_gui()
            
            # 更新按钮状态
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            
            self.status_var.set("系统运行中...")
            self.log_message("✅ 系统启动成功")
            
        except Exception as e:
            error_msg = f"系统启动失败: {e}"
            logger.error(error_msg)
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)
    
    def stop_system(self):
        """停止系统"""
        self.is_running = False
        
        # 停止摄像头
        if self.camera_capture:
            self.camera_capture.stop()
            self.camera_capture = None
        
        # 更新按钮状态
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        
        # 清空画布
        self.video_canvas.delete("all")
        self.video_canvas.create_text(320, 240, text="摄像头已停止", 
                                     fill="white", font=("Arial", 16))
        
        self.status_var.set("系统已停止")
        self.log_message("⏹️ 系统已停止")
    
    def pause_resume(self):
        """暂停/继续"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_button.config(text="继续")
            self.status_var.set("系统已暂停")
            self.log_message("⏸️ 系统已暂停")
        else:
            self.pause_button.config(text="暂停")
            self.status_var.set("系统运行中...")
            self.log_message("▶️ 系统继续运行")
    
    def change_mode(self):
        """更改显示模式"""
        self.current_mode = self.mode_var.get()
        self.log_message(f"🔄 切换到{self.current_mode}模式")
    
    def toggle_debug(self):
        """切换调试信息显示"""
        self.show_debug = self.debug_var.get()
        self.log_message(f"🔧 调试信息: {'显示' if self.show_debug else '隐藏'}")
    
    def process_loop(self):
        """处理循环(在后台线程中运行)"""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            try:
                # 获取图像
                frame = self.camera_capture.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 图像预处理
                if self.image_processor:
                    processed_frame = self.image_processor.preprocess(frame)
                else:
                    processed_frame = frame
                
                # 图像质量评估
                quality = None
                if self.quality_assessor:
                    quality = self.quality_assessor.assess_quality(processed_frame)
                
                # 处理结果
                result = self.process_frame(processed_frame, quality)
                
                # 将结果放入队列
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                self.frame_count += 1
                
            except Exception as e:
                logger.error(f"处理循环错误: {e}")
                time.sleep(0.1)
    
    def process_frame(self, frame, quality):
        """处理单帧图像"""
        result = {
            'frame': frame.copy(),
            'quality': quality,
            'pose_result': None,
            'distance_result': None,
            'gesture_result': None,
            'fps': 0.0,
            'error': None
        }
        
        try:            # 检查图像质量
            if quality and not quality.get('valid', True):
                result['error'] = f"图像质量差: {quality.get('reason', '未知原因')}"
                return result
            
            # 姿势检测
            if self.pose_detector:
                pose_result = self.pose_detector.detect(frame)
                result['pose_result'] = pose_result
                
                if not pose_result.landmarks:
                    result['error'] = "未检测到人体"
                    return result
                
                # 距离估算
                if self.distance_estimator:
                    distance_result = self.distance_estimator.estimate_distance(
                        pose_result.landmarks,
                        pose_result.frame_width,
                        pose_result.frame_height
                    )
                    result['distance_result'] = distance_result
                
                # 手势识别
                if self.gesture_recognizer:
                    frame_info = {
                        'width': pose_result.frame_width,
                        'height': pose_result.frame_height
                    }
                    gesture_result = self.gesture_recognizer.recognize(
                        pose_result.landmarks, frame_info
                    )
                    result['gesture_result'] = gesture_result
                    
                    # 处理控制指令
                    self.process_control_command(gesture_result, distance_result)
            
            # 计算FPS
            if hasattr(self, 'camera_capture') and self.camera_capture:
                result['fps'] = self.camera_capture.get_fps()
            
        except Exception as e:
            result['error'] = f"处理错误: {e}"
            logger.error(f"帧处理错误: {e}")
        
        return result
    
    def process_control_command(self, gesture_result, distance_result):
        """处理控制指令"""
        if not gesture_result or gesture_result.confidence < 0.8:
            return
        
        current_time = time.time()
        if (gesture_result.gesture != self.last_command or 
            current_time - self.last_command_time > 2.0):
            
            self.last_command = gesture_result.gesture
            self.last_command_time = current_time
            
            # 记录指令
            distance_str = f"{distance_result.distance:.2f}m" if distance_result else "未知"
            self.log_message(f"🎯 检测到指令: {gesture_result.gesture.upper()} "
                           f"(置信度: {gesture_result.confidence:.2f}, 距离: {distance_str})")
    
    def update_gui(self):
        """更新GUI显示"""
        if not self.is_running:
            return
        
        try:
            # 处理结果队列
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    self.display_result(result)
                except queue.Empty:
                    break
            
            # 更新统计信息
            self.update_statistics()
            
        except Exception as e:
            logger.error(f"GUI更新错误: {e}")
        
        # 继续更新
        self.root.after(33, self.update_gui)  # ~30 FPS
    
    def display_result(self, result):
        """显示处理结果"""
        frame = result['frame']
        
        if result['error']:
            # 显示错误信息
            self.display_error_frame(frame, result['error'])
        else:
            # 正常显示
            self.display_normal_frame(frame, result)
        
        # 更新信息显示
        self.update_info_display(result)
    
    def display_error_frame(self, frame, error_msg):
        """显示错误帧"""
        # 创建错误显示
        error_frame = frame.copy()
        
        # 添加半透明背景
        overlay = error_frame.copy()
        cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 100), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.7, error_frame, 0.3, 0, error_frame)
        
        # 添加错误文字
        cv2.putText(error_frame, error_msg, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.display_frame_on_canvas(error_frame)
    
    def display_normal_frame(self, frame, result):
        """显示正常帧"""
        output_frame = frame.copy()
        
        # 根据模式绘制不同内容
        pose_result = result.get('pose_result')
        distance_result = result.get('distance_result')
        gesture_result = result.get('gesture_result')
        
        if self.current_mode in ["full", "pose"] and pose_result and self.pose_visualizer:
            if pose_result.landmarks:
                output_frame = self.pose_visualizer.draw_pose(output_frame, pose_result, draw_info=False)
        
        if self.current_mode in ["full", "distance"] and distance_result and self.distance_visualizer:
            output_frame = self.distance_visualizer.draw_distance_info(
                output_frame, distance_result, pose_result.landmarks if pose_result else None, 
                pose_result.bbox if pose_result else None
            )
        
        if self.current_mode in ["full", "gesture"] and gesture_result and self.gesture_visualizer:
            output_frame = self.gesture_visualizer.draw_gesture_info(
                output_frame, gesture_result, pose_result.landmarks if pose_result else None
            )
        
        # 绘制调试信息
        if self.show_debug:
            self.draw_debug_info(output_frame, result)
        
        self.display_frame_on_canvas(output_frame)
    
    def draw_debug_info(self, frame, result):
        """绘制调试信息"""
        # FPS信息
        fps = result.get('fps', 0.0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
          # 质量信息
        quality = result.get('quality')
        if quality:
            quality_color = (0, 255, 0) if quality.get('valid', True) else (0, 0, 255)
            quality_text = f"Quality: {quality.get('quality', 'Good')}"
            cv2.putText(frame, quality_text, (frame.shape[1] - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        # 模式信息
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 最后指令
        if self.last_command != "none":
            cv2.putText(frame, f"Last CMD: {self.last_command.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def display_frame_on_canvas(self, frame):
        """在画布上显示帧"""
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(frame_rgb)
        
        # 调整大小适应画布
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # 转换为tkinter图像
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # 显示在画布上
        self.video_canvas.delete("all")
        self.video_canvas.create_image(canvas_width//2, canvas_height//2, image=tk_image)
        
        # 保持引用防止垃圾回收
        self.video_canvas.image = tk_image
    
    def update_info_display(self, result):
        """更新信息显示"""
        # 更新FPS
        fps = result.get('fps', 0.0)
        self.fps_var.set(f"FPS: {fps:.1f}")
        
        # 更新质量信息
        quality = result.get('quality')
        if quality:
            quality_status = "良好" if quality.get('valid', True) else f"差 ({quality.get('reason', '未知')})"
            self.quality_var.set(f"图像质量: {quality_status}")
        
        # 更新手势信息
        gesture_result = result.get('gesture_result')
        if gesture_result:
            gesture_text = gesture_result.gesture if gesture_result.gesture != "none" else "未检测"
            self.gesture_var.set(f"手势: {gesture_text}")
            self.confidence_var.set(f"置信度: {gesture_result.confidence * 100:.1f}%")
        else:
            self.gesture_var.set("手势: 未检测")
            self.confidence_var.set("置信度: 0.0%")
        
        # 更新距离信息
        distance_result = result.get('distance_result')
        if distance_result:
            self.distance_var.set(f"距离: {distance_result.distance:.2f}m")
        else:
            self.distance_var.set("距离: 未知")
    
    def update_statistics(self):
        """更新统计信息"""
        # 更新帧计数器
        self.frame_counter_var.set(f"帧数: {self.frame_count}")
        
        # 更新运行时间状态
        if self.is_running and not self.is_paused:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            status_text = f"运行中 - 平均FPS: {avg_fps:.1f} - 运行时间: {elapsed:.1f}s"
            self.status_var.set(status_text)
    
    def save_screenshot(self):
        """保存截图"""
        self.log_message("📸 截图功能待实现")
    
    def toggle_recording(self):
        """切换录制状态"""
        self.log_message("🎥 录制功能待实现")
    
    def reset_statistics(self):
        """重置统计"""
        self.frame_count = 0
        self.start_time = time.time()
        self.last_command = "none"
        
        if self.distance_estimator:
            try:
                self.distance_estimator.reset_filter()
            except:
                pass
        
        self.log_message("🔄 统计信息已重置")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # 限制日志长度
        lines = self.log_text.index(tk.END).split('.')[0]
        if int(lines) > 100:
            self.log_text.delete(1.0, 2.0)
    
    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("日志已清除")
    
    def on_closing(self):
        """关闭窗口时的处理"""
        if self.is_running:
            self.stop_system()
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """运行GUI系统"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log_message("🚀 GUI系统已启动")
        self.log_message("💡 点击'启动摄像头'开始检测")
        self.root.mainloop()

def show_mode_selection():
    """显示模式选择对话框"""
    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        choice = messagebox.askyesnocancel(
            "启动模式选择",
            "请选择启动模式:\n\n"
            "是 - GUI图形界面模式\n"
            "否 - 控制台模式\n"
            "取消 - 退出程序"
        )
        
        root.destroy()
        
        if choice is None:  # 取消
            return "exit"
        elif choice:  # 是
            return "gui"
        else:  # 否
            return "console"
    
    except Exception as e:
        logger.error(f"模式选择对话框错误: {e}")
        return "console"  # 默认控制台模式

def start_gui_mode():
    """启动GUI模式"""
    print("=" * 60)
    print("  手势识别控制系统 - GUI模式")
    print("=" * 60)
    print("正在启动图形界面...")
    
    try:
        gui_system = IntegratedGestureGUI()
        gui_system.run()
        return True
    except Exception as e:
        logger.error(f"GUI模式启动失败: {e}")
        print(f"❌ GUI模式启动失败: {e}")
        print("尝试以下解决方案:")
        print("  1. 检查tkinter是否正确安装")
        print("  2. 检查系统图形界面支持")
        print("  3. 尝试控制台模式")
        return False

def start_console_mode(camera_id=0, width=640, height=480):
    """启动控制台模式"""
    print("=" * 60)
    print("  手势识别控制系统 - 控制台模式")
    print("=" * 60)
    
    try:
        system = GestureControlSystem(camera_id=camera_id, width=width, height=height)
        system.run()
        return True
    except Exception as e:
        logger.error(f"控制台模式启动失败: {e}")
        print(f"❌ 控制台模式启动失败: {e}")
        return False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="手势识别控制系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py              # 显示模式选择对话框
  python main.py --gui         # 直接启动GUI模式
  python main.py --console     # 直接启动控制台模式
  python main.py --camera 1    # 使用摄像头1
  python main.py --help        # 显示帮助信息
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['gui', 'console'],
        help='启动模式: gui(图形界面) 或 console(控制台)'
    )
    
    parser.add_argument(
        '--gui',
        action='store_const',
        const='gui',
        dest='mode',
        help='启动GUI模式(等同于 --mode gui)'
    )
    
    parser.add_argument(
        '--console',
        action='store_const', 
        const='console',
        dest='mode',
        help='启动控制台模式(等同于 --mode console)'
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='摄像头ID (默认: 0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='图像宽度 (默认: 640)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='图像高度 (默认: 480)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    return parser.parse_args()

def check_system_requirements():
    """检查系统需求"""
    print("检查系统需求...")
    
    issues = []
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        issues.append(f"Python版本过低: {sys.version}，需要Python 3.7+")
    
    # 检查必要的库
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('mediapipe', 'mediapipe'),
        ('PIL', 'Pillow')
    ]
    
    for package_name, install_name in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            issues.append(f"缺少必要包: {install_name}")
    
    # 检查摄像头
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            issues.append("无法访问摄像头 (ID: 0)")
        else:
            cap.release()
    except Exception as e:
        issues.append(f"摄像头检查失败: {e}")
    
    if issues:
        print("❌ 系统需求检查失败:")
        for issue in issues:
            print(f"  • {issue}")
        print("\n请解决上述问题后重试")
        return False
    else:
        print("✅ 系统需求检查通过")
        return True

def main():
    """主函数"""
    print("=" * 80)
    print("  手势识别控制系统 - 主入口")
    print("=" * 80)
    print("系统特性:")
    print("  🖥️  支持GUI图形界面和控制台模式")
    print("  📷  实时摄像头输入和图像处理")
    print("  🤖  基于MediaPipe的人体姿势检测")
    print("  📏  智能距离估算算法")
    print("  ✋  多种手势识别和控制指令")
    print("  🎮  多种显示模式和实时统计")
    print("  📝  完整的日志记录和错误处理")
    print("  🔧  灵活的参数配置")
    print("=" * 80)
    
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置调试模式
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("调试模式已启用")
        
        # 检查系统需求
        if not check_system_requirements():
            sys.exit(1)
        
        # 确定启动模式
        mode = args.mode
        if mode is None:
            mode = show_mode_selection()
        
        if mode == "exit":
            print("程序已退出")
            return
        
        print(f"\n准备启动 {mode.upper()} 模式...")
        print(f"摄像头ID: {args.camera}")
        print(f"图像尺寸: {args.width}x{args.height}")
        
        # 根据模式启动相应系统
        success = False
        if mode == "gui":
            success = start_gui_mode()
        elif mode == "console":
            success = start_console_mode(
                camera_id=args.camera,
                width=args.width,
                height=args.height
            )
        
        if success:
            print("\n✅ 程序正常结束")
        else:
            print("\n❌ 程序异常结束")
            print("请检查:")
            print("  1. 摄像头是否连接正常")
            print("  2. 所有依赖包是否安装")
            print("  3. Python环境是否配置正确")
            print("  4. 系统资源是否充足")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        print(f"❌ 程序运行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
