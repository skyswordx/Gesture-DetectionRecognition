"""
手势识别控制系统主入口 - 已修改为图片集批处理模式

功能：遍历指定文件夹下的图片，对每张图片进行骨骼点检测、距离估算和手势识别。
"""

import cv2
import time
import numpy as np
import sys
import os
import logging
import argparse
import threading
import queue
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 模块路径添加 (保持不变) ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_ws'))

try:
    # 假设这些模块已存在且能被导入
    from image_processing.image_processor import ImageProcessor, ImageQualityAssessment, ImageVisualizer
    from pose_detection.pose_detector import PoseDetector, PoseVisualizer, PoseAnalyzer
    from distance_estimation.distance_estimator import DistanceEstimator, DistanceVisualizer
    from gesture_recognition.gesture_recognizer import GestureRecognizer, GestureVisualizer
    
    # 假设 CameraCapture 已经不再需要导入
    
except ImportError as e:
    logger.error(f"模块导入失败: {e}")
    print(f"模块导入失败: {e}")
    print("请确保所有模块都在正确的位置")
    sys.exit(1)


# =================================================================
# 【新增/替换模块】：ImageSetCapture - 用于读取图片集 (保持不变)
# =================================================================

class ImageSetCapture:
    """
    图片集捕获器：用于读取文件夹中的所有图片文件，替代CameraCapture
    """

    def __init__(self, folder_path: str, width: int, height: int, supported_exts: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
        self.folder_path = folder_path
        self.supported_exts = supported_exts
        self.width = width
        self.height = height
        self.image_files = self._get_image_files()
        self.current_index = 0
        self.num_frames = len(self.image_files)
        self.last_frame_time = time.time()
        self.last_fps = 0.0
        logger.info(f"图片集加载完成，共找到 {self.num_frames} 张图片。")

    def _get_image_files(self):
        """扫描文件夹，获取所有支持的图片路径并按名称排序"""
        if not os.path.isdir(self.folder_path):
            logger.error(f"图片文件夹不存在: {self.folder_path}")
            return []
        
        # 按文件名排序，确保处理顺序一致
        files = []
        for f in sorted(os.listdir(self.folder_path)):
            if f.lower().endswith(self.supported_exts):
                files.append(os.path.join(self.folder_path, f))
        
        return files

    def start(self):
        """开始读取 - 检查文件数量"""
        if not self.image_files:
            logger.error("在指定文件夹中未找到任何图片。")
            return False
        self.current_index = 0
        self.last_frame_time = time.time()
        print(f"[OK] 开始处理图片集: {self.folder_path}，总计 {self.num_frames} 张图片。")
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """
        读取下一张图片作为一帧
        :return: OpenCV格式的图像 (BGR) 或 None (表示处理完毕)
        """
        if self.current_index >= self.num_frames:
            # 达到图片集末尾，停止
            return None

        # 计算模拟的FPS
        current_time = time.time()
        if self.current_index > 0:
            self.last_fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time

        # 读取图片
        image_path = self.image_files[self.current_index]
        print(f"[IMG] 正在处理 [{self.current_index + 1}/{self.num_frames}]: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        
        # 【修改】：保持宽高比，只限制最大尺寸
        if frame is not None:
            h, w = frame.shape[:2]
            max_dim = max(self.width, self.height)
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

        # 增加索引
        self.current_index += 1
        
        return frame

    def get_fps(self) -> float:
        """返回当前处理速度（模拟FPS）"""
        return self.last_fps

    def stop(self):
        """停止读取（重置索引）"""
        self.current_index = 0
        pass

# =================================================================
# 【核心修改】：GestureControlSystem
# =================================================================

class GestureControlSystem:
    """手势控制系统 - 集成所有模块"""
    
    # 【修改点 1】：接收 folder_path 代替 camera_id
    def __init__(self, folder_path: str, width=640, height=480):
        """初始化系统"""
        print("初始化手势控制系统 (图片集模式)...")
        
        self.folder_path = folder_path
        self.width = width
        self.height = height
        
        try:
            # 图像处理模块
            # 【修改点 2】：使用 ImageSetCapture 替换 CameraCapture
            self.camera_capture = ImageSetCapture(folder_path=folder_path, width=width, height=height)
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
            print(f"[ERROR] 组件初始化失败: {e}")
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
        """打印支持的手势 (从原始代码复制过来)"""
        print("支持的手势:")
        print("  [TAKEOFF] 起飞: 双手高举过头")
        print("  [LANDING] 降落: 双手向下压")
        print("  [FORWARD] 前进: 右手前推")
        print("  [LEFT]    左移: 左手指向左侧")
        print("  [RIGHT]   右移: 右手指向右侧")
        print("  [UP]      上升: 双手向上推举")
        print("  [DOWN]    下降: 双手向下压")
        print("  [STOP]    停止: 双手胸前交叉")
    
    # =======================================================
    # 【核心修改区：修复 'GestureResult' object has no attribute 'get'】
    # =======================================================
    def _process_control_command(self, gesture_result: Any, distance_result: Any):
        """
        处理手势识别结果，确定控制指令
        【修复：使用 getattr 兼容 GestureResult 和 DistanceResult 对象】
        """
        current_time = time.time()
        
        # 1. 尝试从 GestureResult 对象中获取 'gesture' 属性 (推荐)
        command_name = getattr(gesture_result, 'gesture', None) 
        
        if command_name is None:
            # 2. 如果不是对象或没有 'gesture' 属性，则尝试从字典中获取 'command' 键 (兼容旧接口)
            if hasattr(gesture_result, 'get'):
                command_name = gesture_result.get('command', None)
        
        # 检查是否识别到了有效手势
        if command_name and command_name != "none":
            self.current_command = command_name
            self.last_command_time = current_time
            # 【此处是执行指令的核心逻辑，在图片集模式下，我们只打印】
            self._execute_command(self.current_command)
            return

        # 距离和姿势相关指令 (如果需要)
        # 示例：如果距离太近，强制停止
        # 【修复：使用 getattr 兼容 DistanceResult 对象】
        distance_value = getattr(distance_result, 'distance', None)
        if distance_value is None and hasattr(distance_result, 'get'):
            distance_value = distance_result.get('distance_cm', 999) / 100.0  # 转换为米
        
        if distance_value is not None and distance_value < 0.5:  # 0.5米
            if current_time - self.last_command_time > 2: # 避免频繁触发
                self.current_command = "forced_stop (too close)"
                self.last_command_time = current_time
                self._execute_command(self.current_command)
                return

        # 如果长时间没有新的有效指令，命令恢复为 none
        if current_time - self.last_command_time > 1.0: # 1秒不执行任何指令
            self.current_command = "none"

    def _create_visualization(self, processed_frame, pose_result, distance_result, gesture_result, quality):
        """综合所有模块的可视化结果"""
        
        # 1. 绘制姿势和骨骼 (使用 draw_pose 方法)
        output_frame = self.pose_visualizer.draw_pose(
            processed_frame, 
            pose_result,
            draw_landmarks=True,
            draw_bbox=True,
            draw_info=True
        )
        
        # 2. 绘制距离信息
        output_frame = self.distance_visualizer.draw_distance_info(
            output_frame, 
            distance_result, 
            pose_result.landmarks,
            pose_result.bbox
        )
        
        # 3. 绘制手势信息 (使用 draw_gesture_info 方法)
        output_frame = self.gesture_visualizer.draw_gesture_info(
            output_frame, 
            gesture_result, 
            pose_result.landmarks
        )
        
        # 4. 绘制FPS和系统信息
        output_frame = self._show_statistics(output_frame)
        
        # 5. 绘制质量/警告信息 (如果需要)
        if not quality.get('valid', True):
            # 这是一个补充，主要警告在 process_frame 中返回
            cv2.putText(output_frame, "LOW QUALITY WARNING", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return output_frame

    def _show_statistics(self, frame: np.ndarray) -> np.ndarray:
        """在帧上显示FPS和系统信息"""
        fps = self.camera_capture.get_fps()
        info = f"FPS: {fps:.2f} | Command: {self.current_command.upper()}"
        
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def _draw_error_frame(self, frame: np.ndarray, error_message: str) -> np.ndarray:
        """绘制一个红色的错误框"""
        h, w, _ = frame.shape
        # 绘制红色背景
        cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 200), -1)
        text = f"ERROR: {error_message[:40]}..."
        cv2.putText(frame, text, (10, h-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
        
    def _draw_enhanced_quality_warning(self, frame: np.ndarray, quality: Dict[str, Any]) -> np.ndarray:
        """绘制增强的图像质量警告"""
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h-80), (w, h), (0, 165, 255), -1) # 橙色背景
        
        warning_msg = f"Low Quality: {quality.get('reason', 'Unknown').upper()}"
        cv2.putText(frame, warning_msg, (10, h-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Please improve lighting/focus.", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def _draw_enhanced_no_person_warning(self, frame: np.ndarray) -> np.ndarray:
        """绘制增强的未检测到人体的警告"""
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), -1) # 红色背景
        
        cv2.putText(frame, "NO PERSON DETECTED", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
    
    

    def _execute_command(self, command: str):
        """执行具体的控制命令 (在图片集模式中，只做打印)"""
        print(f"\n[COMMAND EXECUTED] -> {command.upper()}")
        # 实际应用中，这里会调用无人机API、机器人控制等
        pass

    def start(self):
        """启动系统 (现在是启动图片集读取)"""
        print("启动图片集读取...")
        # 【修改点 3】：调用 ImageSetCapture 的 start
        if not self.camera_capture.start():
            print("[ERROR] 图片集启动失败，请检查路径和文件。")
            return False
        
        self.is_running = True
        print("[OK] 系统启动成功")
        return True

    def stop(self):
        """停止系统"""
        self.is_running = False
        if self.camera_capture:
            self.camera_capture.stop()
        if self.image_visualizer:
            # 由于是图片集处理，不需要 waitKey，但可以保留 close_all 习惯
            self.image_visualizer.close_all() 
        print("系统已停止")

    def run(self):
        """运行主循环 - 遍历图片集"""
        if not self.start():
            return
            
        print("\n" + "=" * 60)
        print(f"  综合手势控制系统演示 (图片集模式: {self.camera_capture.num_frames} 张图片)")
        print("=" * 60)
        print("控制键:")
        print("  'q' 或 'ESC' - 退出程序")
        print("  <任何其他键> - 处理下一张图片") # 【修改点 4】：等待用户按键处理下一张图片
        print("=" * 60)
            
        try:
            while self.is_running:
                # 处理一帧
                output_frame = self.process_frame()
                
                # 【修改点 5】：如果 output_frame 为 None，表示图片集已处理完毕
                if output_frame is None:
                    print("\n[DONE] 所有图片已处理完毕，系统退出。")
                    break
                
                self.frame_count += 1
                
                # 显示结果
                # 在图片集模式下，使用 waitKey(0) 来实现按任意键继续
                key = self.image_visualizer.show_image(output_frame, "Gesture Control System", wait_key=0)
                
                # 处理按键
                if key == ord('q') or key == 27: # q 或 ESC 退出
                    break
                # 任何其他键都继续处理下一张图片 
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
            logger.error(f"运行错误: {e}")
        finally:
            self.stop()

    def process_frame(self):
        """处理单帧图像 (逻辑不变，但现在输入来自文件)"""
        frame = None
        try:
            # 1. 获取图像 - ImageSetCapture 会读取下一张图片
            frame = self.camera_capture.get_frame()
            if frame is None:
                return None
            
            # 【图片集模式关键修改】：每张图片独立识别，重置手势识别器状态
            # 避免时序一致性检查导致的误识别
            self.gesture_recognizer.reset_statistics()
            
            # 2. 图像预处理 (图片集模式下不做镜像翻转，直接使用原图)
            # 注意：preprocess 会做镜像翻转，对于图片集模式不适用
            processed_frame = frame.copy()
            
            # 3. 图像质量评估
            quality = self.quality_assessor.assess_quality(processed_frame)
            
            if not quality.get('valid', True):
                warning_frame = self._draw_enhanced_quality_warning(processed_frame, quality)
                return warning_frame
            
            # 4. 姿势检测
            pose_result = self.pose_detector.detect(processed_frame)
            
            if not pose_result.landmarks:
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
            # gesture_result 是 GestureResult 对象
            gesture_result = self.gesture_recognizer.recognize(
                pose_result.landmarks, frame_info
            )
            
            # 7. 处理控制指令 (这里修复了调用问题)
            self._process_control_command(gesture_result, distance_result)
            
            # 8. 可视化结果
            output_frame = self._create_visualization(
                processed_frame, pose_result, distance_result, gesture_result, quality
            )
            
            return output_frame
            
        except Exception as e:
            logger.error(f"帧处理错误: {e}")
            # 确保在出错时，即使 frame 是 None 也能返回一个黑色的错误框
            error_frame = frame if frame is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return self._draw_error_frame(error_frame, str(e))

    def _handle_key_input(self, key):
        """处理键盘输入 - 仅用于模式切换，不控制循环"""
        if key == ord('q'):
            return True
        # ... (其他键处理) ...
        return False

# =================================================================
# 【主执行块】：使用命令行参数传入图片文件夹路径 (保持不变)
# =================================================================

if __name__ == '__main__':
    
    # 【修改点 6】：使用 argparse 接收文件夹路径
    parser = argparse.ArgumentParser(description="手势识别控制系统 - 图片集模式")
    parser.add_argument('--folder_path', type=str, required=True, 
                        help='包含待处理图片 (JPG/PNG/BMP) 的文件夹路径。')
    parser.add_argument('--width', type=int, default=640, help='处理图像宽度 (默认为 640)。')
    parser.add_argument('--height', type=int, default=480, help='处理图像高度 (默认为 480)。')
    args = parser.parse_args()
    
    try:
        # 【修改点 7】：使用 folder_path 实例化系统
        system = GestureControlSystem(
            folder_path=args.folder_path, 
            width=args.width, 
            height=args.height
        )
        system.run()
    except Exception as e:
        logger.critical(f"系统启动失败或发生致命错误: {e}")
        sys.exit(1)