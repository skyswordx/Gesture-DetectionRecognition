"""
高级集成演示 - 展示模块的高级用法和性能优化
"""

import cv2
import time
import numpy as np
import sys
import os
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any
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

@dataclass
class FrameData:
    """帧数据结构"""
    frame: np.ndarray
    timestamp: float
    frame_id: int

@dataclass
class ProcessingResult:
    """处理结果结构"""
    frame_data: FrameData
    pose_result: Any
    distance_result: Any
    gesture_result: Any
    quality_result: Any

class AdvancedGestureSystem:
    """高级手势控制系统 - 支持多线程和性能优化"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化系统"""
        self.config = config or self._get_default_config()
        
        # 初始化模块
        self._init_modules()
        
        # 多线程组件
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.visualization_thread = None
        
        # 系统状态
        self.is_running = False
        self.frame_counter = 0
        self.performance_stats = {
            'fps': 0.0,
            'processing_time': 0.0,
            'detection_success_rate': 0.0
        }
        
        print("高级手势控制系统初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'camera': {
                'id': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'pose_detection': {
                'model_complexity': 1,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            'distance_estimation': {
                'use_kalman_filter': True,
                'shoulder_width_cm': 40.0
            },
            'gesture_recognition': {
                'confidence_threshold': 0.8,
                'temporal_smoothing': True
            },
            'performance': {
                'max_fps': 30,
                'enable_threading': True,
                'profiling': False
            }
        }
    
    def _init_modules(self):
        """初始化所有模块"""
        # 图像处理模块
        self.camera_capture = CameraCapture(
            camera_id=self.config['camera']['id'],
            width=self.config['camera']['width'],
            height=self.config['camera']['height']
        )
        self.image_processor = ImageProcessor()
        self.quality_assessor = ImageQualityAssessment()
        self.image_visualizer = ImageVisualizer()
        
        # 姿势检测模块
        self.pose_detector = PoseDetector(
            model_complexity=self.config['pose_detection']['model_complexity']
        )
        self.pose_visualizer = PoseVisualizer()
        self.pose_analyzer = PoseAnalyzer()
        
        # 距离估算模块
        self.distance_estimator = DistanceEstimator()
        
        # 手势识别模块
        self.gesture_recognizer = GestureRecognizer()
        
        # 可视化器
        self.distance_visualizer = DistanceVisualizer()
        self.gesture_visualizer = GestureVisualizer()
    
    def start(self) -> bool:
        """启动系统"""
        print("启动高级手势控制系统...")
        
        # 启动摄像头
        if not self.camera_capture.start():
            print("❌ 摄像头启动失败")
            return False
        
        self.is_running = True
        
        if self.config['performance']['enable_threading']:
            # 启动多线程处理
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            
            self.processing_thread.start()
            self.visualization_thread.start()
        
        print("✅ 系统启动成功")
        return True
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
        
        self.camera_capture.stop()
        self.image_visualizer.close_all()
        print("系统已停止")
    
    def run_single_threaded(self):
        """单线程运行模式"""
        print("运行单线程模式...")
        
        try:
            while self.is_running:
                # 捕获帧
                frame = self.camera_capture.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_data = FrameData(
                    frame=frame,
                    timestamp=time.time(),
                    frame_id=self.frame_counter
                )
                self.frame_counter += 1
                
                # 处理帧
                result = self._process_frame_data(frame_data)
                
                # 显示结果
                if result:
                    visualization = self._create_advanced_visualization(result)
                    key = self.image_visualizer.show_image(visualization, "Advanced Gesture System")
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._print_performance_stats()
                    elif key == ord('r'):
                        self._reset_performance_stats()
                        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.stop()
    
    def run_multi_threaded(self):
        """多线程运行模式"""
        print("运行多线程模式...")
        
        try:
            # 主线程负责捕获帧
            while self.is_running:
                frame = self.camera_capture.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_data = FrameData(
                    frame=frame,
                    timestamp=time.time(),
                    frame_id=self.frame_counter
                )
                self.frame_counter += 1
                
                # 将帧放入处理队列
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # 丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.stop()
    
    def _processing_loop(self):
        """处理循环线程"""
        while self.is_running:
            try:
                # 获取帧数据
                frame_data = self.frame_queue.get(timeout=0.1)
                
                # 处理帧
                result = self._process_frame_data(frame_data)
                
                if result:
                    # 将结果放入显示队列
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # 丢弃最旧的结果
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理循环错误: {e}")
    
    def _visualization_loop(self):
        """可视化循环线程"""
        while self.is_running:
            try:
                # 获取处理结果
                result = self.result_queue.get(timeout=0.1)
                
                # 创建可视化
                visualization = self._create_advanced_visualization(result)
                
                # 显示结果
                key = self.image_visualizer.show_image(visualization, "Advanced Gesture System")
                
                if key == ord('q'):
                    self.is_running = False
                    break
                elif key == ord('s'):
                    self._print_performance_stats()
                elif key == ord('r'):
                    self._reset_performance_stats()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"可视化循环错误: {e}")
    
    def _process_frame_data(self, frame_data: FrameData) -> Optional[ProcessingResult]:
        """处理帧数据"""
        start_time = time.time()
        
        try:
            # 图像预处理
            processed_frame = self.image_processor.preprocess(frame_data.frame)
            
            # 图像质量评估
            quality_result = self.quality_assessor.assess_quality(processed_frame)
            
            if not quality_result['valid']:
                return ProcessingResult(
                    frame_data=frame_data,
                    pose_result=None,
                    distance_result=None,
                    gesture_result=None,
                    quality_result=quality_result
                )
            
            # 姿势检测
            pose_result = self.pose_detector.detect(processed_frame)
            
            if not pose_result.landmarks:
                return ProcessingResult(
                    frame_data=frame_data,
                    pose_result=pose_result,
                    distance_result=None,
                    gesture_result=None,
                    quality_result=quality_result
                )
            
            # 距离估算
            distance_result = self.distance_estimator.estimate_distance(
                pose_result.landmarks,
                pose_result.frame_width,
                pose_result.frame_height
            )
            
            # 手势识别
            frame_info = {
                'width': pose_result.frame_width,
                'height': pose_result.frame_height
            }
            gesture_result = self.gesture_recognizer.recognize(
                pose_result.landmarks, frame_info
            )
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            return ProcessingResult(
                frame_data=frame_data,
                pose_result=pose_result,
                distance_result=distance_result,
                gesture_result=gesture_result,
                quality_result=quality_result
            )
            
        except Exception as e:
            logger.error(f"帧处理错误: {e}")
            return None
    
    def _create_advanced_visualization(self, result: ProcessingResult) -> np.ndarray:
        """创建高级可视化"""
        output = result.frame_data.frame.copy()
        
        # 绘制质量警告
        if not result.quality_result['valid']:
            return self._draw_quality_warning(output, result.quality_result)
        
        # 绘制无人体警告
        if not result.pose_result or not result.pose_result.landmarks:
            return self._draw_no_person_warning(output)
        
        # 绘制姿势
        if result.pose_result.landmarks:
            output = self.pose_visualizer.draw_pose(output, result.pose_result, draw_info=False)
        
        # 绘制距离信息
        if result.distance_result:
            output = self.distance_visualizer.draw_distance_info(
                output, result.distance_result, result.pose_result.landmarks, result.pose_result.bbox
            )
        
        # 绘制手势信息
        if result.gesture_result:
            output = self.gesture_visualizer.draw_gesture_info(
                output, result.gesture_result, result.pose_result.landmarks
            )
        
        # 绘制高级状态信息
        self._draw_advanced_status(output, result)
        
        return output
    
    def _draw_advanced_status(self, image: np.ndarray, result: ProcessingResult):
        """绘制高级状态信息"""
        # 性能信息
        fps = self.performance_stats['fps']
        processing_time = self.performance_stats['processing_time']
        
        # 系统信息面板
        panel_height = 120
        panel = np.zeros((panel_height, image.shape[1], 3), dtype=np.uint8)
        panel[:, :] = (40, 40, 40)  # 深灰色背景
        
        # 绘制性能信息
        text_color = (255, 255, 255)
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.putText(panel, f"Processing: {processing_time*1000:.1f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.putText(panel, f"Frame ID: {result.frame_data.frame_id}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # 系统状态
        status_color = (0, 255, 0) if result.quality_result['valid'] else (0, 0, 255)
        cv2.putText(panel, f"Status: {'Running' if result.quality_result['valid'] else 'Warning'}", 
                   (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 检测状态
        pose_status = "✓" if result.pose_result and result.pose_result.landmarks else "✗"
        distance_status = "✓" if result.distance_result else "✗"
        gesture_status = "✓" if result.gesture_result and result.gesture_result.gesture != "none" else "✗"
        
        cv2.putText(panel, f"Pose: {pose_status}  Distance: {distance_status}  Gesture: {gesture_status}", 
                   (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # 控制提示
        cv2.putText(panel, "Press 'q' to quit, 's' for stats, 'r' to reset", 
                   (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 将面板添加到图像底部
        output_with_panel = np.vstack([image, panel])
        
        # 更新原图像
        image[:] = output_with_panel[:image.shape[0], :]
    
    def _draw_quality_warning(self, frame: np.ndarray, quality_result: Dict) -> np.ndarray:
        """绘制质量警告"""
        output = frame.copy()
        warning_text = f"图像质量差: {quality_result['reason']}"
        
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (500, 80), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        cv2.putText(output, warning_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output
    
    def _draw_no_person_warning(self, frame: np.ndarray) -> np.ndarray:
        """绘制无人体警告"""
        output = frame.copy()
        warning_text = "未检测到人体"
        
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        cv2.putText(output, warning_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return output
    
    def _update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        self.performance_stats['processing_time'] = processing_time
        
        # 计算FPS（简单移动平均）
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        alpha = 0.1  # 平滑因子
        self.performance_stats['fps'] = (
            alpha * current_fps + (1 - alpha) * self.performance_stats['fps']
        )
    
    def _print_performance_stats(self):
        """打印性能统计"""
        print("\n" + "=" * 60)
        print("  高级系统性能统计")
        print("=" * 60)
        print(f"当前FPS: {self.performance_stats['fps']:.2f}")
        print(f"处理时间: {self.performance_stats['processing_time']*1000:.2f}ms")
        print(f"总处理帧数: {self.frame_counter}")
        print(f"队列状态: Frame={self.frame_queue.qsize()}, Result={self.result_queue.qsize()}")
        print(f"多线程模式: {'启用' if self.config['performance']['enable_threading'] else '禁用'}")
        print("=" * 60)
    
    def _reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'fps': 0.0,
            'processing_time': 0.0,
            'detection_success_rate': 0.0
        }
        self.frame_counter = 0
        print("性能统计已重置")
    
    def run(self):
        """运行系统"""
        if not self.start():
            return
        
        if self.config['performance']['enable_threading']:
            self.run_multi_threaded()
        else:
            self.run_single_threaded()

def main():
    """主函数"""
    print("=" * 70)
    print("  高级手势控制系统演示")
    print("=" * 70)
    print("特性:")
    print("  🚀 多线程处理架构")
    print("  📊 实时性能监控")
    print("  🎨 高级可视化界面")
    print("  ⚙️ 可配置参数")
    print("  🔧 错误处理和恢复")
    print("  📈 详细统计信息")
    print("=" * 70)
    
    # 高级配置
    config = {
        'camera': {
            'id': 0,
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'pose_detection': {
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        },
        'distance_estimation': {
            'use_kalman_filter': True,
            'shoulder_width_cm': 40.0
        },
        'gesture_recognition': {
            'confidence_threshold': 0.8,
            'temporal_smoothing': True
        },
        'performance': {
            'max_fps': 30,
            'enable_threading': True,
            'profiling': True
        }
    }
    
    try:
        system = AdvancedGestureSystem(config)
        system.run()
    except Exception as e:
        print(f"系统启动失败: {e}")
        logger.exception(e)

if __name__ == "__main__":
    main()
