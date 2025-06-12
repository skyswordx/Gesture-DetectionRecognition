"""
图像处理模块
提供摄像头采集、图像预处理、图像增强等基础功能
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Tuple, Callable
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraCapture:
    """摄像头采集类"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        初始化摄像头
        
        Args:
            camera_id: 摄像头ID
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        
        # 统计信息
        self.frame_count = 0
        self.actual_fps = 0
        self.last_fps_time = time.time()
        
    def start(self) -> bool:
        """启动摄像头采集"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_id}")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info(f"摄像头启动成功: {self.width}x{self.height}@{self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"摄像头启动失败: {e}")
            return False
    
    def stop(self):
        """停止摄像头采集"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("摄像头已停止")
    
    def _capture_loop(self):
        """摄像头采集循环"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("无法读取摄像头帧")
                continue
            
            # 更新FPS统计
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.actual_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # 添加到队列
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # 队列满时，移除最旧的帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """获取实际FPS"""
        return self.actual_fps

class ImageProcessor:
    """图像处理类"""
    
    def __init__(self):
        """初始化图像处理器"""
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.enable_noise_reduction = True
        self.enable_enhancement = True
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        if image is None:
            return None
        
        processed = image.copy()
        
        # 镜像翻转
        processed = cv2.flip(processed, 1)
        
        # 亮度对比度调整
        if self.enable_enhancement:
            processed = self.adjust_brightness_contrast(processed)
        
        # 噪声减少
        if self.enable_noise_reduction:
            processed = self.reduce_noise(processed)
        
        return processed
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                 brightness: float = None, 
                                 contrast: float = None) -> np.ndarray:
        """
        调整亮度和对比度
        
        Args:
            image: 输入图像
            brightness: 亮度因子 (None使用默认值)
            contrast: 对比度因子 (None使用默认值)
            
        Returns:
            调整后的图像
        """
        if brightness is None:
            brightness = self.brightness_factor
        if contrast is None:
            contrast = self.contrast_factor
        
        # 自动亮度调整
        if brightness == 1.0:  # 自动模式
            brightness = self.calculate_auto_brightness(image)
        
        # 应用调整
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1.0) * 30)
        return adjusted
    
    def calculate_auto_brightness(self, image: np.ndarray) -> float:
        """计算自动亮度因子"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # 目标亮度为128
        target_brightness = 128
        if mean_brightness < 80:
            return 1.3  # 增加亮度
        elif mean_brightness > 180:
            return 0.8  # 降低亮度
        else:
            return 1.0  # 保持原样
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        降噪处理
        
        Args:
            image: 输入图像
            
        Returns:
            降噪后的图像
        """
        # 双边滤波 - 保持边缘的同时降噪
        denoised = cv2.bilateralFilter(image, 5, 80, 80)
        return denoised
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 对L通道进行CLAHE (对比度限制自适应直方图均衡)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def detect_blur(self, image: np.ndarray) -> float:
        """
        检测图像模糊程度
        
        Args:
            image: 输入图像
            
        Returns:
            模糊度分数 (越高越清晰)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def detect_brightness(self, image: np.ndarray) -> float:
        """
        检测图像亮度
        
        Args:
            image: 输入图像
            
        Returns:
            平均亮度值 (0-255)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def set_parameters(self, brightness: float = 1.0, contrast: float = 1.0, 
                      noise_reduction: bool = True, enhancement: bool = True):
        """
        设置处理参数
        
        Args:
            brightness: 亮度因子
            contrast: 对比度因子
            noise_reduction: 是否启用降噪
            enhancement: 是否启用增强
        """
        self.brightness_factor = brightness
        self.contrast_factor = contrast
        self.enable_noise_reduction = noise_reduction
        self.enable_enhancement = enhancement

class ImageQualityAssessment:
    """图像质量评估类"""
    
    def __init__(self):
        """初始化质量评估器"""
        self.blur_threshold = 100.0  # 模糊度阈值
        self.brightness_range = (50, 200)  # 亮度范围
        
    def assess_quality(self, image: np.ndarray) -> dict:
        """
        评估图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            质量评估结果字典
        """
        if image is None:
            return {"valid": False, "reason": "图像为空"}
        
        # 检测模糊度
        processor = ImageProcessor()
        blur_score = processor.detect_blur(image)
        is_sharp = blur_score > self.blur_threshold
        
        # 检测亮度
        brightness = processor.detect_brightness(image)
        is_bright_ok = self.brightness_range[0] <= brightness <= self.brightness_range[1]
        
        # 检测图像尺寸
        height, width = image.shape[:2]
        is_size_ok = width >= 320 and height >= 240
        
        # 综合评估
        overall_quality = "good" if all([is_sharp, is_bright_ok, is_size_ok]) else "poor"
        
        if not is_sharp:
            quality_reason = "图像模糊"
        elif not is_bright_ok:
            quality_reason = "亮度异常"
        elif not is_size_ok:
            quality_reason = "尺寸过小"
        else:
            quality_reason = "质量良好"
        
        return {
            "valid": overall_quality == "good",
            "quality": overall_quality,
            "reason": quality_reason,
            "blur_score": blur_score,
            "brightness": brightness,
            "is_sharp": is_sharp,
            "is_bright_ok": is_bright_ok,
            "is_size_ok": is_size_ok,
            "width": width,
            "height": height
        }

class ImageVisualizer:
    """图像可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.window_names = set()
        
    def show_image(self, image: np.ndarray, title: str = "Image", 
                  wait_key: int = 1) -> int:
        """
        显示图像
        
        Args:
            image: 图像
            title: 窗口标题
            wait_key: 等待按键时间(ms)
            
        Returns:
            按键值
        """
        if image is None:
            return -1
        
        cv2.imshow(title, image)
        self.window_names.add(title)
        return cv2.waitKey(wait_key) & 0xFF
    
    def show_comparison(self, original: np.ndarray, processed: np.ndarray, 
                       title: str = "Comparison"):
        """
        显示对比图像
        
        Args:
            original: 原始图像
            processed: 处理后图像
            title: 窗口标题
        """
        if original is None or processed is None:
            return
        
        # 调整图像尺寸一致
        height = max(original.shape[0], processed.shape[0])
        original_resized = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        processed_resized = cv2.resize(processed, (int(processed.shape[1] * height / processed.shape[0]), height))
        
        # 水平拼接
        comparison = np.hstack([original_resized, processed_resized])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Processed", (original_resized.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        
        self.show_image(comparison, title)
    
    def draw_info(self, image: np.ndarray, info: dict) -> np.ndarray:
        """
        在图像上绘制信息
        
        Args:
            image: 图像
            info: 信息字典
            
        Returns:
            绘制了信息的图像
        """
        if image is None:
            return None
        
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(result, text, (10, y_offset), font, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return result
    
    def close_all(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()
        self.window_names.clear()

def test_camera_capture():
    """测试摄像头采集功能"""
    print("测试摄像头采集...")
    
    capture = CameraCapture(camera_id=0)
    if not capture.start():
        print("摄像头启动失败")
        return
    
    print("摄像头启动成功，按 'q' 键退出")
    
    visualizer = ImageVisualizer()
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 显示FPS信息
                fps_info = {"FPS": f"{capture.get_fps():.1f}"}
                frame_with_info = visualizer.draw_info(frame, fps_info)
                
                key = visualizer.show_image(frame_with_info, "Camera Test")
                if key == ord('q'):
                    break
            else:
                time.sleep(0.01)
    
    finally:
        capture.stop()
        visualizer.close_all()

def test_image_processing():
    """测试图像处理功能"""
    print("测试图像处理...")
    
    capture = CameraCapture(camera_id=0)
    processor = ImageProcessor()
    quality_assessor = ImageQualityAssessment()
    visualizer = ImageVisualizer()
    
    if not capture.start():
        print("摄像头启动失败")
        return
    
    print("图像处理测试，按 'q' 退出, 'e' 切换增强, 'n' 切换降噪")
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 处理图像
                processed = processor.preprocess(frame)
                
                # 质量评估
                quality = quality_assessor.assess_quality(processed)
                
                # 显示对比
                visualizer.show_comparison(frame, processed, "Original vs Processed")
                
                # 显示质量信息
                quality_info = {
                    "Quality": quality["quality"],
                    "Blur Score": f"{quality['blur_score']:.1f}",
                    "Brightness": f"{quality['brightness']:.1f}",
                    "FPS": f"{capture.get_fps():.1f}"
                }
                
                info_image = visualizer.draw_info(processed, quality_info)
                key = visualizer.show_image(info_image, "Quality Info")
                
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    processor.enable_enhancement = not processor.enable_enhancement
                    print(f"图像增强: {'开启' if processor.enable_enhancement else '关闭'}")
                elif key == ord('n'):
                    processor.enable_noise_reduction = not processor.enable_noise_reduction
                    print(f"降噪: {'开启' if processor.enable_noise_reduction else '关闭'}")
            else:
                time.sleep(0.01)
    
    finally:
        capture.stop()
        visualizer.close_all()

if __name__ == "__main__":
    print("图像处理模块测试")
    print("1. 摄像头采集测试")
    print("2. 图像处理测试")
    
    choice = input("请选择测试项目 (1 或 2): ")
    
    if choice == "1":
        test_camera_capture()
    elif choice == "2":
        test_image_processing()
    else:
        print("无效选择")
