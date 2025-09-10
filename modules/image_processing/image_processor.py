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
            quality_reason = "Blurry image"
        elif not is_bright_ok:
            quality_reason = "Abnormal brightness"
        elif not is_size_ok:
            quality_reason = "Size too small"
        else:
            quality_reason = "Good quality"
        
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
    """图像可视化类 - 支持分层信息显示，防止信息重叠"""
    
    def __init__(self):
        """初始化可视化器"""
        self.window_names = set()
        self.info_display_config = {
            'essential_only': True,  # 只显示重要信息
            'position_top_right': True,  # 信息显示在右上角
            'max_info_items': 3,  # 最多显示的信息条目
            'font_scale': 0.5,  # 字体大小
            'line_spacing': 20,  # 行间距
            'show_status_indicator': True,  # 显示状态指示器
            'show_debug_info': False,  # 显示调试信息
            'enable_info_panels': False,  # 启用独立信息面板
        }
        # 区域占用跟踪，避免信息重叠
        self.occupied_regions = []
        
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
    
    def draw_essential_info(self, image: np.ndarray, info: dict, 
                           position: str = 'top_right') -> np.ndarray:
        """
        在图像上绘制核心信息（精简版），智能避免重叠
        
        Args:
            image: 图像
            info: 信息字典
            position: 显示位置 ('top_right', 'top_left', 'bottom_right', 'bottom_left')
            
        Returns:
            绘制了信息的图像
        """
        if image is None:
            return None
        
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.info_display_config['font_scale']
        line_spacing = self.info_display_config['line_spacing']
        
        # 过滤并排序信息（只显示最重要的）
        essential_keys = ['FPS', 'Quality', 'Gesture', 'Distance', 'Status']
        filtered_info = {k: v for k, v in info.items() if k in essential_keys}
        
        # 限制显示条目数
        max_items = self.info_display_config['max_info_items']
        if len(filtered_info) > max_items:
            # 按重要性排序
            priority_order = ['FPS', 'Status', 'Gesture', 'Quality', 'Distance']
            sorted_items = []
            for key in priority_order:
                if key in filtered_info and len(sorted_items) < max_items:
                    sorted_items.append((key, filtered_info[key]))
            filtered_info = dict(sorted_items)
        
        # 计算文本位置
        text_lines = [f"{k}: {v}" for k, v in filtered_info.items()]
        if not text_lines:
            return result
        
        text_sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in text_lines]
        max_width = max([size[0] for size in text_sizes]) if text_sizes else 0
        total_height = len(text_lines) * line_spacing
        
        h, w = image.shape[:2]
        
        # 智能计算起始坐标，避免与状态指示器重叠
        start_x, start_y = self._get_safe_position(w, h, max_width, total_height, position)
        
        # 绘制半透明背景
        if text_lines:
            overlay = result.copy()
            bg_rect = (start_x - 5, start_y - 15, 
                      start_x + max_width + 5, start_y + total_height - 5)
            cv2.rectangle(overlay, bg_rect[:2], bg_rect[2:], (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
            
            # 记录占用区域
            self._mark_region_occupied(bg_rect)
        
        # 绘制文本
        for i, text in enumerate(text_lines):
            y_pos = start_y + i * line_spacing
            cv2.putText(result, text, (start_x, y_pos), font, font_scale, (255, 255, 255), 1)
        
        return result
    
    def draw_debug_info(self, image: np.ndarray, debug_info: dict) -> np.ndarray:
        """
        在图像底部绘制调试信息
        
        Args:
            image: 图像
            debug_info: 调试信息字典
            
        Returns:
            绘制了调试信息的图像
        """
        if image is None or not debug_info:
            return image
        
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        line_spacing = 15
        
        h, w = image.shape[:2]
        
        # 格式化调试信息
        debug_lines = []
        for key, value in debug_info.items():
            if isinstance(value, float):
                debug_lines.append(f"{key}: {value:.2f}")
            else:
                debug_lines.append(f"{key}: {value}")
        
        # 在底部绘制调试信息
        total_height = len(debug_lines) * line_spacing + 10
        start_y = h - total_height
        
        # 绘制半透明背景
        overlay = result.copy()
        cv2.rectangle(overlay, (0, start_y - 5), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # 绘制调试文本
        for i, text in enumerate(debug_lines):
            y_pos = start_y + i * line_spacing + 15
            cv2.putText(result, text, (10, y_pos), font, font_scale, (200, 200, 200), 1)
        
        return result
    
    def draw_status_indicator(self, image: np.ndarray, status: str, 
                            confidence: float = 0.0) -> np.ndarray:
        """
        在图像左上角绘制状态指示器，并记录占用区域
        
        Args:
            image: 图像
            status: 状态文本
            confidence: 置信度 (0-1)
            
        Returns:
            绘制了状态指示器的图像
        """
        if image is None:
            return None
        
        result = image.copy()
        
        # 根据状态选择颜色
        if status.lower() in ['good', 'ok', 'ready', '就绪', '正常']:
            color = (0, 255, 0)  # 绿色
        elif status.lower() in ['warning', 'poor', '警告', '差']:
            color = (0, 255, 255)  # 黄色
        elif status.lower() in ['error', 'bad', '错误', '失败']:
            color = (0, 0, 255)  # 红色
        else:
            color = (255, 255, 255)  # 白色
        
        # 绘制状态圆圈
        circle_center = (25, 25)
        cv2.circle(result, circle_center, 10, color, -1)
        
        # 绘制状态文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        status_text_pos = (45, 30)
        cv2.putText(result, status, status_text_pos, font, 0.5, color, 1)
        
        # 计算状态指示器占用区域
        status_region = (10, 10, 160, 50)  # 左上角区域
        self._mark_region_occupied(status_region)
        
        # 如果有置信度，绘制置信度条
        if confidence > 0:
            bar_width = 100
            bar_height = 8
            bar_x = 45
            bar_y = 35
            
            # 背景条
            cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # 置信度条
            confidence_width = int(bar_width * confidence)
            cv2.rectangle(result, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
            
            # 置信度文本
            cv2.putText(result, f"{confidence*100:.0f}%", (bar_x + bar_width + 5, bar_y + 6), 
                       font, 0.3, color, 1)
            
            # 更新占用区域（包含置信度条）
            status_region = (10, 10, 160, 55)
            self._mark_region_occupied(status_region)
        
        return result
    
    def _get_safe_position(self, width: int, height: int, content_width: int, 
                          content_height: int, preferred_position: str) -> tuple:
        """
        获取安全的绘制位置，避免与已占用区域重叠
        
        Args:
            width: 图像宽度
            height: 图像高度  
            content_width: 内容宽度
            content_height: 内容高度
            preferred_position: 首选位置
            
        Returns:
            (x, y) 坐标
        """
        margin = 10
        
        # 定义可能的位置
        positions = {
            'top_right': (width - content_width - margin, 25),
            'top_left': (margin, 25),
            'bottom_right': (width - content_width - margin, height - content_height - margin),
            'bottom_left': (margin, height - content_height - margin),
            'center_right': (width - content_width - margin, height // 2 - content_height // 2),
            'center_left': (margin, height // 2 - content_height // 2),
        }
        
        # 首先尝试首选位置
        if preferred_position in positions:
            x, y = positions[preferred_position]
            test_region = (x - 5, y - 15, x + content_width + 5, y + content_height - 5)
            if not self._is_region_occupied(test_region):
                return x, y
        
        # 如果首选位置被占用，尝试其他位置
        position_priority = ['top_right', 'top_left', 'bottom_right', 'bottom_left', 
                           'center_right', 'center_left']
        
        for pos in position_priority:
            if pos in positions:
                x, y = positions[pos]
                test_region = (x - 5, y - 15, x + content_width + 5, y + content_height - 5)
                if not self._is_region_occupied(test_region):
                    return x, y
        
        # 如果所有预定义位置都被占用，返回默认位置
        return positions.get(preferred_position, (margin, 25))
    
    def _is_region_occupied(self, region: tuple) -> bool:
        """
        检查区域是否被占用
        
        Args:
            region: (x1, y1, x2, y2) 矩形区域
            
        Returns:
            是否被占用
        """
        x1, y1, x2, y2 = region
        for occupied in self.occupied_regions:
            ox1, oy1, ox2, oy2 = occupied
            # 检查矩形重叠
            if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                return True
        return False
    
    def _mark_region_occupied(self, region: tuple):
        """
        标记区域为已占用
        
        Args:
            region: (x1, y1, x2, y2) 矩形区域
        """
        self.occupied_regions.append(region)
    
    def _clear_occupied_regions(self):
        """清除所有占用区域记录"""
        self.occupied_regions.clear()
    
    def draw_info(self, image: np.ndarray, info: dict, 
                 info_type: str = 'essential') -> np.ndarray:
        """
        根据信息类型智能绘制信息，自动避免重叠
        
        Args:
            image: 图像
            info: 信息字典
            info_type: 信息类型 ('essential', 'debug', 'full', 'smart')
            
        Returns:
            绘制了信息的图像
        """
        if image is None:
            return None
        
        # 每次绘制前清除占用区域记录
        self._clear_occupied_regions()
        
        result = image.copy()
        
        if info_type == 'smart':
            # 智能模式：根据信息类型分层显示
            result = self._draw_smart_info(result, info)
        elif info_type == 'essential':
            result = self.draw_essential_info(result, info)
        elif info_type == 'debug':
            result = self.draw_debug_info(result, info)
        elif info_type == 'full':
            # 保持原有的完整信息显示方式
            result = self._draw_full_info(result, info)
        else:
            result = self.draw_essential_info(result, info)
        
        return result
    
    def _draw_smart_info(self, image: np.ndarray, info: dict) -> np.ndarray:
        """
        智能分层信息显示，避免重叠并优化布局
        
        Args:
            image: 图像
            info: 完整信息字典
            
        Returns:
            绘制了信息的图像
        """
        result = image.copy()
        
        # 1. 首先绘制状态指示器（左上角）
        if 'Status' in info and self.info_display_config['show_status_indicator']:
            confidence = info.get('Confidence', 0)
            result = self.draw_status_indicator(result, info['Status'], confidence)
        
        # 2. 绘制核心信息（右上角，避开状态指示器）
        essential_info = {k: v for k, v in info.items() 
                         if k in ['FPS', 'Quality', 'Gesture', 'Distance']}
        if essential_info:
            result = self.draw_essential_info(result, essential_info, 'top_right')
        
        # 3. 如果启用了调试信息，在底部显示
        if self.info_display_config['show_debug_info']:
            debug_info = {k: v for k, v in info.items() 
                         if k not in ['FPS', 'Quality', 'Gesture', 'Distance', 'Status']}
            if debug_info:
                result = self.draw_debug_info(result, debug_info)
        
        return result
    
    def _draw_full_info(self, image: np.ndarray, info: dict) -> np.ndarray:
        """
        绘制完整信息（原方法）
        
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
    
    def configure_display(self, essential_only: bool = True, 
                         position_top_right: bool = True,
                         max_info_items: int = 3,
                         font_scale: float = 0.5,
                         show_status_indicator: bool = True,
                         show_debug_info: bool = False,
                         enable_info_panels: bool = False):
        """
        配置显示参数
        
        Args:
            essential_only: 是否只显示核心信息
            position_top_right: 信息是否显示在右上角
            max_info_items: 最大信息条目数
            font_scale: 字体大小
            show_status_indicator: 是否显示状态指示器
            show_debug_info: 是否显示调试信息
            enable_info_panels: 是否启用独立信息面板
        """
        self.info_display_config.update({
            'essential_only': essential_only,
            'position_top_right': position_top_right,
            'max_info_items': max_info_items,
            'font_scale': font_scale,
            'show_status_indicator': show_status_indicator,
            'show_debug_info': show_debug_info,
            'enable_info_panels': enable_info_panels
        })
        
    def get_display_config(self) -> dict:
        """
        获取当前显示配置
        
        Returns:
            显示配置字典
        """
        return self.info_display_config.copy()
    
    def set_display_mode(self, mode: str):
        """
        设置显示模式（预设配置）
        
        Args:
            mode: 显示模式 ('minimal', 'standard', 'detailed', 'debug')
        """
        if mode == 'minimal':
            self.configure_display(
                essential_only=True,
                max_info_items=2,
                show_status_indicator=True,
                show_debug_info=False,
                enable_info_panels=False
            )
        elif mode == 'standard':
            self.configure_display(
                essential_only=True,
                max_info_items=3,
                show_status_indicator=True,
                show_debug_info=False,
                enable_info_panels=False
            )
        elif mode == 'detailed':
            self.configure_display(
                essential_only=False,
                max_info_items=5,
                show_status_indicator=True,
                show_debug_info=True,
                enable_info_panels=True
            )
        elif mode == 'debug':
            self.configure_display(
                essential_only=False,
                max_info_items=8,
                show_status_indicator=True,
                show_debug_info=True,
                enable_info_panels=True
            )
    
    def create_info_panel(self, info: dict, width: int = 300, height: int = 200) -> np.ndarray:
        """
        创建独立的信息面板图像
        
        Args:
            info: 信息字典
            width: 面板宽度
            height: 面板高度
            
        Returns:
            信息面板图像
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(40)  # 深灰色背景
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_spacing = 25
        
        # 标题
        cv2.putText(panel, "System Info", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.line(panel, (10, 35), (width - 10, 35), (100, 100, 100), 1)
        
        # 信息内容
        y_offset = 60
        for key, value in info.items():
            if y_offset > height - 30:
                break
            
            text = f"{key}: {value}"
            cv2.putText(panel, text, (15, y_offset), font, font_scale, (200, 200, 200), 1)
            y_offset += line_spacing
        
        return panel
    
    def create_detailed_info_panel(self, info: dict, panel_type: str = "system", 
                                  width: int = 400, height: int = 300) -> np.ndarray:
        """
        创建详细的信息面板，支持不同类型
        
        Args:
            info: 信息字典
            panel_type: 面板类型 ('system', 'performance', 'detection', 'gesture')
            width: 面板宽度
            height: 面板高度
            
        Returns:
            信息面板图像
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(30)  # 深灰色背景
        
        # 添加渐变背景
        for i in range(height):
            alpha = 0.3 + 0.1 * (i / height)
            panel[i, :] = (int(30 * alpha), int(40 * alpha), int(50 * alpha))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 根据面板类型设置标题和内容
        if panel_type == "system":
            self._draw_system_panel(panel, info, font, width, height)
        elif panel_type == "performance":
            self._draw_performance_panel(panel, info, font, width, height)
        elif panel_type == "detection":
            self._draw_detection_panel(panel, info, font, width, height)
        elif panel_type == "gesture":
            self._draw_gesture_panel(panel, info, font, width, height)
        else:
            self._draw_generic_panel(panel, info, font, width, height)
        
        return panel
    
    def _draw_system_panel(self, panel: np.ndarray, info: dict, font, width: int, height: int):
        """绘制系统信息面板"""
        # 标题
        cv2.putText(panel, "System Status", (10, 30), font, 0.8, (100, 255, 100), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (100, 255, 100), 2)
        
        # 系统信息
        y_offset = 70
        system_keys = ['FPS', 'Quality', 'Status', 'Processing Time', 'Memory Usage']
        
        for key in system_keys:
            if key in info and y_offset < height - 30:
                value = info[key]
                # 根据值类型选择颜色
                if key == 'FPS':
                    color = (0, 255, 0) if float(str(value).split()[0]) > 20 else (0, 255, 255)
                elif key == 'Quality':
                    color = (0, 255, 0) if value in ['Good', 'Excellent'] else (0, 165, 255)
                else:
                    color = (200, 200, 200)
                
                cv2.putText(panel, f"{key}:", (15, y_offset), font, 0.6, (255, 255, 255), 1)
                cv2.putText(panel, str(value), (150, y_offset), font, 0.6, color, 1)
                y_offset += 30
    
    def _draw_performance_panel(self, panel: np.ndarray, info: dict, font, width: int, height: int):
        """绘制性能信息面板"""
        cv2.putText(panel, "Performance", (10, 30), font, 0.8, (255, 255, 100), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (255, 255, 100), 2)
        
        # 性能指标
        y_offset = 70
        perf_keys = ['FPS', 'Processing Time', 'CPU Usage', 'Memory Usage', 'Frame Count']
        
        for key in perf_keys:
            if key in info and y_offset < height - 30:
                value = info[key]
                cv2.putText(panel, f"{key}: {value}", (15, y_offset), font, 0.5, (200, 200, 200), 1)
                y_offset += 25
    
    def _draw_detection_panel(self, panel: np.ndarray, info: dict, font, width: int, height: int):
        """绘制检测信息面板"""
        cv2.putText(panel, "Detection Info", (10, 30), font, 0.8, (100, 100, 255), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (100, 100, 255), 2)
        
        y_offset = 70
        detection_keys = ['Pose Detected', 'Landmarks Count', 'Confidence', 'Distance', 'Quality']
        
        for key in detection_keys:
            if key in info and y_offset < height - 30:
                value = info[key]
                cv2.putText(panel, f"{key}: {value}", (15, y_offset), font, 0.5, (200, 200, 200), 1)
                y_offset += 25
    
    def _draw_gesture_panel(self, panel: np.ndarray, info: dict, font, width: int, height: int):
        """绘制手势信息面板"""
        cv2.putText(panel, "Gesture Recognition", (10, 30), font, 0.8, (255, 100, 100), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (255, 100, 100), 2)
        
        y_offset = 70
        gesture_keys = ['Current Gesture', 'Confidence', 'Duration', 'Last Command', 'Success Rate']
        
        for key in gesture_keys:
            if key in info and y_offset < height - 30:
                value = info[key]
                cv2.putText(panel, f"{key}: {value}", (15, y_offset), font, 0.5, (200, 200, 200), 1)
                y_offset += 25
    
    def _draw_generic_panel(self, panel: np.ndarray, info: dict, font, width: int, height: int):
        """绘制通用信息面板"""
        cv2.putText(panel, "Information", (10, 30), font, 0.8, (200, 200, 200), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (200, 200, 200), 2)
        
        y_offset = 70
        for key, value in info.items():
            if y_offset < height - 30:
                cv2.putText(panel, f"{key}: {value}", (15, y_offset), font, 0.5, (200, 200, 200), 1)
                y_offset += 25
    
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
    
    print("摄像头启动成功，按 'q' 键退出, 'e' 切换显示模式")
    
    visualizer = ImageVisualizer()
    display_mode = 'essential'  # essential, debug, full
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 准备显示信息
                fps_info = {
                    "FPS": f"{capture.get_fps():.1f}",
                    "Status": "Running",
                    "Frame": f"{capture.frame_count}"
                }
                
                # 根据显示模式绘制信息
                if display_mode == 'essential':
                    frame_with_info = visualizer.draw_essential_info(frame, fps_info)
                elif display_mode == 'debug':
                    debug_info = {
                        "Actual FPS": capture.get_fps(),
                        "Target FPS": capture.fps,
                        "Frame Count": capture.frame_count,
                        "Queue Size": capture.frame_queue.qsize()
                    }
                    frame_with_info = visualizer.draw_debug_info(frame, debug_info)
                else:  # full
                    frame_with_info = visualizer.draw_info(frame, fps_info, 'full')
                
                # 添加状态指示器
                frame_with_info = visualizer.draw_status_indicator(frame_with_info, "OK", 0.9)
                
                key = visualizer.show_image(frame_with_info, "Camera Test")
                
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    # 切换显示模式
                    modes = ['essential', 'debug', 'full']
                    current_index = modes.index(display_mode)
                    display_mode = modes[(current_index + 1) % len(modes)]
                    print(f"切换到 {display_mode} 显示模式")
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
    
    print("图像处理测试，按键说明:")
    print("  'q' - 退出")
    print("  'e' - 切换图像增强")
    print("  'n' - 切换降噪")
    print("  'd' - 切换显示模式")
    print("  'i' - 显示独立信息面板")
    
    display_mode = 'essential'
    show_info_panel = False
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is not None:
                # 处理图像
                processed = processor.preprocess(frame)
                
                # 质量评估
                quality = quality_assessor.assess_quality(processed)
                
                # 准备显示信息
                essential_info = {
                    "FPS": f"{capture.get_fps():.1f}",
                    "Quality": quality["quality"],
                    "Status": "Good" if quality["valid"] else "Poor"
                }
                
                debug_info = {
                    "Blur Score": quality['blur_score'],
                    "Brightness": quality['brightness'],
                    "Enhancement": "On" if processor.enable_enhancement else "Off",
                    "Noise Reduction": "On" if processor.enable_noise_reduction else "Off",
                    "Resolution": f"{quality['width']}x{quality['height']}"
                }
                
                # 根据显示模式绘制信息
                if display_mode == 'essential':
                    display_frame = visualizer.draw_essential_info(processed, essential_info, 'top_right')
                elif display_mode == 'debug':
                    display_frame = visualizer.draw_debug_info(processed, debug_info)
                else:  # full
                    all_info = {**essential_info, **debug_info}
                    display_frame = visualizer.draw_info(processed, all_info, 'full')
                
                # 添加状态指示器
                status = "Good" if quality["valid"] else "Poor"
                confidence = min(quality['blur_score'] / 200.0, 1.0)  # 简单的置信度计算
                display_frame = visualizer.draw_status_indicator(display_frame, status, confidence)
                
                # 显示主图像
                key = visualizer.show_image(display_frame, "Image Processing Test")
                
                # 显示独立信息面板
                if show_info_panel:
                    panel_info = {
                        "Camera FPS": f"{capture.get_fps():.1f}",
                        "Image Quality": quality["quality"],
                        "Blur Score": f"{quality['blur_score']:.1f}",
                        "Brightness": f"{quality['brightness']:.1f}",
                        "Resolution": f"{quality['width']}x{quality['height']}",
                        "Enhancement": "On" if processor.enable_enhancement else "Off",
                        "Noise Reduction": "On" if processor.enable_noise_reduction else "Off"
                    }
                    info_panel = visualizer.create_info_panel(panel_info)
                    visualizer.show_image(info_panel, "System Info Panel")
                
                # 处理按键
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    processor.enable_enhancement = not processor.enable_enhancement
                    print(f"图像增强: {'开启' if processor.enable_enhancement else '关闭'}")
                elif key == ord('n'):
                    processor.enable_noise_reduction = not processor.enable_noise_reduction
                    print(f"降噪: {'开启' if processor.enable_noise_reduction else '关闭'}")
                elif key == ord('d'):
                    modes = ['essential', 'debug', 'full']
                    current_index = modes.index(display_mode)
                    display_mode = modes[(current_index + 1) % len(modes)]
                    print(f"切换到 {display_mode} 显示模式")
                elif key == ord('i'):
                    show_info_panel = not show_info_panel
                    if not show_info_panel:
                        cv2.destroyWindow("System Info Panel")
                    print(f"信息面板: {'显示' if show_info_panel else '隐藏'}")
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
