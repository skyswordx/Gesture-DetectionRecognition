# 图像处理模块使用说明

## 模块概述

图像处理模块提供摄像头采集、图像预处理、质量评估和可视化等基础功能，是整个系统的图像输入和预处理基础。

## 主要功能

### 1. 摄像头采集 (CameraCapture)
- 多线程摄像头数据采集
- 实时FPS统计
- 帧缓冲队列管理
- 自动参数设置

### 2. 图像处理 (ImageProcessor)
- 图像预处理 (镜像、亮度、对比度)
- 自动亮度调整
- 双边滤波降噪
- CLAHE图像增强
- 模糊度检测

### 3. 质量评估 (ImageQualityAssessment)
- 图像模糊度评估
- 亮度范围检查
- 图像尺寸验证
- 综合质量打分

### 4. 可视化 (ImageVisualizer)
- 图像显示
- 对比显示
- 信息叠加
- 窗口管理

## 使用方法

### 基础使用
```python
from modules.image_processing.image_processor import CameraCapture, ImageProcessor

# 创建摄像头采集器
capture = CameraCapture(camera_id=0, width=640, height=480)
capture.start()

# 创建图像处理器
processor = ImageProcessor()

# 获取和处理图像
frame = capture.get_frame()
if frame is not None:
    processed_frame = processor.preprocess(frame)

# 停止采集
capture.stop()
```

### 高级功能
```python
from modules.image_processing.image_processor import *

# 质量评估
quality_assessor = ImageQualityAssessment()
quality_result = quality_assessor.assess_quality(image)

# 图像增强
processor = ImageProcessor()
enhanced_image = processor.enhance_image(image)

# 可视化对比
visualizer = ImageVisualizer()
visualizer.show_comparison(original, processed)
```

## 验证程序

### 运行方式
```bash
cd modules/image_processing
python image_processor.py
```

### 测试选项
1. **摄像头采集测试**: 验证摄像头功能和FPS显示
2. **图像处理测试**: 验证图像处理效果和质量评估

### 交互控制
- `q`: 退出程序
- `e`: 切换图像增强开/关
- `n`: 切换降噪开/关

## 配置参数

### 摄像头参数
```python
camera_id = 0        # 摄像头ID
width = 640         # 图像宽度
height = 480        # 图像高度
fps = 30           # 目标帧率
```

### 处理参数
```python
brightness_factor = 1.0    # 亮度因子
contrast_factor = 1.0      # 对比度因子
enable_noise_reduction = True  # 启用降噪
enable_enhancement = True      # 启用增强
```

### 质量阈值
```python
blur_threshold = 100.0        # 模糊度阈值
brightness_range = (50, 200)  # 亮度范围
```

## 常见问题

### 1. 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 尝试不同的camera_id (0, 1, 2...)
- 确认摄像头驱动正常

### 2. 图像质量差
- 调整光线条件
- 启用图像增强功能
- 调整亮度对比度参数

### 3. FPS过低
- 降低图像分辨率
- 关闭不必要的图像处理功能
- 检查系统性能

## 输出示例

### 质量评估结果
```python
{
    "valid": True,
    "quality": "good",
    "reason": "质量良好",
    "blur_score": 150.5,
    "brightness": 128.3,
    "is_sharp": True,
    "is_bright_ok": True,
    "is_size_ok": True,
    "width": 640,
    "height": 480
}
```

## 技术规格

- **支持格式**: BGR彩色图像
- **输入分辨率**: 320x240 到 1920x1080
- **处理速度**: 25-30 FPS (640x480)
- **内存占用**: < 100MB
- **依赖库**: OpenCV, NumPy

## 扩展接口

模块提供了清晰的接口，便于其他模块调用：

```python
# 获取预处理后的图像
def get_processed_frame() -> np.ndarray

# 设置处理参数
def set_parameters(brightness, contrast, ...)

# 评估图像质量
def assess_quality(image) -> dict
```
