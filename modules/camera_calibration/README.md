# 鱼眼摄像头标定模块

本模块为手势识别系统提供全面的鱼眼摄像头标定和畸变校正功能。设计为模块化、易于使用，适用于无人机应用。

## 功能特性

- **鱼眼摄像头标定**: 使用棋盘格模式进行精确标定
- **实时畸变校正**: 使用优化算法进行快速畸变校正
- **交互式标定应用**: 用户友好的摄像头标定应用程序
- **参数调节**: 实时参数调整以获得最佳效果
- **可视化工具**: 标定过程的全面可视化
- **性能监控**: 内置性能统计和质量评估

## 模块结构

```
camera_calibration/
├── __init__.py                 # 模块初始化
├── fisheye_calibrator.py       # 核心标定功能
├── distortion_corrector.py     # 实时畸变校正
├── calibration_visualizer.py   # 可视化工具
├── calibration_app.py          # 交互式标定应用
└── README.md                   # 本文档
```

## 快速开始

### 1. 基本用法

```python
from modules.camera_calibration import (
    FisheyeCalibrator, 
    DistortionCorrector, 
    CalibrationVisualizer
)

# 初始化标定器
calibrator = FisheyeCalibrator(chessboard_size=(9, 6), square_size=1.0)

# 添加标定图像（含棋盘格模式）
for image in calibration_images:
    calibrator.add_calibration_image(image)

# 执行标定
result = calibrator.calibrate()

# 创建畸变校正器
corrector = DistortionCorrector(result)

# 校正图像
corrected_image = corrector.correct_distortion(fisheye_image)
```

### 2. 交互式标定

运行交互式标定应用程序：

```bash
cd modules/camera_calibration
python calibration_app.py --camera 0 --board-width 9 --board-height 6
```

**控制键：**
- `c`: 切换到捕获模式
- `空格键`: 捕获标定图像
- `o`: 切换自动捕获模式
- `l`: 执行标定
- `t`: 测试畸变校正
- `a`: 调整校正参数
- `s`: 保存标定数据
- `d`: 加载标定数据
- `h`: 切换帮助显示
- `q`: 退出应用程序

### 3. 与主系统集成

```python
# 在你的主手势识别系统中
from modules.camera_calibration import DistortionCorrector

class GestureControlSystem:
    def __init__(self):
        # ... 现有初始化 ...
        
        # 初始化畸变校正器
        self.distortion_corrector = DistortionCorrector()
        
        # 加载预标定参数
        if self.load_camera_calibration():
            print("鱼眼校正已启用")
        else:
            print("运行时不使用鱼眼校正")
    
    def load_camera_calibration(self):
        try:
            from modules.camera_calibration import FisheyeCalibrator
            calibrator = FisheyeCalibrator()
            if calibrator.load_calibration("calibration_data/fisheye_calibration.json"):
                self.distortion_corrector.set_calibration(calibrator.calibration_result)
                return True
        except Exception as e:
            logger.warning(f"加载摄像头标定失败: {e}")
        return False
    
    def process_frame(self):
        # 获取摄像头帧
        frame = self.camera_capture.get_frame()
        
        # 如果可用，应用畸变校正
        if self.distortion_corrector.is_initialized:
            frame = self.distortion_corrector.correct_distortion(frame)
        
        # 继续现有处理...
        # 姿势检测、手势识别等
```

## 标定过程

### 1. 准备工作

1. **打印棋盘格模式**: 打印高质量的棋盘格模式
   - 默认尺寸: 9x6内部角点
   - 确保平整打印，无畸变
   - 安装在刚性表面上

2. **设置环境**: 
   - 良好的均匀照明
   - 稳定的摄像头安装
   - 清晰的棋盘格视野

### 2. 标定步骤

1. **启动应用**: `python calibration_app.py`
2. **捕获模式**: 按 `c` 进入捕获模式
3. **收集图像**: 
   - 将棋盘格放置在不同位置和角度
   - 使用`空格键`捕获15-30张图像，或按`o`启用自动捕获
   - 覆盖鱼眼视野的不同区域
4. **标定**: 按 `l` 执行标定
5. **测试结果**: 按 `t` 测试畸变校正
6. **调整参数**: 按 `a` 微调校正参数
7. **保存**: 按 `s` 保存标定数据

### 3. 质量评估

良好的标定应该具有：
- **RMS误差 < 1.0**: 越低越好
- **15张以上图像**: 更多图像提高精度
- **良好覆盖**: 来自不同角度和距离的图像
- **清晰模式**: 检测到所有棋盘格角点

## 参数调节

### 畸变校正参数

- **平衡系数 (0.0-1.0)**: 
  - 0.0: 保留所有原始图像内容（黑色边框）
  - 1.0: 填充整个输出图像（可能裁剪内容）
  
- **视野缩放 (0.1-2.0)**: 
  - <1.0: 放大（裁剪视野）
  - >1.0: 缩小（扩展视野）
  
- **裁剪系数 (0.1-1.0)**: 
  - 校正图像的附加裁剪
  - 用于去除畸变边缘

### 实时调整

在调整模式中（`a`）：
- `1/2`: 增加/减少平衡系数
- `3/4`: 增加/减少视野缩放
- `5/6`: 增加/减少裁剪系数
- `r`: 重置为默认值

## 性能优化

### 实时应用

```python
# 初始化一次
corrector = DistortionCorrector(calibration_result)

# 使用预计算映射进行快速校正
corrected = corrector.correct_distortion(image, fast_mode=True)
```

### 无人机应用

```python
# 对变化条件使用自适应校正器
from modules.camera_calibration import AdaptiveDistortionCorrector

corrector = AdaptiveDistortionCorrector(calibration_result)

# 基于前几帧自动调整参数
corrector.auto_adjust_parameters(sample_image)

# 使用优化设置继续
corrected = corrector.correct_distortion(image)
```

## 文件格式

### 标定数据 (JSON)
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coeffs": [k1, k2, k3, k4],
  "rms_error": 0.5,
  "image_size": [1280, 720],
  "calibration_date": "2024-01-01 12:00:00",
  "is_valid": true
}
```

## 故障排除

### 常见问题

1. **高RMS误差 (>2.0)**：
   - 确保棋盘格平整且高质量
   - 改善照明条件
   - 捕获更多样化的图像
   - 检查摄像头焦距

2. **棋盘格未检测到**：
   - 改善照明（避免阴影/眩光）
   - 确保棋盘格清晰可见
   - 检查棋盘格尺寸参数
   - 清洁摄像头镜头

3. **校正质量差**：
   - 用更多图像重新标定
   - 调整校正参数
   - 使用更高分辨率图像
   - 确保标定覆盖全视野

4. **性能问题**：
   - 对实时应用使用fast_mode=True
   - 降低图像分辨率
   - 优化校正参数
   - 考虑硬件加速

### 调试模式

启用调试日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 集成示例

### 示例1: 手势识别系统

```python
def integrate_fisheye_correction(self):
    """向现有系统添加鱼眼校正"""
    try:
        from modules.camera_calibration import DistortionCorrector, FisheyeCalibrator
        
        # 加载标定
        calibrator = FisheyeCalibrator()
        if calibrator.load_calibration("calibration_data/fisheye_calibration.json"):
            self.fisheye_corrector = DistortionCorrector(calibrator.calibration_result)
            logger.info("鱼眼校正已启用")
            return True
    except Exception as e:
        logger.warning(f"鱼眼校正不可用: {e}")
    
    self.fisheye_corrector = None
    return False

def process_camera_frame(self, frame):
    """使用可选鱼眼校正处理帧"""
    # 如果可用，应用鱼眼校正
    if self.fisheye_corrector and self.fisheye_corrector.is_initialized:
        frame = self.fisheye_corrector.correct_distortion(frame)
    
    # 继续现有处理
    return self.existing_frame_processor(frame)
```

### 示例2: 参数调整界面

```python
def create_adjustment_interface(self):
    """创建参数调整界面"""
    if not self.fisheye_corrector:
        return
    
    cv2.namedWindow("鱼眼参数")
    
    # 创建滑动条
    cv2.createTrackbar("平衡", "鱼眼参数", 0, 100, self.on_balance_change)
    cv2.createTrackbar("视野缩放", "鱼眼参数", 100, 200, self.on_fov_change)
    cv2.createTrackbar("裁剪", "鱼眼参数", 100, 100, self.on_crop_change)

def on_balance_change(self, val):
    balance = val / 100.0
    self.fisheye_corrector.set_correction_parameters(balance=balance)

def on_fov_change(self, val):
    fov_scale = val / 100.0
    self.fisheye_corrector.set_correction_parameters(fov_scale=fov_scale)
```

## 未来增强

- **多摄像头标定**: 支持立体鱼眼设置
- **在线标定**: 持续标定细化
- **GPU加速**: CUDA/OpenCL支持以实现更快处理
- **自动模式检测**: 支持不同标定模式
- **基于质量的参数优化**: 自动参数调整

## 许可证

本模块是手势识别控制系统的一部分，遵循相同的许可条款。
