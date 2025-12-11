# 手势识别控制系统 - 合并版本

## 概述

`main_integrated.py` 是一个完整的手势识别控制系统，整合了原来的 `gui_system.py` 和 `main_integrated.py` 的所有功能。该文件提供了两种运行模式：GUI图形界面模式和控制台模式。

## 功能特性

### 🖥️ 双模式支持
- **GUI模式**: 完整的图形用户界面，支持实时视频显示和交互控制
- **控制台模式**: 传统的控制台界面，适合调试和服务器环境

### 📷 图像处理
- 实时摄像头输入
- 图像预处理和质量assessment
- 多种显示模式切换

### 🤖 检测功能
- **人体姿势检测**: 基于MediaPipe的实时姿势检测
- **距离估算**: 智能计算用户与摄像头的距离
- **手势识别**: 支持多种控制手势识别

### 🎮 控制命令
支持以下手势控制命令：
- 🙌 **起飞**: 双手高举过头
- 👇 **降落**: 双手向下压
- 👉 **前进**: 右手前推
- 👈 **左移**: 左手指向左侧
- 👉 **右移**: 右手指向右侧
- ☝️ **上升**: 双手向上推举
- 👇 **下降**: 双手向下压
- ✋ **停止**: 双手胸前交叉

## 系统要求

### Python 版本
- Python 3.7+

### 必需依赖
```bash
pip install opencv-python
pip install numpy
pip install mediapipe
pip install Pillow
pip install tkinter  # 通常随Python一起安装
```

### 硬件要求
- 摄像头 (USB摄像头或内置摄像头)
- 足够的系统内存 (建议4GB+)

## 安装和使用

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd Gesture-DetectionRecognition
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行系统

#### 方式1: 交互式模式选择
```bash
python launch/main_integrated.py
```
会弹出对话框让您选择运行模式。

#### 方式2: 直接启动GUI模式
```bash
python launch/main_integrated.py --gui
```

#### 方式3: 直接启动控制台模式
```bash
python launch/main_integrated.py --console
```

#### 方式4: 自定义参数
```bash
# 使用摄像头1，设置分辨率为800x600，启用调试模式
python launch/main_integrated.py --console --camera 1 --width 800 --height 600 --debug
```

## 命令行参数

| 参数 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--mode {gui,console}` | `-m` | 启动模式 | 交互选择 |
| `--gui` |  | 启动GUI模式 |  |
| `--console` |  | 启动控制台模式 |  |
| `--camera CAMERA` | `-c` | 摄像头ID | 0 |
| `--width WIDTH` |  | 图像宽度 | 1920 |
| `--height HEIGHT` |  | 图像高度 | 1080 |
| `--debug` |  | 启用调试模式 | False |

## 使用指南

### GUI模式操作

#### 系统控制面板
- **启动摄像头**: 开始视频捕获和检测
- **停止检测**: 停止所有检测功能
- **暂停/继续**: 暂停或继续检测处理

#### 显示模式
- **完整模式**: 显示所有检测信息
- **仅姿势检测**: 只显示骨骼点
- **仅距离估算**: 只显示距离信息
- **仅手势识别**: 只显示手势识别结果

#### 参数设置
- **检测灵敏度**: 调整手势识别的敏感度
- **显示调试信息**: 切换调试信息显示

#### 视频控制
- **截图保存**: 保存当前帧 (功能待实现)
- **录制视频**: 录制视频 (功能待实现)
- **重置统计**: 重置系统统计信息

### 控制台模式操作

#### 键盘控制
- `q` - 退出程序
- `1` - 切换到完整显示模式
- `2` - 切换到仅姿势检测模式
- `3` - 切换到仅距离估算模式
- `4` - 切换到仅手势识别模式
- `d` - 切换调试信息显示
- `s` - 显示统计信息
- `r` - 重置统计信息

## 项目结构

```
launch/
├── main_integrated.py      # 🔥 合并后的主系统文件
├── test_merged.py         # 测试脚本
└── README_merged.md       # 本说明文件

modules/                   # 功能模块
├── image_processing/
├── pose_detection/
├── distance_estimation/
└── gesture_recognition/
```

## 性能优化

### 1. 调整模型复杂度
在代码中可以修改 `model_complexity` 参数：
```python
self.pose_detector = PoseDetector(model_complexity=0)  # 0=快速, 1=平衡, 2=精确
```

### 2. 调整分辨率
降低分辨率可以提高处理速度：
```bash
python main_integrated.py --console --width 1280 --height 720
```

提示（Windows/USB 摄像头）：主程序启动时会优先申请 MJPG 编码，并尝试设置到 1920x1080；
若驱动或设备不支持，会回退到摄像头能提供的实际分辨率（日志会提示实际值）。

 
### 3. 关闭调试信息

在GUI中取消勾选"显示调试信息"，或在控制台模式按 `d` 键。

 
## 故障排除

 
### 常见问题

#### 1. 摄像头无法启动

- 检查摄像头是否被其他程序占用
- 尝试使用不同的摄像头ID: `--camera 1`
- 确认摄像头驱动正常

#### 2. 模块导入错误

```bash
# 确保在正确的目录运行
cd d:\Musii-SnapShot\GithubRepo\Gesture-DetectionRecognition
python launch/main_integrated.py
```

#### 3. 性能问题

- 降低分辨率
- 减少模型复杂度
- 关闭不必要的显示模式

#### 4. GUI无法启动

- 确认tkinter已安装: `python -c "import tkinter"`
- 尝试控制台模式: `--console`

### 系统需求检查

运行内置的系统需求检查：

```bash
python launch/main_integrated.py --debug
```

 
## 开发和扩展

### 添加新手势

1. 修改 `gesture_recognition/gesture_recognizer.py`
2. 在 `_execute_command` 方法中添加新的控制逻辑
3. 更新手势说明文档

### 集成无人机控制

1. 修改 `_execute_command` 方法
2. 添加实际的无人机控制代码
3. 测试控制指令的可靠性

## 版本信息

- **合并版本**: 2024.12
- **基于**: gui_system.py + main_integrated.py
- **Python版本**: 3.7+
- **主要依赖**: OpenCV, MediaPipe, NumPy, Pillow, Tkinter

## 许可证

请参考项目根目录的LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

🚀 **快速开始**: `python launch/main_integrated.py --gui`
