# 项目快速入门指南

## 🚀 立即开始

### 1. 环境检查
```bash
# 运行系统测试
python tests/test_basic.py
```

### 2. 简单演示
```bash
# 运行简单示例（选择姿势检测或手势识别）
python examples/simple_demo.py
```

### 3. 完整演示
```bash
# 运行完整的手势控制系统
python examples/integrated_demo.py
```

## 📁 项目结构

```
Gesture-DetectionRecognition/
├── 📋 README_Modular.md           # 主要项目文档
├── 📋 requirements.txt            # 依赖包列表
├── 
├── 📁 modules/                    # 核心模块
│   ├── 🖼️  image_processing/      # 图像处理模块
│   │   ├── image_processor.py     # 主要实现
│   │   ├── README.md             # 模块文档
│   │   └── __init__.py           # 包初始化
│   │
│   ├── 🕺 pose_detection/         # 姿势检测模块
│   │   ├── pose_detector.py      # 主要实现
│   │   ├── README.md             # 模块文档
│   │   └── __init__.py           # 包初始化
│   │
│   ├── 📏 distance_estimation/    # 距离估算模块
│   │   ├── distance_estimator.py # 主要实现
│   │   ├── README.md             # 模块文档
│   │   └── __init__.py           # 包初始化
│   │
│   ├── 👋 gesture_recognition/    # 手势识别模块
│   │   ├── gesture_recognizer.py # 主要实现
│   │   ├── README.md             # 模块文档
│   │   └── __init__.py           # 包初始化
│   │
│   └── __init__.py               # 模块包初始化
│
├── 📁 examples/                   # 使用示例
│   ├── simple_demo.py            # 简单入门示例
│   └── integrated_demo.py        # 综合演示
│
├── 📁 tests/                     # 测试代码
│   └── test_basic.py             # 基础功能测试
│
├── 📁 code_ws/                   # 原始代码（保留）
└── 📁 docs/                      # 文档目录
```

## 🎯 模块功能

### 🖼️ 图像处理模块
- **摄像头采集**: 多线程实时采集
- **图像预处理**: 镜像、亮度、对比度调整  
- **质量评估**: 模糊度、亮度检测
- **可视化**: 图像显示和信息叠加

### 🕺 姿势检测模块  
- **关键点检测**: 33个人体关键点
- **姿势分析**: 角度计算、位置分析
- **可视化**: 骨架绘制、关键点显示
- **性能统计**: FPS、成功率统计

### 📏 距离估算模块
- **多方法融合**: 肩宽、头高、身高、臂展
- **卡尔曼滤波**: 平滑距离输出
- **相机标定**: 焦距和人体参数配置
- **可视化**: 距离信息和指示器

### 👋 手势识别模块
- **手势分类**: 起飞、降落、方向、停止
- **时序一致性**: 最小持续时间过滤
- **置信度评估**: 多维度评分融合
- **可视化**: 手势信息和关键点高亮

## ⚡ 性能优化

### 速度优化
```python
# 降低模型复杂度
detector = PoseDetector(model_complexity=0)

# 减小图像尺寸
capture = CameraCapture(width=320, height=240)

# 关闭不必要功能
processor.enable_enhancement = False
```

### 精度优化
```python
# 提高模型复杂度
detector = PoseDetector(model_complexity=2)

# 提高置信度阈值
detector.min_detection_confidence = 0.7

# 启用图像增强
processor.enable_enhancement = True
```

## 🔧 配置参数

### 摄像头配置
```python
capture = CameraCapture(
    camera_id=0,        # 摄像头ID
    width=640,          # 宽度
    height=480,         # 高度
    fps=30             # 帧率
)
```

### 手势识别阈值
```python
# 调整各手势的置信度阈值
recognizer.classifiers['takeoff'].confidence_threshold = 0.8
recognizer.classifiers['landing'].confidence_threshold = 0.8
recognizer.classifiers['stop'].confidence_threshold = 0.7
```

### 距离估算参数
```python
# 调整人体测量标准
calibration.average_shoulder_width = 0.45  # 肩宽(米)
calibration.average_body_height = 1.70     # 身高(米)
calibration.focal_length = 500             # 焦距(像素)
```

## 🎮 使用技巧

### 手势标准
- **起飞**: 双手高举过头，保持1-2秒
- **降落**: 双手向下压，低于腰部
- **方向**: 单手指向对应方向，保持水平
- **停止**: 双手胸前交叉或并拢

### 环境要求
- **光照**: 充足均匀的室内光线
- **背景**: 简洁，避免复杂背景
- **距离**: 1-3米为最佳识别距离
- **姿势**: 尽量保持正面朝向摄像头

### 故障排除
1. **手势识别不准确**: 检查光照、距离和手势标准度
2. **系统运行缓慢**: 降低分辨率或模型复杂度
3. **摄像头无法打开**: 检查设备连接和权限
4. **距离偏差**: 进行相机标定或调整人体参数

## 📚 学习路径

### 初学者
1. 运行 `simple_demo.py` 了解基础功能
2. 阅读各模块的 README.md 文档
3. 修改配置参数观察效果变化
4. 尝试添加自定义手势

### 进阶用户
1. 研究模块间的接口设计
2. 优化算法参数提升性能
3. 扩展支持更多手势类型
4. 集成到实际应用项目

### 开发者
1. 理解模块化架构设计
2. 贡献新功能或优化
3. 编写测试用例
4. 完善文档和示例

## 🤝 获取帮助

- 📖 查看模块文档: `modules/<module_name>/README.md`
- 🧪 运行测试: `python tests/test_basic.py`
- 💡 查看示例: `examples/` 目录
- 🐛 问题反馈: GitHub Issues

---
**开始您的手势识别之旅! 🚀**
