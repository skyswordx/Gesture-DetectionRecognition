# 距离估算模块使用说明

## 模块概述

距离估算模块基于人体关键点进行距离测量，支持多种估算方法的融合、卡尔曼滤波平滑和相机标定，为后续的控制决策提供准确的距离信息。

## 主要功能

### 1. 距离估算 (DistanceEstimator)
- 多种估算方法融合
- 卡尔曼滤波平滑
- 异常值检测和过滤
- 置信度评估
- 性能统计分析

### 2. 相机标定 (CameraCalibration)  
- 焦距参数管理
- 人体测量标准
- 自动标定功能
- 视野角度计算

### 3. 卡尔曼滤波 (KalmanFilter)
- 状态预测和更新
- 噪声抑制
- 速度估算
- 平滑处理

### 4. 可视化 (DistanceVisualizer)
- 距离信息显示
- 测量详情展示
- 距离指示器
- 历史趋势图

## 估算方法

### 1. 基于肩宽估算
- **原理**: 利用已知的平均肩宽(45cm)
- **公式**: 距离 = (真实肩宽 × 焦距) / 像素肩宽  
- **优点**: 最可靠，受姿势影响小
- **适用**: 正面或斜面姿势

### 2. 基于头部高度估算
- **原理**: 利用头部的平均高度(23cm)
- **计算**: 通过鼻子到嘴角距离估算头高
- **优点**: 头部特征明显
- **适用**: 头部清晰可见时

### 3. 基于身体高度估算  
- **原理**: 利用平均身高(170cm)
- **计算**: 头顶到脚底的像素距离
- **优点**: 测量范围大，精度较高
- **适用**: 全身可见时

### 4. 基于手臂跨度估算
- **原理**: 利用平均臂展(168cm)  
- **计算**: 左右手腕之间的距离
- **优点**: 张开双臂时精度较高
- **适用**: 特定手势时

## 使用方法

### 基础使用
```python
from modules.distance_estimation.distance_estimator import DistanceEstimator

# 创建估算器
estimator = DistanceEstimator()

# 估算距离
result = estimator.estimate_distance(landmarks, frame_width, frame_height)

print(f"距离: {result.distance:.2f}m")
print(f"置信度: {result.confidence:.2f}")
print(f"方法: {result.method}")
```

### 自定义相机参数
```python
from modules.distance_estimation.distance_estimator import CameraCalibration, DistanceEstimator

# 自定义相机标定
calibration = CameraCalibration(
    focal_length=600.0,      # 焦距
    sensor_width_mm=6.17     # 传感器宽度
)

# 更新人体测量参数
calibration.average_shoulder_width = 0.42  # 自定义肩宽

estimator = DistanceEstimator(calibration)
```

### 距离可视化
```python
from modules.distance_estimation.distance_estimator import DistanceVisualizer

visualizer = DistanceVisualizer()

# 在图像上绘制距离信息
output_image = visualizer.draw_distance_info(
    image, distance_result, landmarks, bbox
)
```

### 统计信息
```python
# 获取性能统计
stats = estimator.get_statistics()
print(f"成功率: {stats['success_rate']:.1f}%")
print(f"平均距离: {stats['average_distance']:.2f}m")
print(f"当前速度: {stats['velocity']:.2f}m/s")
```

## 验证程序

### 运行方式
```bash
cd modules/distance_estimation  
python distance_estimator.py
```

### 交互功能
- `q`: 退出程序
- `s`: 显示统计信息
- `r`: 重置卡尔曼滤波器
- `c`: 进行相机标定

### 输出信息
程序实时显示：
- 当前估算距离
- 估算置信度
- 使用的主要方法
- 各种方法的原始测量值
- 右侧距离指示器

## 数据结构

### DistanceResult
```python
@dataclass
class DistanceResult:
    distance: float              # 估算距离(米)
    confidence: float            # 估算置信度(0-1)
    method: str                  # 主要估算方法
    raw_measurements: Dict       # 原始测量值
    timestamp: float             # 时间戳
```

### 测量方法权重
```python
weights = {
    'shoulder': 0.4,    # 肩宽 - 最可靠
    'body': 0.3,        # 身体高度 - 较可靠  
    'head': 0.2,        # 头部高度 - 一般
    'arm_span': 0.1     # 手臂跨度 - 最不可靠
}
```

## 标定参数

### 默认人体测量值
| 参数 | 默认值 | 说明 |
|------|--------|------|
| 肩宽 | 45cm | 成年人平均肩宽 |
| 头高 | 23cm | 头部平均高度 |
| 身高 | 170cm | 成年人平均身高 |
| 臂展 | 168cm | 双臂展开宽度 |
| 腿长 | 87cm | 腿部长度 |

### 相机参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| 焦距 | 500像素 | 等效焦距 |
| 传感器宽度 | 6.17mm | 典型手机传感器 |
| 视野角度 | 60° | 水平视野角 |

## 精度分析

### 距离精度对比
| 距离范围 | 估算精度 | 推荐方法 |
|----------|----------|----------|
| 0.5-2m | ±10cm | 肩宽 + 头部 |
| 2-5m | ±20cm | 肩宽 + 身体高度 |
| 5-10m | ±50cm | 身体高度 |
| >10m | ±1m | 不推荐 |

### 影响因素
1. **姿势影响**: 正面姿势精度最高
2. **光照条件**: 充足光线提高检测精度
3. **遮挡程度**: 关键点可见性影响估算
4. **人体差异**: 个体差异影响标准值适用性

## 常见问题

### 1. 距离跳动严重
- 启用卡尔曼滤波平滑
- 调整滤波器参数
- 检查关键点检测稳定性

### 2. 估算误差大
- 进行相机标定
- 根据目标人群调整人体参数
- 使用多种方法融合

### 3. 置信度过低
- 改善光照条件
- 确保关键点可见
- 调整可见性阈值

### 4. 特定距离偏差
- 使用已知距离校准
- 检查相机畸变
- 调整焦距参数

## 高级功能

### 自动标定
```python
# 在已知距离下自动校准
def calibrate_with_known_distance(estimator, landmarks, actual_distance):
    # 基于实际距离反推焦距
    shoulder_width_px = get_shoulder_width_pixels(landmarks)
    new_focal_length = (shoulder_width * actual_distance) / shoulder_width_px
    estimator.calibration.focal_length = new_focal_length
```

### 个性化参数
```python
# 根据特定人群调整参数
def set_person_specific_params(calibration, age_group="adult"):
    if age_group == "child":
        calibration.average_shoulder_width = 0.30
        calibration.average_body_height = 1.20
    elif age_group == "elderly":  
        calibration.average_shoulder_width = 0.42
        calibration.average_body_height = 1.65
```

### 多人距离估算
```python
# 扩展支持多人距离估算
def estimate_multiple_distances(estimator, multiple_landmarks):
    results = []
    for landmarks in multiple_landmarks:
        result = estimator.estimate_distance(landmarks, width, height)
        results.append(result)
    return results
```

## 技术规格

- **有效距离范围**: 0.5-10米
- **最佳精度范围**: 1-5米  
- **处理速度**: 1000+ FPS (纯计算)
- **内存占用**: < 10MB
- **依赖库**: NumPy, OpenCV
- **线程安全**: 是

## 输出示例

### 距离结果示例
```python
DistanceResult(
    distance=2.35,           # 2.35米
    confidence=0.87,         # 87%置信度
    method='shoulder',       # 主要基于肩宽
    raw_measurements={       # 原始测量
        'shoulder': 2.34,
        'head': 2.41,
        'body': 2.38
    },
    timestamp=1234567890.123
)
```

### 统计信息示例
```python
{
    "total_estimates": 1500,      # 总估算次数
    "successful_estimates": 1425,  # 成功次数
    "success_rate": 95.0,         # 成功率
    "current_distance": 2.35,     # 当前距离
    "average_distance": 2.41,     # 平均距离
    "distance_std": 0.12,         # 距离标准差
    "velocity": -0.05             # 移动速度(m/s)
}
```

## 扩展接口

模块提供标准化接口供其他模块调用：

```python
# 距离估算接口
def estimate_distance(landmarks, width, height) -> DistanceResult

# 统计信息接口
def get_statistics() -> Dict

# 标定接口  
def calibrate_camera(actual_distance, landmarks)

# 可视化接口
def visualize_distance(image, result) -> np.ndarray
```
