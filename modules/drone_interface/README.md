# 无人机接口模块

## 概述

无人机接口模块提供了抽象的无人机控制接口，支持多种无人机平台，包括模拟无人机和真实硬件无人机（如DJI Tello）。

## 主要特性

- 🚁 **抽象接口设计** - 支持多种无人机平台
- 🎮 **完整控制功能** - 起飞、降落、移动、旋转等
- 🛡️ **安全检查机制** - 电量监控、高度限制、紧急停止
- 📊 **状态监控** - 实时获取无人机状态信息
- 🔧 **模拟模式** - 无需硬件即可测试功能
- 📈 **指令统计** - 记录和分析控制指令

## 类结构

### BaseDroneInterface
抽象基类，定义了无人机控制的标准接口。

### SimulatedDroneInterface
模拟无人机实现，用于演示和测试。

### TelloDroneInterface
DJI Tello无人机的具体实现。

### DroneControlManager
无人机控制管理器，提供安全检查和手势指令处理。

## 使用示例

### 基本使用

```python
from modules.drone_interface import SimulatedDroneInterface, DroneControlManager

# 创建模拟无人机接口
drone = SimulatedDroneInterface()

# 连接无人机
if drone.connect():
    # 起飞
    drone.takeoff(1.0)
    
    # 移动
    drone.move_forward(0.5)
    drone.move_right(0.3)
    
    # 悬停
    drone.hover()
    
    # 降落
    drone.land()
    
    # 断开连接
    drone.disconnect()
```

### 使用控制管理器

```python
from modules.drone_interface import SimulatedDroneInterface, DroneControlManager

# 创建无人机和管理器
drone = SimulatedDroneInterface()
manager = DroneControlManager(drone)

# 连接无人机
drone.connect()

# 执行手势指令
success = manager.execute_gesture_command(
    gesture="takeoff",
    confidence=0.9,
    distance=2.5
)

if success:
    print("指令执行成功")

# 获取统计信息
stats = manager.get_statistics()
print(f"总指令数: {stats['total_commands']}")
```

### Tello无人机使用

```python
from modules.drone_interface import TelloDroneInterface

# 注意：需要先安装 djitellopy 库
# pip install djitellopy

# 创建Tello接口
tello = TelloDroneInterface()

# 连接Tello
if tello.connect():
    print("Tello连接成功")
    
    # 获取状态
    status = tello.get_status()
    print(f"电量: {status.battery_level}%")
    
    # 控制飞行
    tello.takeoff()
    tello.move_forward(0.5)
    tello.land()
    
    tello.disconnect()
```

## API 参考

### 基本控制方法

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `connect()` | - | bool | 连接无人机 |
| `disconnect()` | - | None | 断开连接 |
| `takeoff(altitude)` | altitude: float = 1.0 | bool | 起飞到指定高度 |
| `land()` | - | bool | 降落 |
| `hover()` | - | bool | 悬停 |
| `emergency_stop()` | - | bool | 紧急停止 |

### 移动控制方法

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `move_forward(speed)` | speed: float = 0.5 | bool | 前进 |
| `move_backward(speed)` | speed: float = 0.5 | bool | 后退 |
| `move_left(speed)` | speed: float = 0.5 | bool | 左移 |
| `move_right(speed)` | speed: float = 0.5 | bool | 右移 |
| `move_up(speed)` | speed: float = 0.5 | bool | 上升 |
| `move_down(speed)` | speed: float = 0.5 | bool | 下降 |
| `rotate_left(speed)` | speed: float = 0.5 | bool | 左转 |
| `rotate_right(speed)` | speed: float = 0.5 | bool | 右转 |

### 状态查询

| 方法 | 返回值 | 描述 |
|------|--------|------|
| `get_status()` | DroneStatus | 获取完整状态信息 |

### DroneStatus 数据结构

```python
@dataclass
class DroneStatus:
    state: DroneState           # 无人机状态
    position: DronePosition     # 位置信息
    battery_level: float        # 电量百分比
    connection_strength: float  # 连接强度
    is_armed: bool             # 是否解锁
    altitude: float            # 当前高度
    speed: float               # 当前速度
```

### DroneState 枚举

- `DISCONNECTED` - 未连接
- `CONNECTED` - 已连接
- `ARMED` - 已解锁
- `FLYING` - 飞行中
- `HOVERING` - 悬停
- `LANDING` - 降落中
- `EMERGENCY` - 紧急状态

## 安全特性

### 自动安全检查

1. **电量监控** - 低电量时自动警告或强制降落
2. **高度限制** - 防止飞行过高
3. **状态检查** - 确保指令在合适状态下执行
4. **紧急停止** - 任何时候都可以紧急停止

### 配置安全参数

```python
manager = DroneControlManager(drone)
manager.safety_enabled = True      # 启用安全检查
manager.max_altitude = 3.0         # 最大高度3米
manager.min_battery = 20.0         # 最低电量20%
```

## 测试和调试

### 运行测试

```python
# 直接运行模块进行测试
python modules/drone_interface/drone_interface.py
```

### 模拟模式

模拟模式完全在软件中运行，无需真实硬件：

- 模拟飞行物理
- 模拟电池消耗
- 模拟连接状态
- 支持所有控制指令

## 故障排除

### 常见问题

1. **Tello连接失败**
   - 确保安装了 `djitellopy` 库
   - 检查WiFi连接到Tello
   - 确认Tello电量充足

2. **模拟模式无响应**
   - 检查是否调用了 `connect()` 方法
   - 确认系统状态正确

3. **指令执行失败**
   - 检查无人机状态
   - 确认安全检查是否通过
   - 查看日志输出

### 日志配置

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 扩展开发

### 添加新的无人机平台

1. 继承 `BaseDroneInterface`
2. 实现所有抽象方法
3. 添加平台特定的初始化代码
4. 处理平台特定的错误

```python
class MyDroneInterface(BaseDroneInterface):
    def connect(self) -> bool:
        # 实现连接逻辑
        pass
    
    def takeoff(self, altitude: float = 1.0) -> bool:
        # 实现起飞逻辑
        pass
    
    # ... 实现其他方法
```

## 技术规格

- **Python版本**: 3.7+
- **主要依赖**: 无（模拟模式）
- **可选依赖**: djitellopy（Tello支持）
- **线程安全**: 部分支持
- **性能**: 低延迟控制响应

## 更新日志

### v1.0.0
- 初始版本发布
- 支持模拟和Tello无人机
- 完整的安全检查机制
- 手势指令集成
