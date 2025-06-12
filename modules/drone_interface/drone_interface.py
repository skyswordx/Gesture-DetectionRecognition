"""
无人机接口模块
提供抽象的无人机控制接口，支持多种无人机平台
"""

import time
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

class DroneState(Enum):
    """无人机状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected" 
    ARMED = "armed"
    FLYING = "flying"  
    HOVERING = "hovering"
    LANDING = "landing"
    EMERGENCY = "emergency"

@dataclass
class DronePosition:
    """无人机位置信息"""
    x: float
    y: float
    z: float
    yaw: float

@dataclass
class DroneStatus:
    """无人机状态信息"""
    state: DroneState
    position: DronePosition
    battery_level: float
    connection_strength: float
    is_armed: bool
    altitude: float
    speed: float

class BaseDroneInterface(ABC):
    """无人机接口基类"""
    
    def __init__(self):
        self.state = DroneState.DISCONNECTED
        self.status_callbacks = []
        self.command_queue = []
        self.last_status = None
        
    @abstractmethod
    def connect(self) -> bool:
        """连接无人机"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开无人机连接"""
        pass
    
    @abstractmethod
    def takeoff(self, altitude: float = 1.0) -> bool:
        """起飞"""
        pass
    
    @abstractmethod
    def land(self) -> bool:
        """降落"""
        pass
    
    @abstractmethod
    def move_forward(self, speed: float = 0.5) -> bool:
        """前进"""
        pass
    
    @abstractmethod
    def move_backward(self, speed: float = 0.5) -> bool:
        """后退"""
        pass
    
    @abstractmethod
    def move_left(self, speed: float = 0.5) -> bool:
        """左移"""
        pass
    
    @abstractmethod
    def move_right(self, speed: float = 0.5) -> bool:
        """右移"""
        pass
    
    @abstractmethod
    def move_up(self, speed: float = 0.5) -> bool:
        """上升"""
        pass
    
    @abstractmethod
    def move_down(self, speed: float = 0.5) -> bool:
        """下降"""
        pass
    
    @abstractmethod
    def rotate_left(self, speed: float = 0.5) -> bool:
        """左转"""
        pass
    
    @abstractmethod
    def rotate_right(self, speed: float = 0.5) -> bool:
        """右转"""
        pass
    
    @abstractmethod
    def hover(self) -> bool:
        """悬停"""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """紧急停止"""
        pass
    
    @abstractmethod
    def get_status(self) -> DroneStatus:
        """获取无人机状态"""
        pass
    
    def add_status_callback(self, callback: Callable[[DroneStatus], None]):
        """添加状态回调函数"""
        self.status_callbacks.append(callback)
    
    def _notify_status_change(self, status: DroneStatus):
        """通知状态变化"""
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"状态回调错误: {e}")

class SimulatedDroneInterface(BaseDroneInterface):
    """模拟无人机接口 - 用于演示和测试"""
    
    def __init__(self):
        super().__init__()
        self.position = DronePosition(0.0, 0.0, 0.0, 0.0)
        self.battery_level = 100.0
        self.is_armed = False
        self.target_altitude = 0.0
        self.current_speed = 0.0
        self._simulation_thread = None
        self._simulation_running = False
        
        logger.info("模拟无人机接口初始化完成")
    
    def connect(self) -> bool:
        """连接模拟无人机"""
        logger.info("连接模拟无人机...")
        time.sleep(1)  # 模拟连接延迟
        
        self.state = DroneState.CONNECTED
        self._start_simulation()
        
        logger.info("✅ 模拟无人机连接成功")
        return True
    
    def disconnect(self):
        """断开模拟无人机连接"""
        logger.info("断开模拟无人机连接...")
        
        self.state = DroneState.DISCONNECTED
        self._stop_simulation()
        
        logger.info("模拟无人机已断开连接")
    
    def takeoff(self, altitude: float = 1.0) -> bool:
        """模拟起飞"""
        if self.state != DroneState.CONNECTED:
            logger.warning("无人机未连接，无法起飞")
            return False
        
        logger.info(f"🚁 模拟起飞到高度 {altitude}m...")
        
        self.state = DroneState.FLYING
        self.target_altitude = altitude
        self.is_armed = True
        
        # 模拟起飞过程
        start_altitude = self.position.z
        for i in range(10):
            self.position.z = start_altitude + (altitude - start_altitude) * (i + 1) / 10
            time.sleep(0.1)
        
        self.state = DroneState.HOVERING
        logger.info(f"✅ 起飞完成，当前高度: {self.position.z:.1f}m")
        return True
    
    def land(self) -> bool:
        """模拟降落"""
        if self.state not in [DroneState.FLYING, DroneState.HOVERING]:
            logger.warning("无人机未在飞行状态，无法降落")
            return False
        
        logger.info("🛬 模拟降落...")
        
        self.state = DroneState.LANDING
        
        # 模拟降落过程
        start_altitude = self.position.z
        for i in range(10):
            self.position.z = start_altitude * (1 - (i + 1) / 10)
            time.sleep(0.1)
        
        self.position.z = 0.0
        self.state = DroneState.CONNECTED
        self.is_armed = False
        
        logger.info("✅ 降落完成")
        return True
    
    def move_forward(self, speed: float = 0.5) -> bool:
        """模拟前进"""
        if not self._can_move():
            return False
        
        logger.info(f"⬆️ 模拟前进，速度: {speed}")
        self.position.y += speed * 0.5  # 模拟移动
        self.current_speed = speed
        return True
    
    def move_backward(self, speed: float = 0.5) -> bool:
        """模拟后退"""
        if not self._can_move():
            return False
        
        logger.info(f"⬇️ 模拟后退，速度: {speed}")
        self.position.y -= speed * 0.5
        self.current_speed = speed
        return True
    
    def move_left(self, speed: float = 0.5) -> bool:
        """模拟左移"""
        if not self._can_move():
            return False
        
        logger.info(f"⬅️ 模拟左移，速度: {speed}")
        self.position.x -= speed * 0.5
        self.current_speed = speed
        return True
    
    def move_right(self, speed: float = 0.5) -> bool:
        """模拟右移"""
        if not self._can_move():
            return False
        
        logger.info(f"➡️ 模拟右移，速度: {speed}")
        self.position.x += speed * 0.5
        self.current_speed = speed
        return True
    
    def move_up(self, speed: float = 0.5) -> bool:
        """模拟上升"""
        if not self._can_move():
            return False
        
        logger.info(f"⬆️ 模拟上升，速度: {speed}")
        self.position.z += speed * 0.3
        self.current_speed = speed
        return True
    
    def move_down(self, speed: float = 0.5) -> bool:
        """模拟下降"""
        if not self._can_move():
            return False
        
        if self.position.z <= 0.2:  # 防止撞地
            logger.warning("高度过低，停止下降")
            return False
        
        logger.info(f"⬇️ 模拟下降，速度: {speed}")
        self.position.z -= speed * 0.3
        self.current_speed = speed
        return True
    
    def rotate_left(self, speed: float = 0.5) -> bool:
        """模拟左转"""
        if not self._can_move():
            return False
        
        logger.info(f"↺ 模拟左转，速度: {speed}")
        self.position.yaw -= speed * 30  # 度数
        if self.position.yaw < 0:
            self.position.yaw += 360
        return True
    
    def rotate_right(self, speed: float = 0.5) -> bool:
        """模拟右转"""
        if not self._can_move():
            return False
        
        logger.info(f"↻ 模拟右转，速度: {speed}")
        self.position.yaw += speed * 30
        if self.position.yaw >= 360:
            self.position.yaw -= 360
        return True
    
    def hover(self) -> bool:
        """模拟悬停"""
        if self.state != DroneState.FLYING:
            logger.warning("无人机未在飞行状态，无法悬停")
            return False
        
        logger.info("⏸️ 模拟悬停")
        self.state = DroneState.HOVERING
        self.current_speed = 0.0
        return True
    
    def emergency_stop(self) -> bool:
        """模拟紧急停止"""
        logger.warning("🚨 紧急停止！")
        
        self.state = DroneState.EMERGENCY
        self.current_speed = 0.0
        
        # 紧急降落
        threading.Thread(target=self._emergency_landing, daemon=True).start()
        return True
    
    def get_status(self) -> DroneStatus:
        """获取模拟无人机状态"""
        return DroneStatus(
            state=self.state,
            position=self.position,
            battery_level=self.battery_level,
            connection_strength=1.0,  # 模拟满信号
            is_armed=self.is_armed,
            altitude=self.position.z,
            speed=self.current_speed
        )
    
    def _can_move(self) -> bool:
        """检查是否可以移动"""
        if self.state not in [DroneState.FLYING, DroneState.HOVERING]:
            logger.warning("无人机未在飞行状态，无法移动")
            return False
        return True
    
    def _start_simulation(self):
        """启动模拟线程"""
        self._simulation_running = True
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()
    
    def _stop_simulation(self):
        """停止模拟线程"""
        self._simulation_running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=1.0)
    
    def _simulation_loop(self):
        """模拟循环"""
        while self._simulation_running:
            try:
                # 模拟电池消耗
                if self.state in [DroneState.FLYING, DroneState.HOVERING]:
                    self.battery_level = max(0, self.battery_level - 0.01)  # 每秒消耗0.01%
                
                # 检查低电量
                if self.battery_level < 20 and self.state in [DroneState.FLYING, DroneState.HOVERING]:
                    logger.warning(f"⚠️ 电量不足: {self.battery_level:.1f}%")
                
                if self.battery_level < 5:
                    logger.error("🔋 电量严重不足，强制降落！")
                    self.emergency_stop()
                
                # 通知状态变化
                current_status = self.get_status()
                if current_status != self.last_status:
                    self._notify_status_change(current_status)
                    self.last_status = current_status
                
                time.sleep(1)  # 每秒更新一次
                
            except Exception as e:
                logger.error(f"模拟循环错误: {e}")
    
    def _emergency_landing(self):
        """紧急降落"""
        logger.info("执行紧急降落...")
        
        # 快速降落
        while self.position.z > 0.1:
            self.position.z = max(0, self.position.z - 0.2)
            time.sleep(0.1)
        
        self.position.z = 0.0
        self.state = DroneState.CONNECTED
        self.is_armed = False
        
        logger.info("紧急降落完成")

class TelloDroneInterface(BaseDroneInterface):
    """DJI Tello无人机接口实现"""
    
    def __init__(self):
        super().__init__()
        self.tello = None
        logger.info("Tello无人机接口初始化")
    
    def connect(self) -> bool:
        """连接Tello无人机"""
        try:
            # 这里需要安装 djitellopy 库
            # pip install djitellopy
            from djitellopy import Tello
            
            logger.info("连接Tello无人机...")
            self.tello = Tello()
            self.tello.connect()
            
            # 检查连接状态
            battery = self.tello.get_battery()
            logger.info(f"✅ Tello连接成功，电量: {battery}%")
            
            self.state = DroneState.CONNECTED
            return True
            
        except ImportError:
            logger.error("❌ 请安装 djitellopy 库: pip install djitellopy")
            return False
        except Exception as e:
            logger.error(f"❌ Tello连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开Tello连接"""
        if self.tello:
            self.tello.end()
        self.state = DroneState.DISCONNECTED
        logger.info("Tello已断开连接")
    
    def takeoff(self, altitude: float = 1.0) -> bool:
        """Tello起飞"""
        if not self.tello or self.state != DroneState.CONNECTED:
            return False
        
        try:
            logger.info("🚁 Tello起飞...")
            self.tello.takeoff()
            self.state = DroneState.FLYING
            return True
        except Exception as e:
            logger.error(f"Tello起飞失败: {e}")
            return False
    
    def land(self) -> bool:
        """Tello降落"""
        if not self.tello:
            return False
        
        try:
            logger.info("🛬 Tello降落...")
            self.tello.land()
            self.state = DroneState.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Tello降落失败: {e}")
            return False
    
    def move_forward(self, speed: float = 0.5) -> bool:
        """Tello前进"""
        if not self._can_tello_move():
            return False
        
        try:
            distance = int(speed * 50)  # 转换为cm
            self.tello.move_forward(distance)
            return True
        except Exception as e:
            logger.error(f"Tello前进失败: {e}")
            return False
    
    def move_backward(self, speed: float = 0.5) -> bool:
        """Tello后退"""
        if not self._can_tello_move():
            return False
        
        try:
            distance = int(speed * 50)
            self.tello.move_back(distance)
            return True
        except Exception as e:
            logger.error(f"Tello后退失败: {e}")
            return False
    
    def move_left(self, speed: float = 0.5) -> bool:
        """Tello左移"""
        if not self._can_tello_move():
            return False
        
        try:
            distance = int(speed * 50)
            self.tello.move_left(distance)
            return True
        except Exception as e:
            logger.error(f"Tello左移失败: {e}")
            return False
    
    def move_right(self, speed: float = 0.5) -> bool:
        """Tello右移"""
        if not self._can_tello_move():
            return False
        
        try:
            distance = int(speed * 50)
            self.tello.move_right(distance)
            return True
        except Exception as e:
            logger.error(f"Tello右移失败: {e}")
            return False
    
    def move_up(self, speed: float = 0.5) -> bool:
        """Tello上升"""
        if not self._can_tello_move():
            return False
        
        try:
            distance = int(speed * 30)
            self.tello.move_up(distance)
            return True
        except Exception as e:
            logger.error(f"Tello上升失败: {e}")
            return False
    
    def move_down(self, speed: float = 0.5) -> bool:
        """Tello下降"""
        if not self._can_tello_move():
            return False
        
        try:
            distance = int(speed * 30)
            self.tello.move_down(distance)
            return True
        except Exception as e:
            logger.error(f"Tello下降失败: {e}")
            return False
    
    def rotate_left(self, speed: float = 0.5) -> bool:
        """Tello左转"""
        if not self._can_tello_move():
            return False
        
        try:
            angle = int(speed * 30)
            self.tello.rotate_counter_clockwise(angle)
            return True
        except Exception as e:
            logger.error(f"Tello左转失败: {e}")
            return False
    
    def rotate_right(self, speed: float = 0.5) -> bool:
        """Tello右转"""
        if not self._can_tello_move():
            return False
        
        try:
            angle = int(speed * 30)
            self.tello.rotate_clockwise(angle)
            return True
        except Exception as e:
            logger.error(f"Tello右转失败: {e}")
            return False
    
    def hover(self) -> bool:
        """Tello悬停"""
        # Tello自动悬停，无需特殊指令
        self.state = DroneState.HOVERING
        return True
    
    def emergency_stop(self) -> bool:
        """Tello紧急停止"""
        if not self.tello:
            return False
        
        try:
            logger.warning("🚨 Tello紧急停止！")
            self.tello.emergency()
            self.state = DroneState.EMERGENCY
            return True
        except Exception as e:
            logger.error(f"Tello紧急停止失败: {e}")
            return False
    
    def get_status(self) -> DroneStatus:
        """获取Tello状态"""
        try:
            if not self.tello:
                return DroneStatus(
                    state=self.state,
                    position=DronePosition(0, 0, 0, 0),
                    battery_level=0,
                    connection_strength=0,
                    is_armed=False,
                    altitude=0,
                    speed=0
                )
            
            battery = self.tello.get_battery()
            height = self.tello.get_height()
            
            return DroneStatus(
                state=self.state,
                position=DronePosition(0, 0, height/100, 0),  # Tello返回cm，转换为m
                battery_level=battery,
                connection_strength=1.0,
                is_armed=self.state in [DroneState.FLYING, DroneState.HOVERING],
                altitude=height/100,
                speed=0
            )
            
        except Exception as e:
            logger.error(f"获取Tello状态失败: {e}")
            return DroneStatus(
                state=DroneState.EMERGENCY,
                position=DronePosition(0, 0, 0, 0),
                battery_level=0,
                connection_strength=0,
                is_armed=False,
                altitude=0,
                speed=0
            )
    
    def _can_tello_move(self) -> bool:
        """检查Tello是否可以移动"""
        if not self.tello or self.state not in [DroneState.FLYING, DroneState.HOVERING]:
            logger.warning("Tello未在飞行状态，无法移动")
            return False
        return True

class DroneControlManager:
    """无人机控制管理器"""
    
    def __init__(self, drone_interface: BaseDroneInterface):
        self.drone = drone_interface
        self.command_history = []
        self.safety_enabled = True
        self.max_altitude = 3.0  # 最大飞行高度(米)
        self.min_battery = 20.0  # 最低电量百分比
        
        # 添加状态监控
        self.drone.add_status_callback(self._safety_check)
    
    def execute_gesture_command(self, gesture: str, confidence: float, distance: float) -> bool:
        """执行手势指令"""
        if confidence < 0.8:
            logger.info(f"手势置信度过低({confidence:.2f})，忽略指令: {gesture}")
            return False
        
        # 记录指令
        self.command_history.append({
            'gesture': gesture,
            'confidence': confidence,
            'distance': distance,
            'timestamp': time.time()
        })
        
        # 限制历史记录长度
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-50:]
        
        # 执行安全检查
        if self.safety_enabled and not self._pre_command_safety_check(gesture):
            return False
        
        # 根据距离调整速度
        speed = self._calculate_speed_from_distance(distance)
        
        # 执行指令
        logger.info(f"执行手势指令: {gesture} (置信度: {confidence:.2f}, 距离: {distance:.1f}m, 速度: {speed:.2f})")
        
        success = False
        if gesture == "takeoff":
            success = self.drone.takeoff()
        elif gesture == "landing":
            success = self.drone.land()
        elif gesture == "forward":
            success = self.drone.move_forward(speed)
        elif gesture == "left":
            success = self.drone.move_left(speed)
        elif gesture == "right":
            success = self.drone.move_right(speed)
        elif gesture == "up":
            success = self.drone.move_up(speed)
        elif gesture == "down":
            success = self.drone.move_down(speed)
        elif gesture == "stop":
            success = self.drone.hover()
        else:
            logger.warning(f"未知手势指令: {gesture}")
            return False
        
        if success:
            logger.info(f"✅ 指令执行成功: {gesture}")
        else:
            logger.error(f"❌ 指令执行失败: {gesture}")
        
        return success
    
    def _calculate_speed_from_distance(self, distance: float) -> float:
        """根据距离计算速度"""
        if distance < 2.0:
            return 0.3  # 慢速
        elif distance < 4.0:
            return 0.6  # 中速
        else:
            return 1.0  # 快速
    
    def _pre_command_safety_check(self, gesture: str) -> bool:
        """执行前安全检查"""
        status = self.drone.get_status()
        
        # 检查电量
        if status.battery_level < self.min_battery and gesture != "landing":
            logger.warning(f"电量过低({status.battery_level:.1f}%)，拒绝执行: {gesture}")
            return False
        
        # 检查高度限制
        if gesture == "up" and status.altitude > self.max_altitude:
            logger.warning(f"高度过高({status.altitude:.1f}m)，拒绝上升")
            return False
        
        # 检查紧急状态
        if status.state == DroneState.EMERGENCY:
            logger.warning("无人机处于紧急状态，拒绝所有指令")
            return False
        
        return True
    
    def _safety_check(self, status: DroneStatus):
        """安全检查回调"""
        # 低电量警告
        if status.battery_level < 30:
            logger.warning(f"⚠️ 电量预警: {status.battery_level:.1f}%")
        
        # 高度检查
        if status.altitude > self.max_altitude * 1.2:
            logger.error(f"🚨 高度超限: {status.altitude:.1f}m，执行紧急降落")
            self.drone.emergency_stop()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取控制统计"""
        if not self.command_history:
            return {}
        
        # 统计手势分布
        gesture_counts = {}
        total_commands = len(self.command_history)
        
        for cmd in self.command_history:
            gesture = cmd['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # 平均置信度
        avg_confidence = sum(cmd['confidence'] for cmd in self.command_history) / total_commands
        
        # 最近10条指令
        recent_commands = self.command_history[-10:]
        
        return {
            'total_commands': total_commands,
            'gesture_distribution': gesture_counts,
            'average_confidence': avg_confidence,
            'recent_commands': recent_commands,
            'drone_status': self.drone.get_status(),
            'safety_enabled': self.safety_enabled
        }

def test_drone_interface():
    """测试无人机接口"""
    print("=" * 50)
    print("  无人机接口测试")
    print("=" * 50)
    
    # 测试模拟无人机
    print("测试模拟无人机接口...")
    sim_drone = SimulatedDroneInterface()
    
    if sim_drone.connect():
        # 测试基本功能
        print("测试起飞...")
        sim_drone.takeoff(1.5)
        
        time.sleep(1)
        print("测试移动...")
        sim_drone.move_forward(0.5)
        sim_drone.move_right(0.3)
        sim_drone.hover()
        
        time.sleep(1)
        print("测试降落...")
        sim_drone.land()
        
        print("测试完成")
        sim_drone.disconnect()
    
    print("=" * 50)

if __name__ == "__main__":
    test_drone_interface()
