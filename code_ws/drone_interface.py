"""
无人机通信接口实现
支持MAVLink协议和模拟控制
"""

import time
import threading
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("Warning: pymavlink not available. Using simulation mode.")

logger = logging.getLogger(__name__)

class DroneState(Enum):
    """无人机状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    ERROR = "error"

@dataclass
class DroneStatus:
    """无人机状态信息"""
    state: DroneState
    battery_level: float
    altitude: float
    position: Tuple[float, float, float]  # x, y, z
    velocity: Tuple[float, float, float]  # vx, vy, vz
    heading: float
    is_armed: bool
    flight_mode: str

class MAVLinkDroneInterface:
    """基于MAVLink的无人机接口"""
    
    def __init__(self, connection_string: str = "127.0.0.1:14550"):
        self.connection_string = connection_string
        self.master = None
        self.is_connected = False
        self.current_status = DroneStatus(
            state=DroneState.DISCONNECTED,
            battery_level=0.0,
            altitude=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            heading=0.0,
            is_armed=False,
            flight_mode="UNKNOWN"
        )
        
        # 状态监控线程
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 安全参数
        self.max_altitude = 10.0  # 最大高度(米)
        self.max_velocity = 5.0   # 最大速度(m/s)
        self.safe_battery_level = 20.0  # 安全电量百分比
        
    def connect(self) -> bool:
        """连接到无人机"""
        if not MAVLINK_AVAILABLE:
            logger.warning("MAVLink不可用，使用模拟模式")
            return self._simulate_connection()
        
        try:
            logger.info(f"连接到无人机: {self.connection_string}")
            self.master = mavutil.mavlink_connection(self.connection_string)
            
            # 等待心跳包
            logger.info("等待心跳包...")
            self.master.wait_heartbeat(timeout=10)
            
            self.is_connected = True
            self.current_status.state = DroneState.CONNECTED
            
            # 启动状态监控
            self._start_monitoring()
            
            logger.info("无人机连接成功")
            return True
            
        except Exception as e:
            logger.error(f"无人机连接失败: {e}")
            return False
    
    def _simulate_connection(self) -> bool:
        """模拟连接 (用于测试)"""
        self.is_connected = True
        self.current_status.state = DroneState.CONNECTED
        self.current_status.battery_level = 85.0
        self._start_monitoring()
        logger.info("模拟连接成功")
        return True
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        if self.master:
            self.master.close()
        
        self.current_status.state = DroneState.DISCONNECTED
        logger.info("无人机连接已断开")
    
    def _start_monitoring(self):
        """启动状态监控线程"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """状态监控循环"""
        logger.info("状态监控线程启动")
        
        while self.is_monitoring and self.is_connected:
            try:
                if MAVLINK_AVAILABLE and self.master:
                    self._update_real_status()
                else:
                    self._update_simulated_status()
                
                time.sleep(0.1)  # 10Hz更新频率
                
            except Exception as e:
                logger.error(f"状态监控错误: {e}")
                time.sleep(1.0)
        
        logger.info("状态监控线程结束")
    
    def _update_real_status(self):
        """更新真实无人机状态"""
        try:
            # 获取心跳包
            msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                self.current_status.is_armed = (msg.base_mode & 
                    mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                self.current_status.flight_mode = mavutil.mode_string_v10(msg)
            
            # 获取位置信息
            pos_msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
            if pos_msg:
                self.current_status.position = (pos_msg.x, pos_msg.y, -pos_msg.z)
                self.current_status.velocity = (pos_msg.vx, pos_msg.vy, pos_msg.vz)
                self.current_status.altitude = -pos_msg.z
            
            # 获取电池信息
            battery_msg = self.master.recv_match(type='BATTERY_STATUS', blocking=False)
            if battery_msg:
                self.current_status.battery_level = battery_msg.battery_remaining
            
            # 获取姿态信息
            attitude_msg = self.master.recv_match(type='ATTITUDE', blocking=False)
            if attitude_msg:
                self.current_status.heading = attitude_msg.yaw * 180 / 3.14159  # 转换为度
            
            # 更新状态
            if self.current_status.is_armed:
                if self.current_status.altitude > 0.5:
                    self.current_status.state = DroneState.FLYING
                else:
                    self.current_status.state = DroneState.ARMED
            else:
                self.current_status.state = DroneState.CONNECTED
                
        except Exception as e:
            logger.error(f"状态更新错误: {e}")
    
    def _update_simulated_status(self):
        """更新模拟状态"""
        # 模拟电池消耗
        if self.current_status.state == DroneState.FLYING:
            self.current_status.battery_level -= 0.01  # 每秒消耗0.1%
        
        # 模拟位置变化(简单的随机漂移)
        import random
        drift = 0.01
        x, y, z = self.current_status.position
        self.current_status.position = (
            x + random.uniform(-drift, drift),
            y + random.uniform(-drift, drift),
            max(0, z + random.uniform(-drift, drift))
        )
        
        self.current_status.altitude = self.current_status.position[2]
    
    def arm(self) -> bool:
        """解锁无人机"""
        if not self.is_connected:
            logger.error("无人机未连接")
            return False
        
        try:
            if MAVLINK_AVAILABLE and self.master:
                # 发送解锁命令
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 0, 0, 0, 0, 0, 0
                )
                
                # 等待确认
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
                if msg and msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.info("无人机解锁成功")
                    return True
                else:
                    logger.error("无人机解锁失败")
                    return False
            else:
                # 模拟解锁
                self.current_status.is_armed = True
                self.current_status.state = DroneState.ARMED
                logger.info("模拟解锁成功")
                return True
                
        except Exception as e:
            logger.error(f"解锁失败: {e}")
            return False
    
    def disarm(self) -> bool:
        """锁定无人机"""
        if not self.is_connected:
            logger.error("无人机未连接")
            return False
        
        try:
            if MAVLINK_AVAILABLE and self.master:
                # 发送锁定命令
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                
                # 等待确认
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
                if msg and msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.info("无人机锁定成功")
                    return True
                else:
                    logger.error("无人机锁定失败")
                    return False
            else:
                # 模拟锁定
                self.current_status.is_armed = False
                self.current_status.state = DroneState.CONNECTED
                logger.info("模拟锁定成功")
                return True
                
        except Exception as e:
            logger.error(f"锁定失败: {e}")
            return False
    
    def takeoff(self, altitude: float = 2.0) -> bool:
        """起飞到指定高度"""
        if not self.current_status.is_armed:
            logger.error("无人机未解锁")
            return False
        
        # 安全检查
        if altitude > self.max_altitude:
            logger.error(f"起飞高度超过最大限制: {altitude} > {self.max_altitude}")
            return False
        
        if self.current_status.battery_level < self.safe_battery_level:
            logger.error(f"电量不足: {self.current_status.battery_level}%")
            return False
        
        try:
            if MAVLINK_AVAILABLE and self.master:
                # 发送起飞命令
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0, 0, 0, 0, 0, 0, 0, altitude
                )
                
                # 等待确认
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)
                if msg and msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self.current_status.state = DroneState.FLYING
                    logger.info(f"起飞到 {altitude}m")
                    return True
                else:
                    logger.error("起飞命令被拒绝")
                    return False
            else:
                # 模拟起飞
                self.current_status.state = DroneState.FLYING
                x, y, _ = self.current_status.position
                self.current_status.position = (x, y, altitude)
                self.current_status.altitude = altitude
                logger.info(f"模拟起飞到 {altitude}m")
                return True
                
        except Exception as e:
            logger.error(f"起飞失败: {e}")
            return False
    
    def land(self) -> bool:
        """降落"""
        if self.current_status.state != DroneState.FLYING:
            logger.error("无人机未在飞行状态")
            return False
        
        try:
            if MAVLINK_AVAILABLE and self.master:
                # 发送降落命令
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_LAND,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                
                # 等待确认
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
                if msg and msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self.current_status.state = DroneState.LANDING
                    logger.info("开始降落")
                    return True
                else:
                    logger.error("降落命令被拒绝")
                    return False
            else:
                # 模拟降落
                self.current_status.state = DroneState.LANDING
                x, y, _ = self.current_status.position
                self.current_status.position = (x, y, 0.0)
                self.current_status.altitude = 0.0
                logger.info("模拟降落")
                
                # 模拟降落过程
                time.sleep(2)
                self.current_status.state = DroneState.ARMED
                return True
                
        except Exception as e:
            logger.error(f"降落失败: {e}")
            return False
    
    def move_velocity(self, vx: float, vy: float, vz: float, duration: float = 1.0) -> bool:
        """以指定速度移动"""
        if self.current_status.state != DroneState.FLYING:
            logger.error("无人机未在飞行状态")
            return False
        
        # 安全检查
        speed = (vx**2 + vy**2 + vz**2)**0.5
        if speed > self.max_velocity:
            # 缩放到最大速度
            scale = self.max_velocity / speed
            vx, vy, vz = vx * scale, vy * scale, vz * scale
            logger.warning(f"速度超限，已缩放: {speed:.2f} -> {self.max_velocity}")
        
        # 高度安全检查
        current_altitude = self.current_status.altitude
        predicted_altitude = current_altitude + vz * duration
        
        if predicted_altitude > self.max_altitude:
            vz = (self.max_altitude - current_altitude) / duration
            logger.warning(f"高度限制，调整vz: {vz:.2f}")
        elif predicted_altitude < 0:
            vz = -current_altitude / duration
            logger.warning(f"防止撞地，调整vz: {vz:.2f}")
        
        try:
            if MAVLINK_AVAILABLE and self.master:
                # 发送速度指令
                self.master.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b110111000111,  # type_mask (仅使用速度)
                    0, 0, 0,  # position (忽略)
                    vx, vy, vz,  # velocity
                    0, 0, 0,  # acceleration (忽略)
                    0, 0  # yaw, yaw_rate (忽略)
                )
                
                logger.info(f"设置速度: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
                return True
            else:
                # 模拟移动
                x, y, z = self.current_status.position
                new_x = x + vx * duration
                new_y = y + vy * duration
                new_z = max(0, z + vz * duration)
                
                self.current_status.position = (new_x, new_y, new_z)
                self.current_status.altitude = new_z
                self.current_status.velocity = (vx, vy, vz)
                
                logger.info(f"模拟移动: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"移动失败: {e}")
            return False
    
    def hover(self) -> bool:
        """悬停"""
        return self.move_velocity(0, 0, 0, 0.1)
    
    def emergency_stop(self) -> bool:
        """紧急停止"""
        logger.warning("执行紧急停止")
        
        try:
            # 立即悬停
            if not self.hover():
                logger.error("紧急停止失败")
                return False
            
            # 可以添加更多紧急处理逻辑
            # 比如自动返航、紧急降落等
            
            return True
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
            return False
    
    def get_status(self) -> DroneStatus:
        """获取当前状态"""
        return self.current_status
    
    def is_safe_to_fly(self) -> Tuple[bool, str]:
        """检查是否安全飞行"""
        if not self.is_connected:
            return False, "无人机未连接"
        
        if self.current_status.battery_level < self.safe_battery_level:
            return False, f"电量不足: {self.current_status.battery_level}%"
        
        if self.current_status.state == DroneState.ERROR:
            return False, "无人机状态异常"
        
        return True, "安全"

class DroneCommandExecutor:
    """无人机指令执行器"""
    
    def __init__(self, drone_interface: MAVLinkDroneInterface):
        self.drone = drone_interface
        self.is_executing = False
        self.current_command = None
        
        # 指令映射
        self.command_map = {
            "takeoff": self._execute_takeoff,
            "land": self._execute_land,
            "move": self._execute_move,
            "hover": self._execute_hover,
            "emergency_stop": self._execute_emergency_stop
        }
    
    def execute_command(self, command_data: Dict) -> bool:
        """执行指令"""
        action = command_data.get("action")
        params = command_data.get("params", {})
        
        if action not in self.command_map:
            logger.error(f"未知指令: {action}")
            return False
        
        # 安全检查
        safe, reason = self.drone.is_safe_to_fly()
        if not safe and action not in ["land", "emergency_stop"]:
            logger.error(f"飞行不安全: {reason}")
            return False
        
        try:
            self.is_executing = True
            self.current_command = command_data
            
            # 执行指令
            result = self.command_map[action](params)
            
            self.is_executing = False
            self.current_command = None
            
            return result
            
        except Exception as e:
            logger.error(f"指令执行异常: {e}")
            self.is_executing = False
            self.current_command = None
            return False
    
    def _execute_takeoff(self, params: Dict) -> bool:
        """执行起飞"""
        altitude = params.get("altitude", 2.0)
        
        # 首先解锁
        if not self.drone.current_status.is_armed:
            if not self.drone.arm():
                return False
            time.sleep(1)  # 等待解锁完成
        
        # 起飞
        return self.drone.takeoff(altitude)
    
    def _execute_land(self, params: Dict) -> bool:
        """执行降落"""
        return self.drone.land()
    
    def _execute_move(self, params: Dict) -> bool:
        """执行移动"""
        vx = params.get("vx", 0.0)
        vy = params.get("vy", 0.0)
        vz = params.get("vz", 0.0)
        duration = params.get("duration", 1.0)
        
        return self.drone.move_velocity(vx, vy, vz, duration)
    
    def _execute_hover(self, params: Dict) -> bool:
        """执行悬停"""
        return self.drone.hover()
    
    def _execute_emergency_stop(self, params: Dict) -> bool:
        """执行紧急停止"""
        return self.drone.emergency_stop()

# 使用示例
def test_drone_interface():
    """测试无人机接口"""
    # 创建接口 (使用模拟模式)
    drone = MAVLinkDroneInterface("simulation")
    executor = DroneCommandExecutor(drone)
    
    try:
        # 连接
        if not drone.connect():
            print("连接失败")
            return
        
        print("连接成功，当前状态:", drone.get_status())
        
        # 测试指令序列
        commands = [
            {"action": "takeoff", "params": {"altitude": 3.0}},
            {"action": "move", "params": {"vx": 1.0, "vy": 0, "vz": 0, "duration": 2.0}},
            {"action": "move", "params": {"vx": 0, "vy": 1.0, "vz": 0, "duration": 2.0}},
            {"action": "hover", "params": {}},
            {"action": "land", "params": {}}
        ]
        
        for cmd in commands:
            print(f"\n执行指令: {cmd}")
            success = executor.execute_command(cmd)
            print(f"执行结果: {'成功' if success else '失败'}")
            print(f"当前状态: {drone.get_status()}")
            time.sleep(2)
    
    finally:
        drone.disconnect()

if __name__ == "__main__":
    test_drone_interface()
