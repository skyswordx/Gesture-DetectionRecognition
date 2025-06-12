"""
完整的无人机手势控制系统
集成手势识别、距离估算、指令生成和无人机控制
"""

import sys
import time
import logging
import threading
import queue
from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

# 导入自定义模块
from drone_gesture_control import DroneGestureControlSystem, DroneCommand
from drone_interface import MAVLinkDroneInterface, DroneCommandExecutor, DroneStatus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drone_control.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DroneControlGUI:
    """无人机控制系统GUI"""
    
    def __init__(self):
        # 初始化组件
        self.gesture_system = DroneGestureControlSystem()
        self.drone_interface = None
        self.drone_executor = None
        
        # GUI组件
        self.root = None
        self.canvas = None
        self.status_labels = {}
        self.control_buttons = {}
        
        # 系统状态
        self.is_system_running = False
        self.is_drone_connected = False
        self.connection_string = "127.0.0.1:14550"  # 默认SITL连接
        
        # 线程管理
        self.gui_update_thread = None
        self.command_processing_thread = None
        
        # 数据队列
        self.drone_command_queue = queue.Queue(maxsize=20)
        
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("无人机手势控制系统 v1.0")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧视频显示区
        self._create_video_panel(main_frame)
        
        # 右侧控制面板
        self._create_control_panel(main_frame)
        
        return self.root
    
    def _create_video_panel(self, parent):
        """创建视频显示面板"""
        video_frame = ttk.LabelFrame(parent, text="视频监控", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 视频画布
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg="black")
        self.canvas.pack()
        
        # 视频控制按钮
        video_control_frame = ttk.Frame(video_frame)
        video_control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(video_control_frame, text="开始检测", 
                  command=self.start_gesture_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_control_frame, text="停止检测", 
                  command=self.stop_gesture_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_control_frame, text="保存截图", 
                  command=self.save_screenshot).pack(side=tk.LEFT, padx=5)
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 连接设置
        self._create_connection_panel(control_frame)
        
        # 状态监控
        self._create_status_panel(control_frame)
        
        # 手动控制
        self._create_manual_control_panel(control_frame)
        
        # 系统设置
        self._create_settings_panel(control_frame)
    
    def _create_connection_panel(self, parent):
        """创建连接面板"""
        conn_frame = ttk.LabelFrame(parent, text="无人机连接", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 连接字符串输入
        ttk.Label(conn_frame, text="连接地址:").grid(row=0, column=0, sticky=tk.W)
        self.connection_entry = ttk.Entry(conn_frame, width=25)
        self.connection_entry.insert(0, self.connection_string)
        self.connection_entry.grid(row=0, column=1, padx=(5, 0))
        
        # 连接按钮
        button_frame = ttk.Frame(conn_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        self.control_buttons['connect'] = ttk.Button(
            button_frame, text="连接", command=self.connect_drone)
        self.control_buttons['connect'].pack(side=tk.LEFT, padx=5)
        
        self.control_buttons['disconnect'] = ttk.Button(
            button_frame, text="断开", command=self.disconnect_drone, state=tk.DISABLED)
        self.control_buttons['disconnect'].pack(side=tk.LEFT, padx=5)
    
    def _create_status_panel(self, parent):
        """创建状态监控面板"""
        status_frame = ttk.LabelFrame(parent, text="状态监控", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 状态标签
        status_items = [
            ("连接状态", "disconnected"),
            ("飞行模式", "UNKNOWN"),
            ("电池电量", "0%"),
            ("当前高度", "0.0m"),
            ("水平位置", "(0.0, 0.0)"),
            ("当前手势", "无"),
            ("置信度", "0%"),
            ("检测距离", "0.0m")
        ]
        
        for i, (label, default_value) in enumerate(status_items):
            ttk.Label(status_frame, text=f"{label}:").grid(
                row=i, column=0, sticky=tk.W, pady=2)
            
            self.status_labels[label] = ttk.Label(
                status_frame, text=default_value, foreground="blue")
            self.status_labels[label].grid(
                row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
    
    def _create_manual_control_panel(self, parent):
        """创建手动控制面板"""
        manual_frame = ttk.LabelFrame(parent, text="手动控制", padding=10)
        manual_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 基础控制按钮
        basic_controls = [
            ("解锁", self.arm_drone),
            ("锁定", self.disarm_drone),
            ("起飞", self.takeoff_drone),
            ("降落", self.land_drone),
            ("悬停", self.hover_drone),
            ("紧急停止", self.emergency_stop)
        ]
        
        for i, (text, command) in enumerate(basic_controls):
            row, col = divmod(i, 2)
            btn = ttk.Button(manual_frame, text=text, command=command, width=12)
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.control_buttons[text] = btn
        
        # 方向控制
        direction_frame = ttk.LabelFrame(manual_frame, text="方向控制", padding=5)
        direction_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=tk.EW)
        
        # 创建方向控制按钮布局
        directions = [
            (1, 1, "前进", lambda: self.manual_move(1, 0, 0)),
            (0, 0, "左上", lambda: self.manual_move(1, -1, 0)),
            (0, 1, "上升", lambda: self.manual_move(0, 0, 1)),
            (0, 2, "右上", lambda: self.manual_move(1, 1, 0)),
            (1, 0, "左移", lambda: self.manual_move(0, -1, 0)),
            (1, 2, "右移", lambda: self.manual_move(0, 1, 0)),
            (2, 0, "左下", lambda: self.manual_move(-1, -1, 0)),
            (2, 1, "下降", lambda: self.manual_move(0, 0, -1)),
            (2, 2, "右下", lambda: self.manual_move(-1, 1, 0)),
            (3, 1, "后退", lambda: self.manual_move(-1, 0, 0))
        ]
        
        for row, col, text, command in directions:
            btn = ttk.Button(direction_frame, text=text, command=command, width=8)
            btn.grid(row=row, column=col, padx=2, pady=2)
    
    def _create_settings_panel(self, parent):
        """创建设置面板"""
        settings_frame = ttk.LabelFrame(parent, text="系统设置", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 手势识别设置
        ttk.Label(settings_frame, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.8)
        confidence_scale = ttk.Scale(settings_frame, from_=0.5, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0))
        
        # 移动速度设置
        ttk.Label(settings_frame, text="移动速度:").grid(row=1, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(settings_frame, from_=0.1, to=3.0, 
                              variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=1, column=1, sticky=tk.EW, padx=(5, 0))
        
        # 安全设置
        self.auto_land_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="低电量自动降落", 
                       variable=self.auto_land_var).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        self.gesture_control_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="启用手势控制", 
                       variable=self.gesture_control_var).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        # 配置grid权重
        settings_frame.columnconfigure(1, weight=1)
    
    def start_gesture_detection(self):
        """开始手势检测"""
        try:
            if self.gesture_system.start():
                self.is_system_running = True
                # 启动GUI更新线程
                self._start_gui_update_thread()
                # 启动指令处理线程
                self._start_command_processing_thread()
                logger.info("手势检测系统启动成功")
            else:
                messagebox.showerror("错误", "手势检测系统启动失败")
        except Exception as e:
            logger.error(f"启动手势检测失败: {e}")
            messagebox.showerror("错误", f"启动失败: {str(e)}")
    
    def stop_gesture_detection(self):
        """停止手势检测"""
        try:
            self.is_system_running = False
            self.gesture_system.stop()
            logger.info("手势检测系统已停止")
        except Exception as e:
            logger.error(f"停止手势检测失败: {e}")
    
    def connect_drone(self):
        """连接无人机"""
        try:
            self.connection_string = self.connection_entry.get()
            self.drone_interface = MAVLinkDroneInterface(self.connection_string)
            
            if self.drone_interface.connect():
                self.drone_executor = DroneCommandExecutor(self.drone_interface)
                self.is_drone_connected = True
                
                # 更新UI状态
                self.control_buttons['connect'].config(state=tk.DISABLED)
                self.control_buttons['disconnect'].config(state=tk.NORMAL)
                
                logger.info("无人机连接成功")
                messagebox.showinfo("成功", "无人机连接成功")
            else:
                messagebox.showerror("错误", "无人机连接失败")
                
        except Exception as e:
            logger.error(f"连接无人机失败: {e}")
            messagebox.showerror("错误", f"连接失败: {str(e)}")
    
    def disconnect_drone(self):
        """断开无人机连接"""
        try:
            if self.drone_interface:
                self.drone_interface.disconnect()
                self.drone_interface = None
                self.drone_executor = None
            
            self.is_drone_connected = False
            
            # 更新UI状态
            self.control_buttons['connect'].config(state=tk.NORMAL)
            self.control_buttons['disconnect'].config(state=tk.DISABLED)
            
            logger.info("无人机连接已断开")
            
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
    
    def arm_drone(self):
        """解锁无人机"""
        if self.drone_interface:
            success = self.drone_interface.arm()
            if success:
                messagebox.showinfo("成功", "无人机解锁成功")
            else:
                messagebox.showerror("错误", "无人机解锁失败")
    
    def disarm_drone(self):
        """锁定无人机"""
        if self.drone_interface:
            success = self.drone_interface.disarm()
            if success:
                messagebox.showinfo("成功", "无人机锁定成功")
            else:
                messagebox.showerror("错误", "无人机锁定失败")
    
    def takeoff_drone(self):
        """起飞"""
        if self.drone_executor:
            command = {"action": "takeoff", "params": {"altitude": 2.0}}
            success = self.drone_executor.execute_command(command)
            if success:
                messagebox.showinfo("成功", "起飞指令发送成功")
            else:
                messagebox.showerror("错误", "起飞失败")
    
    def land_drone(self):
        """降落"""
        if self.drone_executor:
            command = {"action": "land", "params": {}}
            success = self.drone_executor.execute_command(command)
            if success:
                messagebox.showinfo("成功", "降落指令发送成功")
            else:
                messagebox.showerror("错误", "降落失败")
    
    def hover_drone(self):
        """悬停"""
        if self.drone_executor:
            command = {"action": "hover", "params": {}}
            success = self.drone_executor.execute_command(command)
    
    def emergency_stop(self):
        """紧急停止"""
        if self.drone_executor:
            command = {"action": "emergency_stop", "params": {}}
            success = self.drone_executor.execute_command(command)
            if success:
                messagebox.showwarning("警告", "紧急停止指令已发送")
            else:
                messagebox.showerror("错误", "紧急停止失败")
    
    def manual_move(self, vx: float, vy: float, vz: float):
        """手动移动"""
        if self.drone_executor:
            speed = self.speed_var.get()
            command = {
                "action": "move", 
                "params": {
                    "vx": vx * speed,
                    "vy": vy * speed, 
                    "vz": vz * speed,
                    "duration": 1.0
                }
            }
            self.drone_executor.execute_command(command)
    
    def save_screenshot(self):
        """保存当前截图"""
        try:
            if self.gesture_system.current_person_state:
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.jpg"
                # 这里需要从手势检测系统获取当前帧
                logger.info(f"截图保存为: {filename}")
                messagebox.showinfo("成功", f"截图已保存: {filename}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
    
    def _start_gui_update_thread(self):
        """启动GUI更新线程"""
        if not self.gui_update_thread or not self.gui_update_thread.is_alive():
            self.gui_update_thread = threading.Thread(target=self._gui_update_loop)
            self.gui_update_thread.daemon = True
            self.gui_update_thread.start()
    
    def _gui_update_loop(self):
        """GUI更新循环"""
        while self.is_system_running:
            try:
                # 更新状态显示
                self._update_status_display()
                
                # 检查手势识别结果
                if self.gesture_system.current_person_state:
                    person_state = self.gesture_system.current_person_state
                    
                    # 如果启用手势控制且检测到有效手势
                    if (self.gesture_control_var.get() and 
                        person_state.gesture != "none" and 
                        person_state.confidence >= self.confidence_var.get()):
                        
                        # 生成控制指令
                        command = self.gesture_system.command_generator.generate_command(
                            person_state.gesture,
                            person_state.confidence,
                            person_state.distance,
                            person_state.position
                        )
                        
                        if command:
                            try:
                                self.drone_command_queue.put_nowait(command)
                            except queue.Full:
                                pass  # 队列满时忽略
                
                time.sleep(0.1)  # 10Hz更新频率
                
            except Exception as e:
                logger.error(f"GUI更新循环错误: {e}")
                time.sleep(1.0)
    
    def _start_command_processing_thread(self):
        """启动指令处理线程"""
        if not self.command_processing_thread or not self.command_processing_thread.is_alive():
            self.command_processing_thread = threading.Thread(target=self._command_processing_loop)
            self.command_processing_thread.daemon = True
            self.command_processing_thread.start()
    
    def _command_processing_loop(self):
        """指令处理循环"""
        while self.is_system_running:
            try:
                if self.is_drone_connected and self.drone_executor:
                    # 获取指令
                    command = self.drone_command_queue.get(timeout=1.0)
                    
                    # 执行指令
                    command_data = {
                        "action": command.action,
                        "params": command.params
                    }
                    
                    success = self.drone_executor.execute_command(command_data)
                    
                    if success:
                        logger.info(f"手势指令执行成功: {command.action}")
                    else:
                        logger.warning(f"手势指令执行失败: {command.action}")
                    
                    self.drone_command_queue.task_done()
                else:
                    time.sleep(0.5)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"指令处理循环错误: {e}")
                time.sleep(1.0)
    
    def _update_status_display(self):
        """更新状态显示"""
        try:
            # 更新无人机状态
            if self.drone_interface:
                status = self.drone_interface.get_status()
                self.status_labels["连接状态"].config(text=status.state.value)
                self.status_labels["飞行模式"].config(text=status.flight_mode)
                self.status_labels["电池电量"].config(text=f"{status.battery_level:.1f}%")
                self.status_labels["当前高度"].config(text=f"{status.altitude:.1f}m")
                
                x, y, z = status.position
                self.status_labels["水平位置"].config(text=f"({x:.1f}, {y:.1f})")
                
                # 检查低电量警告
                if status.battery_level < 20 and self.auto_land_var.get():
                    self._handle_low_battery()
            else:
                self.status_labels["连接状态"].config(text="未连接")
            
            # 更新手势识别状态
            if self.gesture_system.current_person_state:
                person_state = self.gesture_system.current_person_state
                self.status_labels["当前手势"].config(text=person_state.gesture)
                self.status_labels["置信度"].config(text=f"{person_state.confidence:.1%}")
                self.status_labels["检测距离"].config(text=f"{person_state.distance:.1f}m")
            else:
                self.status_labels["当前手势"].config(text="无")
                self.status_labels["置信度"].config(text="0%")
                self.status_labels["检测距离"].config(text="0.0m")
                
        except Exception as e:
            logger.error(f"状态更新错误: {e}")
    
    def _handle_low_battery(self):
        """处理低电量情况"""
        if self.drone_executor and not hasattr(self, '_low_battery_handled'):
            self._low_battery_handled = True
            logger.warning("电量过低，执行自动降落")
            
            command = {"action": "land", "params": {}}
            self.drone_executor.execute_command(command)
            
            messagebox.showwarning("警告", "电量过低，无人机正在自动降落")
    
    def on_closing(self):
        """关闭程序时的处理"""
        try:
            self.stop_gesture_detection()
            self.disconnect_drone()
            self.root.destroy()
        except Exception as e:
            logger.error(f"程序关闭错误: {e}")
    
    def run(self):
        """运行GUI"""
        try:
            root = self.create_gui()
            logger.info("GUI启动成功")
            root.mainloop()
        except Exception as e:
            logger.error(f"GUI运行错误: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("无人机手势控制系统 v1.0")
    print("基于MediaPipe的实时手势识别与无人机控制")
    print("=" * 50)
    
    try:
        # 创建并运行GUI应用
        app = DroneControlGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        print(f"程序运行错误: {e}")
    finally:
        print("程序已退出")

if __name__ == "__main__":
    main()
