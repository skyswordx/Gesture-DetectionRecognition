"""
人体检测与姿势识别系统
基于MediaPipe实现人形框定、骨骼点识别和姿势识别
"""

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import math

class HumanDetectionSystem:
    def __init__(self):
        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 模型初始化
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 摄像头
        self.cap = None
        self.is_running = False
        
        # GUI
        self.root = None
        self.canvas = None
        self.current_mode = "pose"  # pose, holistic, gesture
        
        # 姿势识别相关
        self.gesture_buffer = []
        self.buffer_size = 10
        self.current_gesture = "未知"
        
        # 界面元素
        self.info_text = None
        self.gesture_label = None
        
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            messagebox.showerror("错误", f"摄像头初始化失败: {str(e)}")
            return False
    
    def release_camera(self):
        """释放摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def detect_human_bbox(self, image, landmarks):
        """根据关键点检测人体边界框"""
        if not landmarks:
            return None
        
        h, w = image.shape[:2]
        x_coords = [landmark.x * w for landmark in landmarks.landmark]
        y_coords = [landmark.y * h for landmark in landmarks.landmark]
        
        x_min = int(max(0, min(x_coords) - 20))
        y_min = int(max(0, min(y_coords) - 20))
        x_max = int(min(w, max(x_coords) + 20))
        y_max = int(min(h, max(y_coords) + 20))
        
        return (x_min, y_min, x_max, y_max)
    
    def calculate_angle(self, point1, point2, point3):
        """计算三点构成的角度"""
        # 将点转换为numpy数组
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        # 计算向量
        ba = a - b
        bc = c - b
        
        # 计算角度
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def recognize_gesture(self, landmarks):
        """识别手势姿势"""
        if not landmarks:
            return "未检测到人体"
        
        try:
            # 获取关键点
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # 计算角度
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # 识别具体姿势
            gesture = "站立"
            
            # 双臂举起
            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                if abs(left_wrist.y - right_wrist.y) < 0.1:
                    gesture = "双臂举起"
                elif left_wrist.y < right_wrist.y:
                    gesture = "左臂举起"
                else:
                    gesture = "右臂举起"
            
            # 单臂举起
            elif left_wrist.y < left_shoulder.y:
                gesture = "左臂举起"
            elif right_wrist.y < right_shoulder.y:
                gesture = "右臂举起"
            
            # 双臂展开
            elif (abs(left_wrist.x - left_shoulder.x) > 0.2 and 
                  abs(right_wrist.x - right_shoulder.x) > 0.2 and
                  abs(left_wrist.y - left_shoulder.y) < 0.1 and
                  abs(right_wrist.y - right_shoulder.y) < 0.1):
                gesture = "双臂展开"
            
            # 挥手动作（基于手腕位置变化）
            elif len(self.gesture_buffer) > 5:
                recent_positions = self.gesture_buffer[-5:]
                x_variance = np.var([pos[0] for pos in recent_positions])
                if x_variance > 0.01:
                    gesture = "挥手"
            
            # 添加当前位置到缓冲区
            self.gesture_buffer.append((left_wrist.x, left_wrist.y))
            if len(self.gesture_buffer) > self.buffer_size:
                self.gesture_buffer.pop(0)
            
            return gesture
            
        except Exception as e:
            return f"姿势识别错误: {str(e)}"
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = None
        
        if self.current_mode == "pose":
            results = self.pose.process(rgb_frame)
            landmarks = results.pose_landmarks
        elif self.current_mode == "holistic":
            results = self.holistic.process(rgb_frame)
            landmarks = results.pose_landmarks
        else:
            landmarks = None
        
        # 绘制检测结果
        if landmarks:
            # 绘制骨骼点
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # 绘制人体边界框
            bbox = self.detect_human_bbox(frame, landmarks)
            if bbox:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, "Human Detected", (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 识别姿势
            if self.current_mode == "gesture":
                gesture = self.recognize_gesture(landmarks)
                self.current_gesture = gesture
                
                # 在图像上显示姿势
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 如果是全身模式，还要绘制面部和手部关键点
        if self.current_mode == "holistic" and results:
            # 绘制面部关键点
            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # 绘制手部关键点
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
                )
        
        return frame
    
    def update_frame(self):
        """更新视频帧"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 水平翻转镜像效果
            frame = cv2.flip(frame, 1)
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 转换为PIL格式显示
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((640, 480))
            
            # 转换为tkinter格式
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # 更新canvas
            self.canvas.delete("all")
            self.canvas.create_image(320, 240, image=tk_image)
            self.canvas.image = tk_image
            
            # 更新信息显示
            if self.current_mode == "gesture":
                self.gesture_label.config(text=f"当前姿势: {self.current_gesture}")
        
        # 继续下一帧
        if self.is_running:
            self.root.after(33, self.update_frame)  # 约30FPS
    
    def start_detection(self):
        """开始检测"""
        if not self.init_camera():
            return
        
        self.is_running = True
        self.update_frame()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        self.release_camera()
        if self.canvas:
            self.canvas.delete("all")
    
    def set_mode(self, mode):
        """设置检测模式"""
        self.current_mode = mode
        print(f"切换到模式: {mode}")
    
    def create_gui(self):
        """创建图形界面"""
        self.root = tk.Tk()
        self.root.title("人体检测与姿势识别系统")
        self.root.geometry("800x700")
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模式选择
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_frame, text="检测模式:").pack(side=tk.LEFT)
        
        mode_var = tk.StringVar(value="pose")
        modes = [
            ("骨骼点检测", "pose"),
            ("全身检测", "holistic"),
            ("姿势识别", "gesture")
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=mode_var, value=mode,
                           command=lambda m=mode: self.set_mode(m)).pack(side=tk.LEFT, padx=10)
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="开始检测", command=self.start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="停止检测", command=self.stop_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.quit_app).pack(side=tk.LEFT, padx=5)
        
        # 信息显示
        info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gesture_label = ttk.Label(info_frame, text="当前姿势: 未检测", font=("Arial", 12))
        self.gesture_label.pack()
        
        instructions = ttk.Label(info_frame, text="支持的姿势: 站立、左臂举起、右臂举起、双臂举起、双臂展开、挥手", 
                                font=("Arial", 10))
        instructions.pack(pady=(5, 0))
        
        # 视频显示区域
        video_frame = ttk.LabelFrame(main_frame, text="视频显示", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg="black")
        self.canvas.pack()
        
        return self.root
    
    def quit_app(self):
        """退出应用"""
        self.stop_detection()
        if self.root:
            self.root.quit()
            self.root.destroy()

def main():
    """主函数"""
    try:
        system = HumanDetectionSystem()
        root = system.create_gui()
        
        # 设置退出处理
        def on_closing():
            system.quit_app()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 启动GUI
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("错误", f"系统启动失败: {str(e)}")

if __name__ == "__main__":
    main()
