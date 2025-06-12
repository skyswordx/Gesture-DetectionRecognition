"""
简化版人体检测演示
快速测试MediaPipe功能
"""

import cv2
import mediapipe as mp
import numpy as np

class SimpleHumanDetection:
    def __init__(self):
        # MediaPipe初始化
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.gesture_history = []
        
    def detect_human_bbox(self, image, landmarks):
        """检测人体边界框"""
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
    
    def recognize_simple_gesture(self, landmarks):
        """简单姿势识别"""
        if not landmarks:
            return "未检测到人体"
        
        try:
            # 获取关键点
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # 简单的姿势判断
            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                if abs(left_wrist.y - right_wrist.y) < 0.1:
                    return "双手举起 🙌"
                else:
                    return "一只手举起 ✋"
            elif left_wrist.y < left_shoulder.y:
                return "左手举起 ✋"
            elif right_wrist.y < right_shoulder.y:
                return "右手举起 ✋"
            elif (abs(left_wrist.x - left_shoulder.x) > 0.2 and 
                  abs(right_wrist.x - right_shoulder.x) > 0.2):
                return "双臂展开 🤸"
            else:
                return "正常站立 🧍"
                
        except Exception as e:
            return "识别错误"
    
    def run(self):
        """运行检测系统"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("错误: 无法打开摄像头!")
            print("请检查:")
            print("1. 摄像头是否被其他程序占用")
            print("2. 系统是否授予了摄像头权限")
            return
        
        print("人体检测系统已启动!")
        print("按键说明:")
        print("  'q' - 退出程序")
        print("  's' - 截图保存")
        print("  空格 - 暂停/继续")
        
        paused = False
        frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 镜像翻转
                frame = cv2.flip(frame, 1)
                
                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                # 绘制结果
                if results.pose_landmarks:
                    # 绘制骨骼点和连接线
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # 绘制人体边界框
                    bbox = self.detect_human_bbox(frame, results.pose_landmarks)
                    if bbox:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, "Human Detected", (bbox[0], bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 姿势识别
                    gesture = self.recognize_simple_gesture(results.pose_landmarks)
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 显示关键点数量
                    landmark_count = len(results.pose_landmarks.landmark)
                    cv2.putText(frame, f"Landmarks: {landmark_count}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                else:
                    cv2.putText(frame, "No Human Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 显示帧数
                cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                frame_count += 1
            
            # 显示状态信息
            status = "PAUSED" if paused else "RUNNING"
            cv2.putText(frame, status, (frame.shape[1]-100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示画面
            cv2.imshow('Human Detection & Pose Recognition', frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}检测")
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"截图已保存: {filename}")
        
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("系统已退出")

def main():
    """主函数"""
    print("=== MediaPipe 人体检测与姿势识别系统 ===")
    print("功能:")
    print("1. 人形框定检测")
    print("2. 骨骼关键点识别") 
    print("3. 基础姿势识别")
    print("=" * 40)
    
    try:
        detector = SimpleHumanDetection()
        detector.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序错误: {str(e)}")

if __name__ == "__main__":
    main()
