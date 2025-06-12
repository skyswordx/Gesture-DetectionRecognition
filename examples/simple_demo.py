"""
简单入门示例 - 展示如何使用单个模块
"""

import cv2
import time
import sys
import os

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

def simple_pose_detection_demo():
    """简单的姿势检测演示"""
    print("简单姿势检测演示")
    print("按 'q' 退出")
    
    # 导入必要的模块
    from image_processing.image_processor import CameraCapture
    from pose_detection.pose_detector import PoseDetector, PoseVisualizer
    
    # 初始化组件
    capture = CameraCapture(camera_id=0)
    detector = PoseDetector()
    visualizer = PoseVisualizer()
    
    # 启动摄像头
    if not capture.start():
        print("无法启动摄像头")
        return
    
    try:
        while True:
            # 获取图像
            frame = capture.get_frame()
            if frame is None:
                continue
            
            # 检测姿势
            result = detector.detect(frame)
            
            # 可视化结果
            output = visualizer.draw_pose(frame, result)
            
            # 显示
            cv2.imshow("Simple Pose Detection", output)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        capture.stop()
        cv2.destroyAllWindows()

def simple_gesture_recognition_demo():
    """简单的手势识别演示"""
    print("简单手势识别演示")
    print("支持手势: takeoff, landing, stop")
    print("按 'q' 退出")
    
    # 导入必要的模块
    from image_processing.image_processor import CameraCapture
    from pose_detection.pose_detector import PoseDetector
    from gesture_recognition.gesture_recognizer import GestureRecognizer
    
    # 初始化组件
    capture = CameraCapture(camera_id=0)
    pose_detector = PoseDetector()
    gesture_recognizer = GestureRecognizer()
    
    # 启动摄像头
    if not capture.start():
        print("无法启动摄像头")
        return
    
    try:
        last_gesture = "none"
        
        while True:
            # 获取图像
            frame = capture.get_frame()
            if frame is None:
                continue
            
            # 检测姿势
            pose_result = pose_detector.detect(frame)
            
            # 识别手势
            if pose_result.landmarks:
                gesture_result = gesture_recognizer.recognize(pose_result.landmarks)
                
                # 只在手势改变时打印
                if gesture_result.gesture != last_gesture and gesture_result.confidence > 0.7:
                    print(f"检测到手势: {gesture_result.gesture} (置信度: {gesture_result.confidence:.2f})")
                    last_gesture = gesture_result.gesture
                
                # 在图像上显示当前手势
                text = f"Gesture: {gesture_result.gesture} ({gesture_result.confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示
            cv2.imshow("Simple Gesture Recognition", frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        capture.stop()
        cv2.destroyAllWindows()

def main():
    """主菜单"""
    print("=" * 40)
    print("  简单入门示例")
    print("=" * 40)
    print("1. 姿势检测演示")
    print("2. 手势识别演示")
    print("0. 退出")
    print("=" * 40)
    
    while True:
        try:
            choice = input("请选择演示项目 (0-2): ").strip()
            
            if choice == "0":
                print("退出程序")
                break
            elif choice == "1":
                simple_pose_detection_demo()
            elif choice == "2":
                simple_gesture_recognition_demo()
            else:
                print("无效选择，请重新输入")
        
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    main()
