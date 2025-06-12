"""
基础功能测试
"""

import sys
import os
import time
import numpy as np

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("测试模块导入...")
    
    try:
        from image_processing.image_processor import CameraCapture, ImageProcessor
        print("✅ 图像处理模块导入成功")
    except ImportError as e:
        print(f"❌ 图像处理模块导入失败: {e}")
        return False
    
    try:
        from pose_detection.pose_detector import PoseDetector
        print("✅ 姿势检测模块导入成功")
    except ImportError as e:
        print(f"❌ 姿势检测模块导入失败: {e}")
        return False
    
    try:
        from distance_estimation.distance_estimator import DistanceEstimator
        print("✅ 距离估算模块导入成功")
    except ImportError as e:
        print(f"❌ 距离估算模块导入失败: {e}")
        return False
    
    try:
        from gesture_recognition.gesture_recognizer import GestureRecognizer
        print("✅ 手势识别模块导入成功")
    except ImportError as e:
        print(f"❌ 手势识别模块导入失败: {e}")
        return False
    
    return True

def test_camera_availability():
    """测试摄像头可用性"""
    print("\n测试摄像头可用性...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ 摄像头可用 - 分辨率: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print("❌ 摄像头无法读取帧")
                cap.release()
                return False
        else:
            print("❌ 摄像头无法打开")
            return False
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def test_mediapipe():
    """测试MediaPipe功能"""
    print("\n测试MediaPipe...")
    
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # 创建姿势检测模型
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(test_image)
        
        print("✅ MediaPipe初始化成功")
        pose.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe测试失败: {e}")
        return False

def test_basic_functionality():
    """测试基础功能"""
    print("\n测试基础功能...")
    
    try:
        # 测试图像处理
        from image_processing.image_processor import ImageProcessor, ImageQualityAssessment
        
        processor = ImageProcessor()
        quality_assessor = ImageQualityAssessment()
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试预处理
        processed = processor.preprocess(test_image)
        assert processed is not None, "图像预处理失败"
        
        # 测试质量评估
        quality = quality_assessor.assess_quality(test_image)
        assert isinstance(quality, dict), "质量评估返回格式错误"
        
        print("✅ 图像处理功能正常")
        
        # 测试姿势检测
        from pose_detection.pose_detector import PoseDetector
        
        detector = PoseDetector()
        result = detector.detect(test_image)
        assert result is not None, "姿势检测失败"
        
        print("✅ 姿势检测功能正常")
        
        # 测试距离估算
        from distance_estimation.distance_estimator import DistanceEstimator
        
        estimator = DistanceEstimator()
        # 使用空的landmarks测试
        distance_result = estimator.estimate_distance(None, 640, 480)
        assert distance_result is not None, "距离估算失败"
        
        print("✅ 距离估算功能正常")
        
        # 测试手势识别
        from gesture_recognition.gesture_recognizer import GestureRecognizer
        
        recognizer = GestureRecognizer()
        gesture_result = recognizer.recognize(None)
        assert gesture_result is not None, "手势识别失败"
        
        print("✅ 手势识别功能正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("  模块化系统功能测试")
    print("=" * 50)
    
    tests = [
        ("模块导入测试", test_imports),
        ("摄像头可用性测试", test_camera_availability),
        ("MediaPipe测试", test_mediapipe),
        ("基础功能测试", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name}失败")
        except Exception as e:
            print(f"❌ {test_name}异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 50)
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查系统配置。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n建议接下来的步骤:")
        print("1. 运行综合演示: python examples/integrated_demo.py")
        print("2. 测试各个模块: python modules/<module_name>/<main_file>.py")
        print("3. 查看模块文档: modules/<module_name>/README.md")
    else:
        print("\n故障排除建议:")
        print("1. 检查Python版本 (需要3.8+)")
        print("2. 安装依赖: pip install -r requirements.txt")
        print("3. 检查摄像头连接")
        print("4. 查看详细错误信息")
