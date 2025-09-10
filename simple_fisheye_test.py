"""
简化的鱼眼校正模块测试脚本
使用Anaconda Python环境
"""

import sys
import os

def test_module_import():
    """测试模块导入"""
    print("Testing fisheye module import...")
    
    # 添加模块路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_path = os.path.join(current_dir, 'modules')
    sys.path.insert(0, modules_path)
    
    try:
        # 测试基本导入
        print("Importing camera_calibration module...")
        import camera_calibration
        print("✅ camera_calibration module imported")
        
        # 测试具体类导入
        from camera_calibration import FisheyeCalibrator
        print("✅ FisheyeCalibrator imported")
        
        from camera_calibration import DistortionCorrector
        print("✅ DistortionCorrector imported")
        
        from camera_calibration import CalibrationVisualizer
        print("✅ CalibrationVisualizer imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_class_initialization():
    """测试类初始化"""
    print("\nTesting class initialization...")
    
    try:
        from camera_calibration import FisheyeCalibrator, DistortionCorrector, CalibrationVisualizer
        
        # 测试FisheyeCalibrator初始化
        calibrator = FisheyeCalibrator(chessboard_size=(9, 6), square_size=1.0)
        print("✅ FisheyeCalibrator initialized successfully")
        
        # 测试DistortionCorrector初始化
        corrector = DistortionCorrector()
        print("✅ DistortionCorrector initialized successfully")
        
        # 测试CalibrationVisualizer初始化
        visualizer = CalibrationVisualizer()
        print("✅ CalibrationVisualizer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Class initialization failed: {e}")
        return False

def create_test_calibration_data():
    """创建测试用的校正数据"""
    print("\nCreating test calibration data...")
    
    import json
    
    # 创建虚拟校正数据
    test_data = {
        "camera_matrix": [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        "distortion_coeffs": [0.1, 0.05, -0.02, 0.01],
        "rms_error": 0.5,
        "image_size": [640, 480],
        "calibration_flags": 0,
        "fisheye_flags": 0,
        "is_valid": True,
        "calibration_date": "2024-01-01 12:00:00"
    }
    
    # 保存到文件
    calibration_dir = "calibration_data"
    os.makedirs(calibration_dir, exist_ok=True)
    calibration_path = os.path.join(calibration_dir, "test_fisheye_calibration.json")
    
    with open(calibration_path, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Test calibration data saved to: {calibration_path}")
    return calibration_path

def test_calibration_loading():
    """测试校正数据加载"""
    print("\nTesting calibration loading...")
    
    try:
        from camera_calibration import FisheyeCalibrator
        
        # 创建测试数据
        calibration_path = create_test_calibration_data()
        
        # 测试加载
        calibrator = FisheyeCalibrator()
        
        if calibrator.load_calibration(calibration_path):
            print("✅ Calibration data loaded successfully")
            print(f"   Image size: {calibrator.calibration_result.image_size}")
            print(f"   RMS error: {calibrator.calibration_result.rms_error}")
            print(f"   Valid: {calibrator.calibration_result.is_valid}")
            return True
        else:
            print("❌ Failed to load calibration data")
            return False
            
    except Exception as e:
        print(f"❌ Calibration loading test failed: {e}")
        return False

def test_opencv_dependency():
    """测试OpenCV依赖"""
    print("\nTesting OpenCV dependency...")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # 测试创建简单图像
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        print(f"✅ Test image created: {test_image.shape}")
        
        return True
        
    except ImportError:
        print("❌ OpenCV not available")
        print("   Please install: pip install opencv-python")
        return False
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("Fisheye Camera Calibration Module Test")
    print("=" * 50)
    
    # 显示Python环境信息
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("OpenCV Dependency", test_opencv_dependency),
        ("Module Import", test_module_import),
        ("Class Initialization", test_class_initialization),
        ("Calibration Loading", test_calibration_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        print(f"Running: {test_name}")
        print(f"{'-' * 30}")
        
        result = test_func()
        results.append((test_name, result))
    
    # 测试总结
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
        print("\nNext steps:")
        print("1. The fisheye calibration module is working correctly")
        print("2. You can now run the calibration app:")
        print("   python modules/camera_calibration/calibration_app.py")
        print("3. Or integrate it into the main system")
    else:
        print("⚠️ Some tests failed.")
        print("Please check the error messages above.")
    
    # 清理测试文件
    try:
        calibration_path = os.path.join("calibration_data", "test_fisheye_calibration.json")
        if os.path.exists(calibration_path):
            os.remove(calibration_path)
        if os.path.exists("calibration_data") and not os.listdir("calibration_data"):
            os.rmdir("calibration_data")
        print("\n🧹 Test files cleaned up")
    except:
        pass

if __name__ == "__main__":
    main()
