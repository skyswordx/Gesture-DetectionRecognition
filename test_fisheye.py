"""
Test script for fisheye camera calibration module
"""

import cv2
import numpy as np
import os
import sys

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

def test_fisheye_module():
    """Test the fisheye calibration module"""
    try:
        from camera_calibration import (
            FisheyeCalibrator, 
            DistortionCorrector, 
            CalibrationVisualizer
        )
        
        print("✅ Fisheye calibration module imported successfully")
        
        # Test calibrator initialization
        calibrator = FisheyeCalibrator(chessboard_size=(9, 6), square_size=1.0)
        print("✅ FisheyeCalibrator initialized")
        
        # Test corrector initialization
        corrector = DistortionCorrector()
        print("✅ DistortionCorrector initialized")
        
        # Test visualizer initialization
        visualizer = CalibrationVisualizer()
        print("✅ CalibrationVisualizer initialized")
        
        # Test dummy calibration data
        create_dummy_calibration()
        
        # Test loading calibration
        calibration_path = "test_calibration.json"
        if calibrator.load_calibration(calibration_path):
            print("✅ Dummy calibration loaded successfully")
            
            # Test setting calibration in corrector
            if corrector.set_calibration(calibrator.calibration_result):
                print("✅ Calibration set in corrector")
                
                # Test correction with dummy image
                test_image = create_test_image()
                corrected = corrector.correct_distortion(test_image)
                
                if corrected is not None:
                    print("✅ Image correction successful")
                    print(f"   Original size: {test_image.shape}")
                    print(f"   Corrected size: {corrected.shape}")
                else:
                    print("❌ Image correction failed")
            else:
                print("❌ Failed to set calibration in corrector")
        else:
            print("❌ Failed to load dummy calibration")
        
        # Clean up test file
        if os.path.exists(calibration_path):
            os.remove(calibration_path)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import fisheye module: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def create_dummy_calibration():
    """Create dummy calibration data for testing"""
    import json
    
    dummy_data = {
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
    
    with open("test_calibration.json", "w") as f:
        json.dump(dummy_data, f, indent=2)

def create_test_image():
    """Create a test image for correction testing"""
    # Create a simple test pattern
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.circle(image, (320, 240), 100, (255, 255, 255), 2)
    cv2.rectangle(image, (200, 150), (440, 330), (0, 255, 0), 2)
    cv2.line(image, (0, 240), (640, 240), (255, 0, 0), 2)
    cv2.line(image, (320, 0), (320, 480), (255, 0, 0), 2)
    
    return image

def test_integration_with_main():
    """Test integration with main system"""
    try:
        # Try importing with the same method as main system
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
        
        try:
            from camera_calibration import DistortionCorrector, FisheyeCalibrator
            FISHEYE_AVAILABLE = True
            print("✅ Fisheye module available for integration")
        except ImportError:
            FISHEYE_AVAILABLE = False
            print("❌ Fisheye module not available for integration")
            return False
        
        # Test initialization as would be done in main system
        if FISHEYE_AVAILABLE:
            # Create dummy calibration file in expected location
            calibration_dir = os.path.join(os.path.dirname(__file__), '..', 'calibration_data')
            os.makedirs(calibration_dir, exist_ok=True)
            
            create_dummy_calibration()
            import shutil
            shutil.move("test_calibration.json", 
                       os.path.join(calibration_dir, "fisheye_calibration.json"))
            
            # Test loading as main system would
            calibrator = FisheyeCalibrator()
            calibration_path = os.path.join(calibration_dir, "fisheye_calibration.json")
            
            if calibrator.load_calibration(calibration_path):
                corrector = DistortionCorrector(calibrator.calibration_result)
                print("✅ Integration test successful")
                
                # Clean up
                os.remove(calibration_path)
                return True
            else:
                print("❌ Integration test failed - couldn't load calibration")
                return False
        
        return FISHEYE_AVAILABLE
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fisheye Camera Calibration Module")
    print("=" * 50)
    
    # Test module functionality
    print("\n1. Testing module functionality...")
    module_test = test_fisheye_module()
    
    # Test integration
    print("\n2. Testing integration with main system...")
    integration_test = test_integration_with_main()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Module functionality: {'✅ PASS' if module_test else '❌ FAIL'}")
    print(f"Main system integration: {'✅ PASS' if integration_test else '❌ FAIL'}")
    
    if module_test and integration_test:
        print("\n🎉 All tests passed! Fisheye calibration module is ready.")
        print("\nNext steps:")
        print("1. Run the calibration app: python modules/camera_calibration/calibration_app.py")
        print("2. Capture calibration images using a chessboard pattern")
        print("3. Save calibration data")
        print("4. Run main system with 'f' key to toggle fisheye correction")
    else:
        print("\n⚠️ Some tests failed. Please check the module installation.")
