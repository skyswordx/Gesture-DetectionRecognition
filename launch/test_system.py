"""
系统测试工具
用于验证整合后的手势识别系统功能
"""

import sys
import os
import traceback

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_ws'))

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("  模块导入测试")
    print("=" * 60)
    
    modules_to_test = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("MediaPipe", "mediapipe"),
        ("PIL", "PIL"),
        ("tkinter", "tkinter"),
        ("图像处理模块", "image_processing.image_processor"),
        ("姿势检测模块", "pose_detection.pose_detector"),
        ("距离估算模块", "distance_estimation.distance_estimator"),
        ("手势识别模块", "gesture_recognition.gesture_recognizer"),
    ]
    
    results = []
    
    for name, module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {name} - 导入成功")
            results.append((name, True, None))
        except ImportError as e:
            print(f"❌ {name} - 导入失败: {e}")
            results.append((name, False, str(e)))
        except Exception as e:
            print(f"⚠️ {name} - 其他错误: {e}")
            results.append((name, False, str(e)))
    
    return results

def test_camera_access():
    """测试摄像头访问"""
    print("\n" + "=" * 60)
    print("  摄像头访问测试")
    print("=" * 60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ 摄像头访问成功")
                print(f"   分辨率: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print("❌ 摄像头无法读取画面")
                cap.release()
                return False
        else:
            print("❌ 无法打开摄像头")
            return False
            
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def test_gui_system():
    """测试GUI系统"""
    print("\n" + "=" * 60)
    print("  GUI系统测试")
    print("=" * 60)
    
    try:
        from gui_system import IntegratedGestureGUI
        print("✅ GUI系统模块导入成功")
        
        # 不实际启动GUI，只测试初始化
        try:
            gui = IntegratedGestureGUI()
            print("✅ GUI系统初始化成功")
            gui.root.destroy()  # 立即销毁窗口
            return True
        except Exception as e:
            print(f"❌ GUI系统初始化失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ GUI系统模块导入失败: {e}")
        return False

def test_console_system():
    """测试控制台系统"""
    print("\n" + "=" * 60)
    print("  控制台系统测试")
    print("=" * 60)
    
    try:
        from main_integrated import GestureControlSystem
        print("✅ 控制台系统模块导入成功")
        
        # 测试初始化（不启动摄像头）
        try:
            system = GestureControlSystem(camera_id=0)
            print("✅ 控制台系统初始化成功")
            return True
        except Exception as e:
            print(f"❌ 控制台系统初始化失败: {e}")
            print(f"   详细错误: {traceback.format_exc()}")
            return False
            
    except ImportError as e:
        print(f"❌ 控制台系统模块导入失败: {e}")
        return False

def generate_report(import_results, camera_ok, gui_ok, console_ok):
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("  系统测试报告")
    print("=" * 80)
    
    # 模块导入统计
    total_modules = len(import_results)
    success_modules = sum(1 for _, success, _ in import_results if success)
    
    print(f"模块导入测试: {success_modules}/{total_modules} 成功")
    for name, success, error in import_results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
        if not success and error:
            print(f"      错误: {error}")
    
    print(f"\n摄像头访问测试: {'✅ 成功' if camera_ok else '❌ 失败'}")
    print(f"GUI系统测试: {'✅ 成功' if gui_ok else '❌ 失败'}")
    print(f"控制台系统测试: {'✅ 成功' if console_ok else '❌ 失败'}")
    
    # 总体评估
    if success_modules == total_modules and camera_ok and (gui_ok or console_ok):
        print("\n🎉 系统测试通过！可以正常使用")
        recommendation = "推荐使用GUI模式获得最佳体验" if gui_ok else "建议使用控制台模式"
        print(f"   {recommendation}")
    else:
        print("\n⚠️ 系统测试发现问题，需要解决后才能正常使用")
        
        if success_modules < total_modules:
            print("   • 请安装缺失的依赖包")
        if not camera_ok:
            print("   • 请检查摄像头连接和权限")
        if not gui_ok and not console_ok:
            print("   • 请检查系统组件配置")
    
    print("=" * 80)

def main():
    """主函数"""
    print("手势识别控制系统 - 系统测试工具")
    
    try:
        # 执行各项测试
        import_results = test_imports()
        camera_ok = test_camera_access()
        gui_ok = test_gui_system()
        console_ok = test_console_system()
        
        # 生成报告
        generate_report(import_results, camera_ok, gui_ok, console_ok)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
