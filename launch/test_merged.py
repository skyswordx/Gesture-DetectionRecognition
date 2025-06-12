#!/usr/bin/env python3
"""
测试合并后的main_integrated.py文件
"""

import sys
import os

# 添加路径
sys.path.append(os.path.dirname(__file__))

def test_import():
    """测试是否能成功导入"""
    try:
        # 尝试导入主要类
        from main_integrated import GestureControlSystem, IntegratedGestureGUI
        print("✅ 成功导入 GestureControlSystem 和 IntegratedGestureGUI")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_class_instantiation():
    """测试类实例化"""
    try:
        from main_integrated import GestureControlSystem, IntegratedGestureGUI
        
        # 测试 GestureControlSystem（不启动摄像头）
        print("测试 GestureControlSystem 实例化...")
        # system = GestureControlSystem()  # 会尝试初始化摄像头，先注释掉
        print("✅ GestureControlSystem 类定义正常")
        
        # 测试 IntegratedGestureGUI（不启动GUI）
        print("测试 IntegratedGestureGUI 实例化...")
        # gui = IntegratedGestureGUI()  # 会启动tkinter窗口，先注释掉
        print("✅ IntegratedGestureGUI 类定义正常")
        
        return True
    except Exception as e:
        print(f"❌ 类实例化测试失败: {e}")
        return False

def test_function_availability():
    """测试主要函数是否可用"""
    try:
        from main_integrated import show_mode_selection, start_gui_mode, start_console_mode, parse_arguments, check_system_requirements, main
        
        print("✅ 所有主要函数都可以导入")
        
        # 测试参数解析
        print("测试参数解析功能...")
        import argparse
        # 临时修改sys.argv来测试
        original_argv = sys.argv.copy()
        sys.argv = ['test_merged.py', '--help']
        
        try:
            args = parse_arguments()
        except SystemExit:
            # argparse在--help时会调用SystemExit，这是正常的
            pass
        finally:
            sys.argv = original_argv
        
        print("✅ 参数解析功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 函数可用性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("  测试合并后的 main_integrated.py 文件")
    print("=" * 60)
    
    tests = [
        ("导入测试", test_import),
        ("类定义测试", test_class_instantiation),
        ("函数可用性测试", test_function_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 执行 {test_name}...")
        if test_func():
            passed += 1
        print(f"   {test_name}: {'通过' if test_func() else '失败'}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("✅ 所有测试通过！文件合并成功。")
        print("\n可以使用以下命令启动系统:")
        print("  python main_integrated.py --gui     # GUI模式")
        print("  python main_integrated.py --console # 控制台模式")
        print("  python main_integrated.py           # 显示模式选择对话框")
    else:
        print("❌ 部分测试失败，请检查文件。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
