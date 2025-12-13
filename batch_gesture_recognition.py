# -*- coding: utf-8 -*-
"""
批量手势识别脚本
遍历指定文件夹中的所有图片，进行骨骼点检测和手势识别，
将结果图片保存到输出文件夹。
"""

import cv2
import sys
import os
import argparse
from datetime import datetime

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from pose_detection.pose_detector import PoseDetector, PoseVisualizer
from gesture_recognition.gesture_recognizer import GestureRecognizer, GestureVisualizer


def process_batch(input_folder: str, output_folder: str, supported_exts: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """
    批量处理图片
    
    Args:
        input_folder: 输入图片文件夹
        output_folder: 输出结果文件夹
        supported_exts: 支持的图片格式
    """
    # 检查输入文件夹
    if not os.path.isdir(input_folder):
        print(f"[ERROR] 输入文件夹不存在: {input_folder}")
        return
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件
    image_files = []
    for f in sorted(os.listdir(input_folder)):
        if f.lower().endswith(supported_exts):
            image_files.append(os.path.join(input_folder, f))
    
    if not image_files:
        print(f"[ERROR] 未找到图片文件: {input_folder}")
        return
    
    print("=" * 60)
    print("批量手势识别")
    print("=" * 60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"图片数量: {len(image_files)}")
    print("=" * 60)
    
    # 初始化检测器
    pose_detector = PoseDetector(model_complexity=1)
    pose_visualizer = PoseVisualizer()
    gesture_visualizer = GestureVisualizer()
    
    # 统计
    success_count = 0
    fail_count = 0
    no_person_count = 0
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"[{i+1}/{len(image_files)}] 处理: {filename}", end=" ... ")
        
        # 每张图片使用新的识别器，避免时序一致性影响
        gesture_recognizer = GestureRecognizer()
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print("[FAIL] 无法读取")
            fail_count += 1
            continue
        
        # 姿势检测
        pose_result = pose_detector.detect(img)
        
        if not pose_result.landmarks:
            print("[SKIP] 未检测到人体")
            no_person_count += 1
            # 保存原图并标注未检测到人体
            output = img.copy()
            cv2.putText(output, "NO PERSON DETECTED", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            output_path = os.path.join(output_folder, f"no_person_{filename}")
            cv2.imwrite(output_path, output)
            continue
        
        # 手势识别
        frame_info = {'width': pose_result.frame_width, 'height': pose_result.frame_height}
        gesture_result = gesture_recognizer.recognize(pose_result.landmarks, frame_info)
        
        # 绘制结果
        output = img.copy()
        output = pose_visualizer.draw_pose(output, pose_result)
        output = gesture_visualizer.draw_gesture_info(output, gesture_result, pose_result.landmarks)
        
        # 添加文件名和识别结果标注
        h, w = output.shape[:2]
        # 顶部信息栏
        cv2.rectangle(output, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(output, f"File: {filename}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Gesture: {gesture_result.gesture.upper()}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, f"Confidence: {gesture_result.confidence:.2f}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 保存结果
        output_filename = f"{gesture_result.gesture}_{filename}"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, output)
        
        print(f"[OK] {gesture_result.gesture} ({gesture_result.confidence:.2f})")
        success_count += 1
        results_summary.append({
            'filename': filename,
            'gesture': gesture_result.gesture,
            'confidence': gesture_result.confidence
        })
    
    # 生成汇总报告
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"未检测到人体: {no_person_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(image_files)}")
    print(f"输出目录: {output_folder}")
    
    # 保存汇总报告
    report_path = os.path.join(output_folder, "recognition_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Gesture Recognition Batch Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {input_folder}\n")
        f.write(f"Output: {output_folder}\n")
        f.write(f"Total: {len(image_files)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"No Person: {no_person_count}\n")
        f.write(f"Failed: {fail_count}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Results:\n")
        f.write("=" * 60 + "\n")
        for r in results_summary:
            f.write(f"{r['filename']}: {r['gesture']} ({r['confidence']:.3f})\n")
    
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量手势识别")
    parser.add_argument('--input', type=str, default=r"E:\Project\Trisheye\result",
                        help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, default=r"E:\Project\Trisheye\gesture_output",
                        help='输出结果文件夹路径')
    args = parser.parse_args()
    
    process_batch(args.input, args.output)
