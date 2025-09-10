# -*- coding: utf-8 -*-
"""
Calibrate the RGB Camera with a live feed.

By Gemini, based on You Zhiyuan's code, 2024.
"""

import os
import cv2
import time
import numpy as np
import glob
from calibrate_helper import Calibrator

def main():
    # --- 1. 配置和路径设置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_run_dir = os.path.join(script_dir, f"test_run_{timestamp}")
    os.makedirs(test_run_dir, exist_ok=True)
    
    captured_img_dir = os.path.join(test_run_dir, "captured_images")
    corner_detected_img_dir = os.path.join(test_run_dir, "corner_detected_images")
    undistorted_img_dir = os.path.join(test_run_dir, "undistorted_images")
    os.makedirs(captured_img_dir, exist_ok=True)
    os.makedirs(corner_detected_img_dir, exist_ok=True)
    os.makedirs(undistorted_img_dir, exist_ok=True)
    
    shape_inner_corner = (11, 8)
    size_grid = 0.02
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头！请检查设备或索引。")
        return

    img_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧，退出。")
            break

        info_text = f"Press 's' to save. Saved: {img_count} | Press 'q' to quit and start calib."
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Live Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            img_name = f"calib_img_{img_count:02d}.jpg"
            img_path = os.path.join(captured_img_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_path}")
            img_count += 1
            time.sleep(0.5)
        elif key == ord('q'):
            print("退出视频捕捉，开始自动标定...")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- 2. 自动标定计算 ---
    if img_count > 0:
        print(f"总共保存了 {img_count} 张图片，开始标定...")
        
        calibrator = Calibrator(captured_img_dir, shape_inner_corner, size_grid)
        mat_intri, coff_dis = calibrator.calibrate_camera()
        
        # --- 3. 保存标定结果到TXT文件 ---
        calib_results_path = os.path.join(test_run_dir, "calibration_results.txt")
        with open(calib_results_path, "w") as f:
            f.write("--- Camera Calibration Results ---\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Intrinsic Matrix K:\n{mat_intri}\n\n")
            f.write(f"Distortion Coefficients D:\n{coff_dis}\n")
        print(f"标定参数已保存至: {calib_results_path}")

        # --- 4. 绘制角点并拼接保存去畸变图片 ---
        img_paths = sorted(glob.glob(os.path.join(captured_img_dir, "*.jpg")))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, shape_inner_corner, None)
            
            if ret:
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, shape_inner_corner, corners, ret)
                corner_img_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(corner_detected_img_dir, corner_img_name), img_with_corners)

                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (img.shape[1], img.shape[0]), 0, (img.shape[1], img.shape[0]))
                undistorted_img = cv2.undistort(img, mat_intri, coff_dis, None, newcameramtx)
                combined_img = np.hstack([img, undistorted_img])
                
                combined_img_path = os.path.join(undistorted_img_dir, f"combined_{os.path.basename(img_path)}")
                cv2.imwrite(combined_img_path, combined_img)
                print(f"去畸变图片已保存到: {combined_img_path}")
    else:
        print("没有保存任何图片，标定过程已跳过。")


if __name__ == '__main__':
    main()