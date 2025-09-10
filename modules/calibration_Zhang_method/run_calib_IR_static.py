# -*- coding: utf-8 -*-
"""
Calibrate the IR Camera with static images.

This script processes a set of pre-captured images from a folder
to perform camera calibration and undistortion. It uses the dedicated OpenCV model.
"""

import os
import glob
import cv2
import numpy as np
import time
from calibrate_helper import Calibrator

def main():
    # --- 1. 配置和路径设置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_run_dir = os.path.join(script_dir, f"test_run_{timestamp}")
    os.makedirs(test_run_dir, exist_ok=True)
    
    # 路径配置
    img_dir = os.path.join(script_dir, "pic/IR_camera_calib_img")
    corner_detected_img_dir = os.path.join(test_run_dir, "corner_detected_images")
    undistorted_img_dir = os.path.join(test_run_dir, "undistorted_images")
    os.makedirs(corner_detected_img_dir, exist_ok=True)
    os.makedirs(undistorted_img_dir, exist_ok=True)
    
    shape_inner_corner = (11, 8)
    size_grid = 0.02

    # --- 2. 标定计算 ---
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
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
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, shape_inner_corner, None)
        
        if ret:
            # 绘制角点并保存
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, shape_inner_corner, corners, ret)
            corner_img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(corner_detected_img_dir, corner_img_name), img_with_corners)

            # 去畸变并拼接图片
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (img.shape[1], img.shape[0]), 0, (img.shape[1], img.shape[0]))
            undistorted_img = cv2.undistort(img, mat_intri, coff_dis, None, newcameramtx)
            combined_img = np.hstack([img, undistorted_img])
            
            combined_img_path = os.path.join(undistorted_img_dir, f"combined_{os.path.basename(img_path)}")
            cv2.imwrite(combined_img_path, combined_img)
            print(f"去畸变图片已保存到: {combined_img_path}")
            
    print("\n标定和去畸变已完成！")

if __name__ == '__main__':
    main()