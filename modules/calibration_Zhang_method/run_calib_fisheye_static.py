# -*- coding: utf-8 -*-
"""
Calibrate the Fisheye Camera with static images.

This script processes a set of pre-captured images from a folder
to perform fisheye camera calibration and undistortion.
It uses the dedicated OpenCV fisheye model.
"""

import os
import glob
import cv2
import numpy as np
import time


def main():
    # --- 1. 配置和路径设置 ---
    # 获取脚本文件所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据时间创建本次测试的专用根文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_run_dir = os.path.join(script_dir, f"test_run_{timestamp}")
    os.makedirs(test_run_dir, exist_ok=True)
    
    # 在专用文件夹下创建子目录
    corner_detected_img_dir = os.path.join(test_run_dir, "corner_detected_images")
    undistorted_img_dir = os.path.join(test_run_dir, "fisheye_undistortion_result")
    os.makedirs(corner_detected_img_dir, exist_ok=True)
    os.makedirs(undistorted_img_dir, exist_ok=True)
    
    # 图片文件夹路径，请替换为您的图片所在目录
    img_dir = os.path.join(script_dir, "e:/GithubRepo/Gesture-DetectionRecognition/modules/Calibration-ZhangZhengyou-Method/test_run_20250910_220219/captured_fisheye_images")
    
    # 棋盘格内角点数量 (cols, rows)，例如 11x8，与您之前使用的棋盘格相符
    shape_inner_corner = (11, 8)
    # 棋盘格的物理尺寸，单位为米
    size_grid = 0.02

    # --- 2. 准备标定数据 ---
    w, h = shape_inner_corner
    objp = np.zeros((1, w * h, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp = objp * size_grid

    objpoints = []
    imgpoints = []

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        print("未在指定的文件夹中找到任何图片。请检查路径和图片格式。")
        return

    print("正在处理图片并查找棋盘格角点...")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # 绘制角点并保存到新目录
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, (w, h), corners, ret)
            corner_img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(corner_detected_img_dir, corner_img_name), img_with_corners)
            print(f"在 {os.path.basename(img_path)} 中找到了角点，已保存到 {corner_detected_img_dir}。")
            
    if len(objpoints) == 0:
        print("未在任何图片中找到足够的棋盘格角点。请确保图片清晰且棋盘格完整。")
        return


    # --- 3. 鱼眼相机标定 ---
    print("\n所有图片处理完毕，开始鱼眼相机标定...")
    
    objpoints_np = np.asarray(objpoints, dtype=np.float32)
    imgpoints_np = np.asarray(imgpoints, dtype=np.float32)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints_np,
        imgpoints_np,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_FIX_SKEW
    )

    print("\n--- 标定结果 ---")
    print(f"RMS (重投影) 误差: {rms}")
    print(f"鱼眼内参矩阵 K:\n{K}")
    print(f"鱼眼畸变系数 D:\n{D}")
    
    # --- 保存标定结果到TXT文件 ---
    calib_results_path = os.path.join(test_run_dir, "calibration_results.txt")
    with open(calib_results_path, "w") as f:
        f.write("--- Fisheye Camera Calibration Results ---\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"RMS (Reprojection) Error: {rms}\n")
        f.write(f"Intrinsic Matrix K:\n{K}\n\n")
        f.write(f"Distortion Coefficients D:\n{D}\n")
    print(f"标定参数已保存至: {calib_results_path}")


    # --- 4. 鱼眼去畸变并拼接保存 ---
    print("\n正在对所有图片进行去畸变处理...")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        dim = img.shape[:2][::-1]
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        combined_img = np.hstack([img, undistorted_img])
        
        img_name = os.path.basename(img_path)
        combined_img_path = os.path.join(undistorted_img_dir, f"combined_{img_name}")
        cv2.imwrite(combined_img_path, combined_img)
        print(f"去畸变图片已保存到: {combined_img_path}")

    print("\n鱼眼镜头标定和去畸变已完成！")


if __name__ == '__main__':
    main()