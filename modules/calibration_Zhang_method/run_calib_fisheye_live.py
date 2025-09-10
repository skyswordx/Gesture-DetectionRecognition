# -*- coding: utf-8 -*-
"""
Calibrate the Fisheye Camera with a live feed.

By Gemini, based on You Zhiyuan's code, 2024.
"""

import os
import cv2
import time
import numpy as np
import glob

def main():
    # --- 1. 配置和路径设置 ---
    # 获取脚本文件所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据时间创建本次测试的专用根文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_run_dir = os.path.join(script_dir, f"test_run_{timestamp}")
    os.makedirs(test_run_dir, exist_ok=True)
    
    # 在专用文件夹下创建子目录
    captured_img_dir = os.path.join(test_run_dir, "captured_fisheye_images")
    corner_detected_img_dir = os.path.join(test_run_dir, "corner_detected_images")
    undistorted_img_dir = os.path.join(test_run_dir, "fisheye_undistortion_result")
    
    os.makedirs(captured_img_dir, exist_ok=True)
    os.makedirs(corner_detected_img_dir, exist_ok=True)
    os.makedirs(undistorted_img_dir, exist_ok=True)
    
    shape_inner_corner = (11, 8)  # 棋盘格内角点数量
    size_grid = 0.02              # 棋盘格尺寸，单位为米
    
    cap = cv2.VideoCapture(1)  # 打开默认摄像头，如果不是0，请尝试1, 2等
    if not cap.isOpened():
        print("无法打开摄像头！请检查设备或索引。")
        return

    img_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧，退出。")
            break

        # 在图像上显示提示信息
        info_text = f"Press 's' to save. Saved: {img_count} | Press 'q' to quit and start calib."
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Live Fisheye Camera Feed - Press \'q\' to quit', frame)

        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 保存当前帧
            img_name = f"fisheye_calib_img_{img_count:02d}.jpg"
            img_path = os.path.join(captured_img_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_path}")
            img_count += 1
            # 短暂等待以避免连续保存
            time.sleep(0.5)
            
        elif key == ord('q'):
            print("退出视频捕捉，开始自动鱼眼镜头标定...")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- 2. 自动鱼眼镜头标定计算 ---
    if img_count > 0:
        print(f"总共保存了 {img_count} 张图片，开始标定...")

        w, h = shape_inner_corner
        objp = np.zeros((1, w * h, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        objp = objp * size_grid

        objpoints = []
        imgpoints = []

        img_paths = sorted(glob.glob(os.path.join(captured_img_dir, "*.jpg")))
        
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

        if len(objpoints) > 0:
            print("正在进行鱼眼相机标定...")
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

            print(f"RMS Error: {rms}")
            print(f"Fisheye Intrinsic Matrix:\n{K}")
            print(f"Fisheye Distortion Coefficients:\n{D}")
            
            # --- 3. 保存标定结果到TXT文件 ---
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
                
                # 拼接图片
                combined_img = np.hstack([img, undistorted_img])
                
                img_name = os.path.basename(img_path)
                combined_img_path = os.path.join(undistorted_img_dir, f"combined_{img_name}")
                cv2.imwrite(combined_img_path, combined_img)
                print(f"去畸变图片已保存到: {combined_img_path}")
        else:
            print("未在任何图片中找到足够的棋盘格角点。请确保图片清晰且棋盘格完整。")
    else:
        print("没有保存任何图片，标定过程已跳过。")


if __name__ == '__main__':
    main()