# -*- coding: utf-8 -*-
"""
Calibrate the Fisheye Camera with a live feed.

By Gemini, based on You Zhiyuan's code, 2024.
"""

import os
import cv2
import time
import glob
import numpy as np


def main():
    """Capture fisheye calibration images live and run calibration."""
    # --- 1. 配置和路径设置 ---
    # 获取当前脚本所在目录，后续生成的所有标定结果都保存在此目录下的子文件夹中
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用时间戳区分每一次标定任务，方便回溯与比对不同实验的结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_run_dir = os.path.join(script_dir, f"test_run_{timestamp}")
    os.makedirs(test_run_dir, exist_ok=True)

    # 构造并创建三个核心输出目录：原始采集图、角点标注图、去畸变结果图
    captured_img_dir = os.path.join(test_run_dir, "captured_fisheye_images")
    corner_detected_img_dir = os.path.join(test_run_dir, "corner_detected_images")
    undistorted_img_dir = os.path.join(test_run_dir, "fisheye_undistortion_result")

    os.makedirs(captured_img_dir, exist_ok=True)
    os.makedirs(corner_detected_img_dir, exist_ok=True)
    os.makedirs(undistorted_img_dir, exist_ok=True)

    # 标定参数：棋盘格内角点数、单位边长、期望分辨率以及建议的最少采集张数
    shape_inner_corner = (12, 8)   # 棋盘格内角点数量 (列, 行)
    size_grid = 0.03               # 棋盘格单元尺寸 (米)
    desired_resolution = (1920, 1080)  # 目标分辨率 (宽, 高)
    min_capture_count = 15         # 建议最少采集的有效图片数量

    # 亚像素角点优化的迭代退出条件，高精度角点对于鱼眼标定尤为关键
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-4,
    )
    # 棋盘格检测参数：开启自适应阈值、归一化和快速检查以提升检测鲁棒性
    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    # --- 2. 打开摄像头并采集 ---
    # 默认使用索引为 0 的摄像头，可以根据需要改成其他索引或 RTSP 流
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！请检查设备或索引。")
        return

    # 尝试强制指定 MJPG 编码和 1920x1080 分辨率，避免 USB 摄像头默认的压缩或低分辨率输出
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_resolution[1])

    # 读取摄像头实际应用的分辨率，用于后续标定验证
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != desired_resolution:
        print(
            f"警告: 摄像头无法设置到 {desired_resolution[0]}x{desired_resolution[1]}，"
            f"实际分辨率为 {actual_width}x{actual_height}。"
        )
    else:
        print(f"摄像头分辨率已设置为 {actual_width}x{actual_height}。")

    # 记录成功保存的图像数量，便于退出时快速了解采集量
    img_count = 0
    while True:
        # 逐帧读取摄像头数据，实时显示供用户确认拍摄姿态
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧，退出。")
            break

        # 若摄像头在运行过程中切换分辨率，立即更新实际分辨率信息
        if frame.shape[1] != actual_width or frame.shape[0] != actual_height:
            actual_width = frame.shape[1]
            actual_height = frame.shape[0]
            print(
                f"检测到实时帧分辨率: {actual_width}x{actual_height}。"
                "将以此分辨率进行后续标定。"
            )

        # 在画面左上角叠加交互提示，指导用户按键操作
        info_text = (
            "Press 's' to save. Saved: "
            f"{img_count} | Press 'q' to quit and start calib."
        )
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Fisheye Camera Feed - Press 'q' to quit", frame)

        # 监听按键：s 保存当前帧，q 结束采集进入标定流程
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 以递增编号命名图片，保存原始彩色帧
            img_name = f"fisheye_calib_img_{img_count:02d}.jpg"
            img_path = os.path.join(captured_img_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_path}")
            img_count += 1
            time.sleep(0.5)
        elif key == ord('q'):
            print("退出视频捕捉，开始自动鱼眼镜头标定...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- 3. 鱼眼标定计算 ---
    if img_count == 0:
        # 完全没有采集到任何图片时，直接结束流程
        print("没有保存任何图片，标定过程已跳过。")
        return

    print(f"总共保存了 {img_count} 张图片，开始标定...")

    # 构造单幅棋盘格的三维物理坐标，默认棋盘格位于 z=0 平面
    w, h = shape_inner_corner
    objp = np.zeros((1, w * h, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp *= size_grid

    # objpoints/imgpoints 分别存放 3D/2D 对应点；expected_size 记录图像分辨率一致性
    objpoints = []
    imgpoints = []
    expected_size = None

    # 将采集的图片路径排序，保证标定过程按拍摄顺序处理
    img_paths = sorted(glob.glob(os.path.join(captured_img_dir, "*.jpg")))

    for img_path in img_paths:
        # 读取图片并转换为灰度，用于角点检测
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取 {img_path}，可能文件损坏。")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]

        if expected_size is None:
            expected_size = current_size
        elif current_size != expected_size:
            print(
                f"警告: {os.path.basename(img_path)} 分辨率 {current_size}"
                f" 与期望的 {expected_size} 不一致，已跳过。"
            )
            continue

        # 在灰度图中寻找棋盘格内角点，返回值 ret 表示检测成功与否
        ret, corners = cv2.findChessboardCorners(gray, (w, h), flags=chessboard_flags)
        if not ret:
            print(f"未能在 {os.path.basename(img_path)} 中检测到完整棋盘格角点。")
            continue

        # 使用亚像素优化角点坐标，提高标定精度
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), subpix_criteria)
        # 注意：必须 append objp 的拷贝，避免后续修改影响历史数据
        objpoints.append(objp.copy())
        imgpoints.append(corners)

        # 将检测完成的角点覆盖绘制到原图上，便于肉眼确认识别质量
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, (w, h), corners, ret)
        corner_img_name = os.path.basename(img_path)
        cv2.imwrite(
            os.path.join(corner_detected_img_dir, corner_img_name),
            img_with_corners,
        )
        print(
            f"在 {os.path.basename(img_path)} 中找到了角点，"
            f"已保存到 {corner_detected_img_dir}。"
        )

    # 统计有效样本数量，用于判断标定数据是否充足
    valid_sample_count = len(objpoints)
    if valid_sample_count == 0:
        print("未在任何图片中找到足够的棋盘格角点。请确保图片清晰且棋盘格完整。")
        return

    if valid_sample_count < min_capture_count:
        print(
            f"警告: 仅检测到 {valid_sample_count} 幅有效标定图，"
            f"建议至少 {min_capture_count} 幅以提升精度。"
            " 标定将继续，但结果可能不够稳定。"
        )

    print("正在进行鱼眼相机标定...")

    # OpenCV fisheye 接口需要使用 numpy 数组格式传入批量 3D/2D 点
    objpoints_np = np.asarray(objpoints, dtype=np.float32)
    imgpoints_np = np.asarray(imgpoints, dtype=np.float32)

    # 初始化内参矩阵和畸变系数矩阵，rvecs/tvecs 为每张图的外参（旋转、平移）
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    # 标定的迭代终止条件，精度越高迭代次数可能越多
    calibrate_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    # 常用的鱼眼标定选项：重新估计外参、条件数检查、固定偏斜
    calibrate_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_np,
            imgpoints_np,
            expected_size,
            K,
            D,
            rvecs,
            tvecs,
            flags=calibrate_flags,
            criteria=calibrate_criteria,
        )
    except cv2.error as err:
        if "CALIB_CHECK_COND" in str(err):
            # 当条件数检查失败时，去掉该约束重新尝试，以免整个流程终止
            fallback_flags = calibrate_flags & ~cv2.fisheye.CALIB_CHECK_COND
            print(
                "警告: 原始标定失败 (CALIB_CHECK_COND)。"
                " 尝试移除该约束并重新标定。"
            )
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints_np,
                imgpoints_np,
                expected_size,
                K,
                D,
                rvecs,
                tvecs,
                flags=fallback_flags,
                criteria=calibrate_criteria,
            )
        else:
            raise

    # RMS 表示平均重投影误差，数值越小说明标定越稳定
    print(f"RMS Error: {rms}")
    if rms > 5.0:
        print(
            "警告: 重投影误差较高 (RMS > 5)。"
            " 建议补充更多多角度、高质量的棋盘格图像以提升标定稳定性。"
        )
    print(f"Fisheye Intrinsic Matrix:\n{K}")
    print(f"Fisheye Distortion Coefficients:\n{D}")

    # 将标定结果完整写入文本文件，方便与其他模块共享或后续分析
    calib_results_path = os.path.join(test_run_dir, "calibration_results.txt")
    with open(calib_results_path, "w", encoding="utf-8") as f:
        f.write("--- Fisheye Camera Calibration Results ---\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Captured Frames: {img_count}\n")
        f.write(f"Valid Calibration Frames: {valid_sample_count}\n")
        f.write(
            f"Target Resolution: {desired_resolution[0]}x{desired_resolution[1]}\n"
        )
        f.write(
            f"Actual Calibration Resolution: {expected_size[0]}x{expected_size[1]}\n"
        )
        f.write(f"RMS (Reprojection) Error: {rms}\n")
        f.write(f"Intrinsic Matrix K:\n{K}\n\n")
        f.write(f"Distortion Coefficients D:\n{D}\n")

    print(f"标定参数已保存至: {calib_results_path}")

    # --- 4. 去畸变处理 ---
    print("\n正在对所有图片进行去畸变处理...")

    # balance 控制视场与黑边之间的折中，值越大视场越广但黑边也更多
    balance = 0.0  # 在保持视场的同时尽量裁剪黑边
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        expected_size,
        np.eye(3),
        new_size=expected_size,
        balance=balance,
        fov_scale=1.0,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        new_K,
        expected_size,
        cv2.CV_16SC2,
    )

    # 当检测到去畸图像接近全黑时，懒加载备用映射表（基于原始内参）
    fallback_maps = None

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        if img.shape[1] != expected_size[0] or img.shape[0] != expected_size[1]:
            print(
                f"跳过 {os.path.basename(img_path)}，其分辨率 {img.shape[1]}x{img.shape[0]}"
                f" 与标定分辨率 {expected_size[0]}x{expected_size[1]} 不一致。"
            )
            continue

        # 使用预先计算的映射表对原图进行去畸变
        undistorted_img = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # 通过非零像素比例判断图像是否出现“全黑”异常
        nonzero_ratio = np.count_nonzero(
            cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        ) / float(expected_size[0] * expected_size[1])

        if nonzero_ratio < 0.05:
            if fallback_maps is None:
                print(
                    "警告: 去畸变结果接近全黑，"
                    "尝试使用原始相机矩阵重新生成。"
                )
                # 使用原始内参 K 生成备用映射表，通常能得到较保守但不为全黑的结果
                map1_fb, map2_fb = cv2.fisheye.initUndistortRectifyMap(
                    K,
                    D,
                    np.eye(3),
                    K,
                    expected_size,
                    cv2.CV_16SC2,
                )
                fallback_maps = (map1_fb, map2_fb)
            map1_fb, map2_fb = fallback_maps
            # 使用备用映射表再做一次去畸变尝试
            undistorted_img = cv2.remap(
                img,
                map1_fb,
                map2_fb,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # 如果备用映射仍全黑，提示用户重新采集数据
            nonzero_ratio_fb = np.count_nonzero(
                cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
            ) / float(expected_size[0] * expected_size[1])
            if nonzero_ratio_fb < 0.05:
                print(
                    f"警告: 即使用备用矩阵，{os.path.basename(img_path)} 的去畸变结果仍接近全黑。"
                    " 请检查标定数据质量。"
                )

        # 拼接原图与去畸结果，方便横向对比
        combined_img = np.hstack([img, undistorted_img])

        img_name = os.path.basename(img_path)
        combined_img_path = os.path.join(
            undistorted_img_dir,
            f"combined_{img_name}",
        )
        # 最终输出去畸对比图，便于快速浏览所有样本的效果
        cv2.imwrite(combined_img_path, combined_img)
        print(f"去畸变图片已保存到: {combined_img_path}")


if __name__ == "__main__":
    main()