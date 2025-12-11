# -*- coding: utf-8 -*-
"""
验证 MATLAB 导出的鱼眼标定参数在 OpenCV 中的去畸效果。
- 从 calibration_data/fisheye_calibration.json 读取参数
- 先对一张虚拟图像执行一次去畸烟雾测试
- 可选: 打开摄像头显示去畸对比（按 q 退出）
"""
import os
import sys
import time
from typing import Optional

import numpy as np
import cv2

# 将 modules 加入路径，便于按主程序方式导入
ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(ROOT, 'modules')
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from camera_calibration import FisheyeCalibrator, DistortionCorrector


def smoke_test(calib_path: str) -> bool:
    print(f"读取标定文件: {calib_path}")
    calib = FisheyeCalibrator()
    if not calib.load_calibration(calib_path):
        print("❌ 加载失败")
        return False
    print("✅ 加载成功")

    corr = DistortionCorrector(calib.calibration_result)

    # 构造一张与标定尺寸一致的测试图
    w, h = calib.calibration_result.image_size
    if w <= 0 or h <= 0:
        w, h = 640, 480
    test = np.zeros((h, w, 3), dtype=np.uint8)
    # 绘制参考图形
    cv2.circle(test, (w//2, h//2), min(w,h)//4, (255,255,255), 2)
    cv2.rectangle(test, (w//4, h//4), (3*w//4, 3*h//4), (0,255,0), 2)
    cv2.line(test, (0, h//2), (w, h//2), (255,0,0), 2)
    cv2.line(test, (w//2, 0), (w//2, h), (255,0,0), 2)

    out = corr.correct_distortion(test)
    if out is None or out.shape != test.shape:
        print("❌ 去畸烟雾测试失败")
        return False
    nz = np.count_nonzero(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
    print(f"✅ 去畸烟雾测试通过，非零像素: {nz}")
    side = np.hstack([test, out])
    cv2.imshow('SmokeTest: left=orig, right=undist', side)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    return True


def live_preview(calib_path: str, cam_id: int = 0):
    print("打开摄像头进行实时预览 (按 q 退出)...")
    calib = FisheyeCalibrator()
    if not calib.load_calibration(calib_path):
        print("❌ 标定文件加载失败，无法实时预览")
        return

    corr = DistortionCorrector(calib.calibration_result)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 尝试设置到标定分辨率
    w, h = calib.calibration_result.image_size
    if w > 0 and h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取摄像头帧失败")
            break
        undist = corr.correct_distortion(frame)
        vis = np.hstack([frame, undist])
        cv2.imshow('Fisheye Verify: left=orig, right=undist (q to quit)', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    calib_path = os.path.join(ROOT, 'calibration_data', 'fisheye_calibration.json')
    ok = smoke_test(calib_path)
    if not ok:
        sys.exit(1)
    # 注: 若你的 OpenCV 未包含 fisheye，请安装 opencv-contrib-python
    # 可选实时预览
    try:
        live_preview(calib_path)
    except Exception as e:
        print(f"实时预览失败: {e}")
