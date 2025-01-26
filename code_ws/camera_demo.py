import cv2

# 尝试打开默认摄像头（索引号一般为0）
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    print("可能原因：")
    print("1. 摄像头被其他程序占用（如微信、Zoom）")
    print("2. 系统未授予Python摄像头权限")
    print("3. 摄像头硬件故障")
    exit()

# 设置窗口名称
window_name = 'Camera Test'

try:
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取摄像头画面！")
            break
        
        # 显示画面
        cv2.imshow(window_name, frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已释放")