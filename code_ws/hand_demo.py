import cv2
import mediapipe as mp

# 判断是否导入成功
# print(mp.__version__)  # 应输出版本号（如 0.10.0）
# print(cv2.__version__) 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 转换为RGB格式（MediaPipe需要）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 绘制关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()