"""
泛用新增方法试点
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) #添加引用路径

from src.multiple_hand_gestures import MultipleHandGestures
from src.utils import HandFeatureCalculator,cornerText,create_landmarks_array

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 初始化手部模型
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 初始化特征计算器
feature_calculator = HandFeatureCalculator()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # 镜像显示
    image = cv2.flip(image, 1)
    h, w = image.shape[:2]

    # 转换颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像，检测手部
    results = hands.process(image_rgb)

    # 绘制手部关键点
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 绘制关键点和连接线
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # 将 MediaPipe 关键点转换为 numpy 数组
            landmarks_array = create_landmarks_array(hand_landmarks, (h, w))
            
            # 计算特征
            finger_lengths = feature_calculator.calculate_finger_lengths(landmarks_array)
            finger_angles = feature_calculator.calculate_finger_angles(landmarks_array)
            hand_orientation = feature_calculator.get_hand_orientation(landmarks_array)
            is_open = feature_calculator.is_hand_open(landmarks_array)
            
            # 显示结果
            cornerText(image, f"hand {hand_idx+1}: {hand_orientation}", hand_idx*5)
            cornerText(image, f"situation: {'open' if is_open else 'close'}", hand_idx*5 + 1)
            
            # 显示手指长度（只显示前两个手指，避免太多信息）
            if len(finger_lengths) > 0:
                for i, (finger, length) in enumerate(list(finger_lengths.items())[:2]):
                    cornerText(image, f"{finger}: {length:.1f}px", hand_idx*5 + 2 + i)
            
            # 在图像上标注关键点编号
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                cv2.putText(image, f"{idx}", (x_px, y_px - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                

            ############以下是试验区#############
            result=MultipleHandGestures.DetectNumberOne(hand_landmarks,[800,500],True)
            cv2.putText(image, f"{result[0]}", (400, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ###################################
    
    # 如果没有检测到手，显示提示
    else:
        cornerText(image, "fail", 0)
    
    cv2.imshow('Hand Detection', image)
    
    # 按 ESC 退出
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()