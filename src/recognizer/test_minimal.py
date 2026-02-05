"""
test_minimal.py - 最小化测试脚本
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 加载模型
with open("models/gesture_model_20260205_175617.pkl", "rb") as f:
    model_data = pickle.load(f)

print("模型信息:")
print(f"类别: {model_data.get('class_labels', [])}")
print(f"分类器类型: {model_data.get('classifier_type', 'unknown')}")

# 测试摄像头
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    # 检测手部
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        print("\n" + "="*50)
        print("检测到手部!")
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 简单检查landmarks
        print(f"Landmarks数量: {len(hand_landmarks.landmark)}")
        
        # 检查前几个点的坐标
        for i in range(3):
            lm = hand_landmarks.landmark[i]
            print(f"点{i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
        
        # 检查模型是否能处理
        classifier = model_data.get('classifier')
        if classifier:
            print("分类器对象存在")
        else:
            print("分类器对象不存在!")
        
        break  # 只测试一帧
    
    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()