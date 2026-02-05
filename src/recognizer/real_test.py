"""
real_test.py - 实时预测测试
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature.extractor import GestureFeatureExtractor

# 加载模型和组件
with open("models/gesture_model_20260205_175617.pkl", "rb") as f:
    model_data = pickle.load(f)

classifier = model_data['classifier']
scaler = model_data['scaler']
label_to_index = model_data['label_to_index']
index_to_label = model_data['index_to_label']

# 特征提取器
feature_extractor = GestureFeatureExtractor()

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

print("="*60)
print("实时预测测试")
print("按 'p' 键进行预测")
print("按 'q' 键退出")
print("="*60)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    # 检测手部
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 绘制手部点
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )
    
    # 显示状态
    status = "Hand Detected" if hand_detected else "No Hand"
    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Real Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('p') and hand_detected:
        print("\n" + "="*50)
        print("进行预测...")
        
        try:
            # 1. 提取特征
            features = feature_extractor.extract_features(hand_landmarks.landmark)
            print(f"特征提取成功: {len(features)}维")
            print(f"特征值: {features[:5]}...")  # 显示前5个值
            
            # 2. 标准化
            if hasattr(scaler, 'transform'):
                features_scaled = scaler.transform([features])
                print(f"标准化成功")
            else:
                features_scaled = [features]
                print("使用未标准化的特征")
            
            # 3. 预测
            if hasattr(classifier, 'predict'):
                prediction_idx = classifier.predict(features_scaled)[0]
                print(f"预测索引: {prediction_idx}")
                
                # 获取类别
                if prediction_idx in index_to_label:
                    gesture = index_to_label[prediction_idx]
                    print(f"预测手势: {gesture}")
                else:
                    print(f"未知索引: {prediction_idx}")
            
            # 4. 概率
            if hasattr(classifier, 'predict_proba'):
                probabilities = classifier.predict_proba(features_scaled)[0]
                print("\n所有类别概率:")
                for idx, prob in enumerate(probabilities):
                    gesture_name = index_to_label.get(idx, f"class_{idx}")
                    print(f"  {gesture_name}: {prob:.2%}")
            
        except Exception as e:
            print(f"预测失败: {e}")
            import traceback
            traceback.print_exc()

cap.release()
cv2.destroyAllWindows()
