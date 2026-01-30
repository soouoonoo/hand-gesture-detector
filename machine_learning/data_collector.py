import cv2
import mediapipe as mp
import numpy as np
import os
import time

# 初始化MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureDataCollector:
    def __init__(self, data_dir="gesture_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 创建标签映射：手势0-10
        self.label_map = {
            0: "zero",      # 拳头
            1: "one",       # 食指
            2: "two",       # 胜利
            3: "three",     # 三指
            4: "four",      # 四指
            5: "five",      # 手掌
            6: "six",       # 小指+拇指
            7: "seven",     # OK手势
            8: "eight",     # 手枪
            9: "nine",      # 九
            10: "ten"       # 交叉手指
        }
        
        # 存储数据
        self.features = []
        self.labels = []
        
    def collect_gesture(self, gesture_label, num_samples=100):
        """收集指定手势的数据"""
        print(f"\n准备收集手势: {self.label_map[gesture_label]} (编号: {gesture_label})")
        print(f"按空格键保存样本，按'q'键结束收集")
        
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            
            collected = 0
            while collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测手部
                results = hands.process(rgb_frame)
                
                # 显示提示
                cv2.putText(frame, f"Gesture: {self.label_map[gesture_label]}", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Collected: {collected}/{num_samples}", 
                          (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "SPACE: Save  Q: Quit", 
                          (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if results.multi_hand_landmarks:
                    # 只取第一只手
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 显示按空格提示
                    cv2.putText(frame, "Hand detected! Press SPACE", 
                              (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # 空格键保存
                        # 提取特征（21个关键点的x,y坐标）
                        landmarks = hand_landmarks.landmark
                        feature = []
                        for lm in landmarks:
                            feature.extend([lm.x, lm.y])  # 忽略z坐标
                        
                        self.features.append(feature)
                        self.labels.append(gesture_label)
                        collected += 1
                        print(f"✓ 已收集 {collected}/{num_samples}")
                        
                    elif key == ord('q'):
                        break
                
                cv2.imshow(f"Collect Gesture {gesture_label}", frame)
                cv2.waitKey(1)
            
            print(f"✅ 手势 {self.label_map[gesture_label]} 收集完成")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_data(self):
        """保存所有数据"""
        if len(self.features) > 0:
            np.save(os.path.join(self.data_dir, "features.npy"), np.array(self.features))
            np.save(os.path.join(self.data_dir, "labels.npy"), np.array(self.labels))
            print(f"✅ 数据已保存: {len(self.features)} 个样本")
        else:
            print("⚠️ 没有数据可保存")

if __name__ == "__main__":
    collector = GestureDataCollector()
    
    # 依次收集0-10的手势
    for gesture_id in range(11):
        input(f"\n按回车开始收集手势 {gesture_id} ({collector.label_map[gesture_id]})...")
        collector.collect_gesture(gesture_id, num_samples=50)  # 每个手势50个样本
    
    collector.save_data()