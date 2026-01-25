"""
手部特征计算器方法试点
"""

import cv2
import mediapipe as mp
import numpy as np

class HandFeatureCalculator:
    '''手部特征计算器'''

    def __init__(self):
        pass
    
    def calculate_finger_lengths(self, landmarks_array):
        '''计算手指长度'''
        # 手部关键点索引
        finger_indices = {
            'thumb': [1, 2, 3, 4],        # 拇指
            'index': [5, 6, 7, 8],        # 食指
            'middle': [9, 10, 11, 12],    # 中指
            'ring': [13, 14, 15, 16],     # 无名指
            'pinky': [17, 18, 19, 20]     # 小指
        }

        lengths = {}

        for finger_name, indices in finger_indices.items():
            total_length = 0
            for i in range(len(indices) - 1):
                p1 = landmarks_array[indices[i]]
                p2 = landmarks_array[indices[i + 1]]
                segment_length = np.linalg.norm(p2 - p1)
                total_length += segment_length

            lengths[finger_name] = total_length
        
        return lengths

    def calculate_finger_angles(self, landmarks_array):
        '''计算手指关节角度'''
        angles = {}

        joint_angles = {
            'thumb_mcp': [1, 2, 3], 'thumb_ip': [2, 3, 4],
            'index_pip': [5, 6, 7], 'index_dip': [6, 7, 8],
            'middle_pip': [9, 10, 11], 'middle_dip': [10, 11, 12],
            'ring_pip': [13, 14, 15], 'ring_dip': [14, 15, 16],
            'pinky_pip': [17, 18, 19], 'pinky_dip': [18, 19, 20],  # 修正：thumb_pip -> pinky_pip
        }

        for joint_name, indices in joint_angles.items():
            p1 = landmarks_array[indices[0]]
            p2 = landmarks_array[indices[1]]
            p3 = landmarks_array[indices[2]]

            angle = self._calculate_angle(p1, p2, p3)
            angles[joint_name] = angle

        return angles
    
    def _calculate_angle(self, p1, p2, p3):
        '''计算三个点之间的角度'''
        # 计算向量
        v1 = p1 - p2  # 从 p2 指向 p1
        v2 = p3 - p2  # 从 p2 指向 p3

        # 计算向量夹角
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        # 避免除以零
        if norm_product == 0:
            return 0.0
            
        cos_angle = dot_product / norm_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在有效范围内
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def get_hand_orientation(self, landmarks_array):
        '''判断手部朝向'''
        # 使用手腕和手掌点判断
        wrist = landmarks_array[0]
        palm_center = (landmarks_array[5] + landmarks_array[17]) / 2

        # 计算方向向量
        direction = palm_center - wrist

        # 判断主要方向
        if abs(direction[0]) > abs(direction[1]):
            return 'horizontal' if direction[0] > 0 else 'horizontal'
        else:
            return 'vertical' if direction[1] > 0 else 'vertical'

    def is_hand_open(self, landmarks_array, threshold=2):
        '''判断手是否张开'''
        # 计算指尖到手腕的平均距离
        wrist = landmarks_array[0]
        finger_tips = landmarks_array[[4, 8, 12, 16, 20]]

        distances = [np.linalg.norm(tip - wrist) for tip in finger_tips]
        avg_distance = np.mean(distances)

        # 计算手掌宽度作为参考
        palm_width = np.linalg.norm(landmarks_array[5] - landmarks_array[17])

        ratio = avg_distance / palm_width

        return ratio > threshold

def cornerText(img, text, y_offset=0):
    '''在左上角显示文本'''
    cv2.putText(img, text, (10, 25 + y_offset*30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def create_landmarks_array(hand_landmarks, image_shape):
    '''将 MediaPipe Landmark 对象转换为 numpy 数组'''
    h, w = image_shape[:2]
    
    landmarks_array = []
    for landmark in hand_landmarks.landmark:
        # 转换为像素坐标
        x_px = landmark.x * w
        y_px = landmark.y * h
        z_px = landmark.z  # z坐标通常保持原值
        landmarks_array.append([x_px, y_px, z_px])
    
    return np.array(landmarks_array)

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
            
            # 打印详细信息到控制台
            print(f"\n=== 手 {hand_idx + 1} ===")
            print(f"朝向: {hand_orientation}")
            print(f"状态: {'张开' if is_open else '闭合'}")
            print("手指长度:")
            for finger, length in finger_lengths.items():
                print(f"  {finger}: {length:.1f} 像素")
            
            # 在图像上标注关键点编号
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                cv2.putText(image, f"{idx}", (x_px, y_px - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
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
