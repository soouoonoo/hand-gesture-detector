
"""
多手势识别核心功能包
"""
import mediapipe as mp
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) #添加引用路径

from src.utils import HandDetector, HandFeatureCalculator, create_landmarks_array

detector = HandDetector()
class MultipleHandGestures:
    def __init__(self):
        pass
    
    def DetectNumberOne(hand_landmarks, image_shape, debug=False):
        index = 0 # 满足条件数
        request = 7 # 条件总数
        standard = 6 # 标准
        
        handFeatureCalculator = HandFeatureCalculator()

        pre_landmarks_array = create_landmarks_array(hand_landmarks, image_shape)  # 21组三维点坐标
        # 仅使用x,y
        landmarks_array = []
        for landmark in pre_landmarks_array:
            x = landmark[0]
            y = landmark[1]
            landmarks_array.append(np.array([x, y]))
        
        # 距离计算
        distance_16_20 = np.linalg.norm([landmarks_array[16][0] - landmarks_array[20][0], 
                                         landmarks_array[16][1] - landmarks_array[20][1]])
        distance_12_16 = np.linalg.norm([landmarks_array[12][0] - landmarks_array[16][0], 
                                         landmarks_array[12][1] - landmarks_array[16][1]])
        distance_4_10 = np.linalg.norm([landmarks_array[4][0] - landmarks_array[10][0], 
                                        landmarks_array[4][1] - landmarks_array[10][1]])
        distance_4_14 = np.linalg.norm([landmarks_array[4][0] - landmarks_array[14][0], 
                                        landmarks_array[4][1] - landmarks_array[14][1]])
        distance_5_8 = np.linalg.norm([landmarks_array[5][0] - landmarks_array[8][0], 
                                       landmarks_array[5][1] - landmarks_array[8][1]])
        
        # 角度计算
        angle_9_10_11 = handFeatureCalculator.calculate_angle(landmarks_array[9], landmarks_array[10], landmarks_array[11])
        angle_8_5_10 = handFeatureCalculator.calculate_angle(landmarks_array[8], landmarks_array[5], landmarks_array[10])
        angle_5_6_7 = handFeatureCalculator.calculate_angle(landmarks_array[5], landmarks_array[6], landmarks_array[7])
        angle_6_7_8 = handFeatureCalculator.calculate_angle(landmarks_array[6], landmarks_array[7], landmarks_array[8])
        
        # 防止除零错误
        if distance_12_16 < 0.001:
            distance_12_16 = 0.001
        if distance_5_8 < 0.001:
            distance_5_8 = 0.001
        
        # 条件1: 无名指和小指尖的距离与中指和无名指尖的距离比值接近1
        condition1_ratio = distance_16_20 / distance_12_16
        condition1_met = abs(condition1_ratio - 1) < 0.6
        if debug:
            print(f"条件1 - 无名指小指尖距比: {condition1_ratio:.3f} {'满足' if condition1_met else '不满足'}")
        if condition1_met:
            index += 1
        
        # 条件2: 拇指尖到中指的MCP距离与食指长度的比值
        condition2_ratio = distance_4_10 / distance_5_8
        condition2_met = condition2_ratio < 1.2
        if debug:
            print(f"条件2 - 拇指到中指距离比: {condition2_ratio:.3f} {'满足' if condition2_met else '不满足'}")
        if condition2_met:
            index += 1
        
        # 条件3: 拇指尖到无名指的MCP距离与食指长度的比值
        condition3_ratio = distance_4_14 / distance_5_8
        condition3_met = condition3_ratio < 1.2
        if debug:
            print(f"条件3 - 拇指到无名指距离比: {condition3_ratio:.3f} {'满足' if condition3_met else '不满足'}")
        if condition3_met:
            index += 1
        
        # 条件4: 中指弯曲（角度小于100度）
        condition6_met = angle_9_10_11 < 110
        if debug:
            print(f"条件4 - 中指弯曲角度: {angle_9_10_11:.1f}度 {'满足' if condition6_met else '不满足'}")
        if condition6_met:
            index += 1

        # 条件5: 食指第一关节伸直（接近180度）
        condition7_diff = abs(angle_5_6_7 - 180)
        condition7_met = condition7_diff < 30
        if debug:
            print(f"条件5 - 食指第一关节角度: {angle_5_6_7:.1f}度, 偏差: {condition7_diff:.1f}度 {'满足' if condition7_met else '不满足'}")
        if condition7_met:
            index += 1

        # 条件6: 食指第二关节伸直（接近180度）
        condition8_diff = abs(angle_6_7_8 - 180)
        condition8_met = condition8_diff < 40
        if debug:
            print(f"条件6 - 食指第二关节角度: {angle_6_7_8:.1f}度, 偏差: {condition8_diff:.1f}度 {'满足' if condition8_met else '不满足'}")
        if condition8_met:
            index += 1

        # 条件7: 食指与手掌的角度接近90度
        condition9_diff = abs(angle_8_5_10 - 90)
        condition9_met = condition9_diff < 90  
        if debug:
            print(f"条件7 - 食指手掌角度: {angle_8_5_10:.1f}度, 偏差: {condition9_diff:.1f}度 {'满足' if condition9_met else '不满足'}")
        if condition9_met:
            index += 1

        return (index>=standard),index,request