"""
基于几何特征法的单手势识别功能包
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
        """
        检测"比一"手势（食指伸直，其他手指弯曲）
        返回: (是否检测到, 满足条件数, 总条件数)
        """
        index = 0  # 满足条件数
        request = 9  # 增加总条件数为9
        standard = 9  # 提高标准到9

        handFeatureCalculator = HandFeatureCalculator()

        pre_landmarks_array = create_landmarks_array(hand_landmarks, image_shape)  # 21组三维点坐标
        # 仅使用x,y
        landmarks_array = []
        for landmark in pre_landmarks_array:
            x = landmark[0]
            y = landmark[1]
            landmarks_array.append(np.array([x, y]))

        # 原始距离计算
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

        # 新增距离计算
        distance_8_12 = np.linalg.norm([landmarks_array[8][0] - landmarks_array[12][0], 
                                    landmarks_array[8][1] - landmarks_array[12][1]])
        distance_0_5 = np.linalg.norm([landmarks_array[0][0] - landmarks_array[5][0], 
                                    landmarks_array[0][1] - landmarks_array[5][1]])
        distance_8_12 = np.linalg.norm([landmarks_array[8][0] - landmarks_array[12][0], 
                               landmarks_array[8][1] - landmarks_array[12][1]])
        distance_8_16 = np.linalg.norm([landmarks_array[8][0] - landmarks_array[16][0], 
                                    landmarks_array[8][1] - landmarks_array[16][1]])
        distance_8_20 = np.linalg.norm([landmarks_array[8][0] - landmarks_array[20][0], 
                                    landmarks_array[8][1] - landmarks_array[20][1]])
        # 手掌宽度参考（手腕到中指根部的距离）
        distance_0_9 = np.linalg.norm([landmarks_array[0][0] - landmarks_array[9][0], 
                                    landmarks_array[0][1] - landmarks_array[9][1]])

        # 防止除零
        if distance_0_9 < 0.001:
            distance_0_9 = 0.001

        # 角度计算
        angle_9_10_11 = handFeatureCalculator.calculate_angle(landmarks_array[9], landmarks_array[10], landmarks_array[11])
        angle_8_5_10 = handFeatureCalculator.calculate_angle(landmarks_array[8], landmarks_array[5], landmarks_array[10])
        angle_5_6_7 = handFeatureCalculator.calculate_angle(landmarks_array[5], landmarks_array[6], landmarks_array[7])
        angle_6_7_8 = handFeatureCalculator.calculate_angle(landmarks_array[6], landmarks_array[7], landmarks_array[8])
        angle_13_14_15 = handFeatureCalculator.calculate_angle(landmarks_array[13], landmarks_array[14], landmarks_array[15])
        angle_17_18_19 = handFeatureCalculator.calculate_angle(landmarks_array[17], landmarks_array[18], landmarks_array[19])

        # 同时检查拇指尖与中指、无名指尖的距离
        thumb_tip = landmarks_array[4]      # 拇指尖 (4)
        middle_tip = landmarks_array[12]    # 中指尖 (12)
        ring_tip = landmarks_array[16]      # 无名指尖 (16)

        # 防止除零错误
        if distance_12_16 < 0.001:
            distance_12_16 = 0.001
        if distance_5_8 < 0.001:
            distance_5_8 = 0.001
        if distance_0_9 < 0.001:
            distance_0_9 = 0.001

        # 条件1: 无名指和小指尖的距离与中指和无名指尖的距离比值接近1
        condition1_ratio = distance_16_20 / distance_12_16
        condition1_met = 0.40 < condition1_ratio < 1.40  # 收紧范围
        if debug:
            print(f"条件1 - 无名指小指尖距比: {condition1_ratio:.3f} {'满足' if condition1_met else '不满足'}")
        if condition1_met:
            index += 1

        # 条件2: 拇指尖到中指的MCP距离与食指长度的比值
        condition2_ratio = distance_4_10 / distance_5_8
        condition2_met = condition2_ratio < 1.0  # 收紧阈值
        if debug:
            print(f"条件2 - 拇指到中指距离比: {condition2_ratio:.3f} {'满足' if condition2_met else '不满足'}")
        if condition2_met:
            index += 1

        # 条件3: 拇指尖到无名指的MCP距离与食指长度的比值
        condition3_ratio = distance_4_14 / distance_5_8
        condition3_met = condition3_ratio < 1.0  # 收紧阈值
        if debug:
            print(f"条件3 - 拇指到无名指距离比: {condition3_ratio:.3f} {'满足' if condition3_met else '不满足'}")
        if condition3_met:
            index += 1

        # 条件4: 食指第一关节伸直（接近180度）
        condition4_diff = abs(angle_5_6_7 - 180)
        condition4_met = condition4_diff < 28  # 收紧偏差
        if debug:
            print(f"条件4 - 食指第一关节角度: {angle_5_6_7:.1f}度, 偏差: {condition4_diff:.1f}度 {'满足' if condition4_met else '不满足'}")
        if condition4_met:
            index += 1

        # 条件5: 食指第二关节伸直（接近180度）
        condition5_diff = abs(angle_6_7_8 - 180)
        condition5_met = condition5_diff < 33  # 收紧偏差
        if debug:
            print(f"条件5 - 食指第二关节角度: {angle_6_7_8:.1f}度, 偏差: {condition5_diff:.1f}度 {'满足' if condition5_met else '不满足'}")
        if condition5_met:
            index += 1

        # 条件6: 食指与手掌的角度接近90度
        condition6_diff = abs(angle_8_5_10 - 90)
        condition6_met = condition6_diff < 90  # 收紧偏差
        if debug:
            print(f"条件6 - 食指手掌角度: {angle_8_5_10:.1f}度, 偏差: {condition6_diff:.1f}度 {'满足' if condition6_met else '不满足'}")
        if condition6_met:
            index += 1

        # 新增条件7: 食指尖与中指尖的距离大于阈值（防止食指和中指同时伸出）
        condition7_met = distance_8_12 > distance_0_5 * 0.6
        if debug:
            print(f"条件7 - 食指中指距离比: {distance_8_12/distance_0_5:.3f} {'满足' if condition7_met else '不满足'}")
        if condition7_met:
            index += 1

        # 新增条件8: 微调
        condition8_met = (distance_8_12 > distance_0_9 * 0.92 and 
                  distance_8_16 > distance_0_9 * 0.92 and 
                  distance_8_20 > distance_0_9 * 0.92)
        if debug:
            print(f"条件8 - {'满足' if condition8_met else '不满足'}")
        if condition8_met:
            index += 1

        # 新增条件9：检查手掌的朝向（手腕到中指根部的向量方向）
        wrist = landmarks_array[0]        # 手腕 (0)
        middle_base = landmarks_array[9]  # 中指根部 (9)

        # 计算手腕到中指根部的向量
        vector_y = middle_base[1] - wrist[1]  # y方向分量

        # 判断手掌是否向下伸展
        # 在图像坐标系中，y坐标越大越靠下
        # 如果vector_y > 0，说明手掌向下伸展
        is_palm_down = vector_y > distance_0_9 * 0.16  # 使用相对阈值

        condition9_met = not is_palm_down  # 手掌不向下时才满足

        if debug:
            print(f"条件9 - {'满足' if condition9_met else '不满足'}")
        if condition9_met:
            index += 1

        return (index >= standard), index, request
    