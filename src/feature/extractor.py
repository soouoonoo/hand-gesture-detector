"""
特征提取器框架，请修改
"""

import numpy as np
import mediapipe as mp

class GestureFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.landmark_indices = {
            'wrist': 0,
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
    
    def extract_features(self, landmarks):
        """提取标准化特征向量"""
        features = {}
        
        # 1. 手掌尺寸归一化
        palm_size = self._get_palm_size(landmarks)
        
        # 2. 提取手指角度特征
        features['finger_angles'] = self._get_finger_angles(landmarks)
        
        # 3. 提取指尖距离特征（归一化）
        features['tip_distances'] = self._get_normalized_tip_distances(landmarks, palm_size)
        
        # 4. 手指弯曲程度
        features['finger_curls'] = self._get_finger_curl_ratios(landmarks)
        
        # 5. 手掌朝向（法向量）
        features['palm_orientation'] = self._get_palm_orientation(landmarks)
        
        # 转换为特征向量
        feature_vector = self._flatten_features(features)
        return feature_vector
    
    def _get_palm_size(self, landmarks):
        """计算手掌参考尺寸（手腕到中指根部距离）"""
        wrist = landmarks[self.landmark_indices['wrist']]
        middle_mcp = landmarks[self.landmark_indices['middle'][0]]
        return np.linalg.norm(np.array([wrist.x, wrist.y]) - 
                              np.array([middle_mcp.x, middle_mcp.y]))
    
    def _get_finger_angles(self, landmarks):
        """计算每个手指相对于手掌的角度"""
        # 实现细节...
        pass
    
    # 其他方法实现...