"""
特征提取器
"""

import numpy as np
import mediapipe as mp

class GestureFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # 21个关键点的索引
        self.landmark_indices = {
            'wrist': 0,
            'thumb': [1, 2, 3, 4],      # CMC, MCP, IP, Tip
            'index': [5, 6, 7, 8],      # MCP, PIP, DIP, Tip
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        # 手指名称列表
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        # 新增：统一的数值稳定性设置
        self.EPSILON = 1e-10
        self.MIN_PALM_SIZE = 0.001
        
        # 新增：特征归一化范围配置
        self.feature_ranges = {
            'angle': (0, 180),      # 角度范围
            'distance': (0, 3.0),   # 距离范围
            'curl': (0, 1.0),       # 弯曲比例范围
            'palm_dir': (-1, 1),    # 方向范围
            'state': (0, 1)         # 状态范围
        }
    
    def extract_features(self, landmarks):
        """
        主特征提取方法
        输入: MediaPipe landmarks列表
        输出: 扁平化的特征向量
        """
        features = {}
        
        # 1. 计算手掌参考尺寸（改进版本）
        palm_size = self._get_palm_size(landmarks)
        
        # 2. 提取各种特征
        features['finger_angles'] = self._get_finger_angles(landmarks)
        features['tip_distances'] = self._get_normalized_tip_distances(landmarks, palm_size)
        features['finger_curls'] = self._get_finger_curl_ratios(landmarks)
        features['palm_orientation'] = self._get_palm_orientation(landmarks)
        features['finger_states'] = self._get_finger_states(landmarks, palm_size)
        
        # 3. 转换为特征向量
        feature_vector = self._flatten_features(features)

        # 4. 特征后处理
        processed_features = self._post_process_features(feature_vector)

        return processed_features
    
    def _get_palm_size(self, landmarks):
        """
        计算手掌参考尺寸（改进版本）
        原理: 使用手腕到中指根部的距离作为归一化参考，增加数值稳定性
        """
        wrist = landmarks[self.landmark_indices['wrist']]
        middle_mcp = landmarks[self.landmark_indices['middle'][0]]
        
        # 使用改进的坐标转换方法
        wrist_pos = self._landmark_to_array(wrist)
        middle_mcp_pos = self._landmark_to_array(middle_mcp)
        
        distance = np.linalg.norm(wrist_pos - middle_mcp_pos)
        
        # 改进的防除零处理
        if distance < self.MIN_PALM_SIZE:
            # 备选方案：计算其他参考距离
            index_mcp = landmarks[self.landmark_indices['index'][0]]
            pinky_mcp = landmarks[self.landmark_indices['pinky'][0]]
            index_pos = self._landmark_to_array(index_mcp)
            pinky_pos = self._landmark_to_array(pinky_mcp)
            
            distance2 = np.linalg.norm(index_pos - pinky_pos)
            distance = max(distance, distance2, self.MIN_PALM_SIZE)
            
        return distance
    
    def _landmark_to_array(self, landmark):
        """统一的landmark坐标转换方法"""
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)
    
    def _get_finger_angles(self, landmarks):
        """
        计算每个手指相对于手掌平面的角度（改进版本）
        改进：使用更稳定的手掌平面计算方法
        """
        angles = {}
        
        # 1. 改进的手掌平面法向量计算（使用三个稳定点）
        wrist = self._landmark_to_array(landmarks[0])
        index_mcp = self._landmark_to_array(landmarks[5])
        pinky_mcp = self._landmark_to_array(landmarks[17])
        
        # 手掌平面的两个边向量
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        # 法向量 = v1 × v2 (叉积)
        palm_normal = np.cross(v1, v2)
        palm_normal_norm = np.linalg.norm(palm_normal)
        
        # 改进的归一化方法
        if palm_normal_norm > self.EPSILON:
            palm_normal = palm_normal / palm_normal_norm
        else:
            # 如果手掌平面计算失败，使用备份方案
            middle_mcp = self._landmark_to_array(landmarks[9])
            v3 = middle_mcp - wrist
            palm_normal = np.cross(v1, v3)
            palm_normal_norm = np.linalg.norm(palm_normal)
            
            if palm_normal_norm > self.EPSILON:
                palm_normal = palm_normal / palm_normal_norm
            else:
                palm_normal = np.array([0.0, 0.0, 1.0])
        
        # 2. 为每个手指计算角度（改进拇指计算）
        for finger in self.finger_names:
            # 手指MCP和Tip的索引
            mcp_idx = self.landmark_indices[finger][0]
            tip_idx = self.landmark_indices[finger][3]
            
            mcp = self._landmark_to_array(landmarks[mcp_idx])
            tip = self._landmark_to_array(landmarks[tip_idx])
            
            # 手指方向向量 (从MCP指向Tip)
            finger_vector = tip - mcp
            finger_norm = np.linalg.norm(finger_vector)
            
            if finger_norm > self.EPSILON:
                finger_vector = finger_vector / finger_norm
            else:
                finger_vector = np.array([0.0, 0.0, 1.0])
            
            # 计算角度: cosθ = a·b / (|a||b|)
            dot_product = np.dot(finger_vector, palm_normal)
            dot_product = np.clip(dot_product, -1.0 + self.EPSILON, 1.0 - self.EPSILON)  # 更安全的clip
            
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            # 角度范围控制在0-180度
            angles[finger] = float(min(max(angle_deg, 0.0), 180.0))
        
        return angles

    def _get_normalized_tip_distances(self, landmarks, palm_size):
        """
        计算归一化的指尖距离（改进版本）
        改进：增加更多相对距离特征
        """
        distances = {}
        
        # 1. 获取所有指尖的3D坐标（使用改进的方法）
        tip_indices = [4, 8, 12, 16, 20]
        tip_positions = [self._landmark_to_array(landmarks[idx]) for idx in tip_indices]
        
        # 2. 计算重要相对距离（归一化）
        # 拇指-食指距离
        thumb_index_dist = np.linalg.norm(tip_positions[0] - tip_positions[1])
        distances['thumb_index'] = float(thumb_index_dist / max(palm_size, self.MIN_PALM_SIZE))
        
        # 食指-中指距离
        index_middle_dist = np.linalg.norm(tip_positions[1] - tip_positions[2])
        distances['index_middle'] = float(index_middle_dist / max(palm_size, self.MIN_PALM_SIZE))
        
        # 新增：其他指尖距离组合
        thumb_middle_dist = np.linalg.norm(tip_positions[0] - tip_positions[2])
        distances['thumb_middle'] = float(thumb_middle_dist / max(palm_size, self.MIN_PALM_SIZE))
        
        ring_pinky_dist = np.linalg.norm(tip_positions[3] - tip_positions[4])
        distances['ring_pinky'] = float(ring_pinky_dist / max(palm_size, self.MIN_PALM_SIZE))
        
        # 3. 计算所有指尖到手腕的平均距离
        wrist_pos = self._landmark_to_array(landmarks[0])
        tip_to_wrist_dists = []
        
        for tip_pos in tip_positions:
            dist = np.linalg.norm(tip_pos - wrist_pos)
            tip_to_wrist_dists.append(dist / max(palm_size, self.MIN_PALM_SIZE))
        
        distances['avg_tip_distance'] = float(np.mean(tip_to_wrist_dists))
        distances['max_tip_distance'] = float(np.max(tip_to_wrist_dists))
        distances['min_tip_distance'] = float(np.min(tip_to_wrist_dists))
        
        # 新增：指尖距离的标准差
        if len(tip_to_wrist_dists) > 1:
            distances['std_tip_distance'] = float(np.std(tip_to_wrist_dists))
        else:
            distances['std_tip_distance'] = 0.0
        
        return distances
    
    def _get_finger_curl_ratios(self, landmarks):
        """
        计算手指弯曲程度比例（改进版本）
        改进：考虑拇指的特殊性
        """
        curl_ratios = {}
        
        for finger in self.finger_names:
            indices = self.landmark_indices[finger]
            
            # 获取该手指的所有关节点坐标
            points = np.array([self._landmark_to_array(landmarks[idx]) for idx in indices])
            
            # 改进：检查点是否有效
            if len(points) < 2:
                curl_ratios[finger] = 0.0
                continue
            
            # 计算曲线总长度
            curve_length = 0.0
            for i in range(len(points) - 1):
                segment_length = np.linalg.norm(points[i+1] - points[i])
                if not np.isnan(segment_length):
                    curve_length += segment_length
            
            # 计算直线长度（首尾点距离）
            straight_length = np.linalg.norm(points[-1] - points[0])
            
            # 改进的弯曲比例计算
            if straight_length > self.EPSILON:
                curl_ratio = (curve_length - straight_length) / straight_length
            else:
                curl_ratio = 0.0
            
            # 改进的范围控制
            curl_ratio = np.clip(curl_ratio, 0.0, 1.0)
            
            # 拇指特殊处理：因为拇指只有3个主要关节
            if finger == 'thumb':
                curl_ratio = curl_ratio * 1.2  # 稍微放大拇指的弯曲比例
                curl_ratio = min(curl_ratio, 1.0)
            
            curl_ratios[finger] = float(curl_ratio)
        
        return curl_ratios

    def _get_palm_orientation(self, landmarks):
        """
        计算手掌的2D方向向量（改进版本）
        改进：增加数值稳定性
        """
        # 使用手腕和食指MCP定义方向
        wrist = self._landmark_to_array(landmarks[0])
        index_mcp = self._landmark_to_array(landmarks[5])
        
        # 2D方向向量
        direction_x = index_mcp[0] - wrist[0]
        direction_y = index_mcp[1] - wrist[1]
        
        # 改进的归一化方法
        direction_norm = np.sqrt(direction_x**2 + direction_y**2 + self.EPSILON)
        
        if direction_norm > self.EPSILON:
            direction_x /= direction_norm
            direction_y /= direction_norm
        else:
            # 如果方向太小，使用默认方向
            direction_x, direction_y = 1.0, 0.0
        
        # 确保返回值类型一致
        return {
            'direction_x': float(direction_x),
            'direction_y': float(direction_y)
        }

    def _get_finger_states(self, landmarks, palm_size):
        """
        判断每个手指是否张开
        使用更可靠的方法：结合弯曲比例和相对距离
        """
        states = {}
        
        # 1. 计算手指弯曲比例
        curl_ratios = self._get_finger_curl_ratios(landmarks)
        
        # 2. 计算每个手指的相对长度（指尖到手腕 vs MCP到手腕）
        wrist = self._landmark_to_array(landmarks[0])
        
        for finger in self.finger_names:
            # 获取MCP和Tip的索引
            mcp_idx = self.landmark_indices[finger][0]
            tip_idx = self.landmark_indices[finger][3]
            
            mcp_pos = self._landmark_to_array(landmarks[mcp_idx])
            tip_pos = self._landmark_to_array(landmarks[tip_idx])
            
            # 计算两个距离
            tip_to_wrist = np.linalg.norm(tip_pos - wrist)
            mcp_to_wrist = np.linalg.norm(mcp_pos - wrist)
            
            # 计算长度比例：指尖距离 / MCP距离
            length_ratio = tip_to_wrist / max(mcp_to_wrist, self.MIN_PALM_SIZE)
            
            # 获取弯曲比例
            curl_ratio = curl_ratios.get(finger, 0.0)
            
            # 综合判断逻辑
            if finger == 'thumb':
                # 拇指的特殊处理：需要更高的阈值，因为拇指天生较长
                # 同时考虑拇指的角度（与其他手指不同）
                
                # 计算拇指的角度特征
                thumb_angle = self._get_thumb_angle(landmarks)
                
                # 改进的拇指判断：
                if curl_ratio > 0.3:  # 拇指有一定弯曲
                    # 弯曲的拇指：需要更高的长度比例才算张开
                    is_extended = length_ratio > 3.5 and thumb_angle < 100
                else:  # 拇指较直
                    # 直的拇指：也需要较高的长度比例
                    is_extended = length_ratio > 3.3 and thumb_angle < 100
                    
            elif finger == 'pinky':
                # 小指：需要更宽松的条件
                if curl_ratio > 0.7:
                    is_extended = length_ratio > 1.5
                else:
                    is_extended = length_ratio > 1.3
                    
            else:
                # 食指、中指、无名指
                if curl_ratio > 0.7:
                    # 弯曲程度较高
                    is_extended = length_ratio > 1.6
                elif curl_ratio > 0.4:
                    # 中等弯曲
                    is_extended = length_ratio > 1.4
                else:
                    # 较直
                    is_extended = length_ratio > 1.3
            
            states[finger] = 1 if is_extended else 0
        
        return states
    
    def _flatten_features(self, features):
        """
        将特征字典转换为扁平的特征向量（改进版本）
        确保特征顺序一致
        """
        feature_vector = []
        # 确保特征顺序的一致性
        # 1. 手指角度特征 (5个)
        if 'finger_angles' in features:
            for finger in self.finger_names:
                feature_vector.append(float(features['finger_angles'].get(finger, 0.0)))

        # 2. 指尖距离特征 (现在8个：原来的5个 + 新增的3个)
        if 'tip_distances' in features:
            # 保持原有顺序，增加新特征到后面
            distance_keys = ['thumb_index', 'index_middle', 'avg_tip_distance', 
                        'max_tip_distance', 'min_tip_distance']
            for key in distance_keys:
                feature_vector.append(float(features['tip_distances'].get(key, 0.0)))

            # 新增的距离特征
            new_distance_keys = ['thumb_middle', 'ring_pinky', 'std_tip_distance']
            for key in new_distance_keys:
                if key in features['tip_distances']:
                    feature_vector.append(float(features['tip_distances'][key]))
                else:
                    feature_vector.append(0.0)

        # 3. 手指弯曲特征 (5个)
        if 'finger_curls' in features:
            for finger in self.finger_names:
                feature_vector.append(float(features['finger_curls'].get(finger, 0.0)))

        # 4. 手掌方向特征 (2个)
        if 'palm_orientation' in features:
            orientation = features['palm_orientation']
            feature_vector.append(float(orientation.get('direction_x', 0.0)))
            feature_vector.append(float(orientation.get('direction_y', 0.0)))

        # 5. 手指状态特征 (5个)
        if 'finger_states' in features:
            for finger in self.finger_names:
                feature_vector.append(float(features['finger_states'].get(finger, 0.0)))

        # 转换为numpy数组，确保类型一致
        return np.array(feature_vector, dtype=np.float32)

    def _post_process_features(self, raw_features):
        """
        特征后处理方法（修复版）- 解决硬问题
        """
        # 1. 处理NaN和Inf
        processed = np.nan_to_num(raw_features, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 检查特征维度
        if len(processed) != 25:
            print(f"警告：特征维度{len(processed)}不等于25")
            return processed

        # 2. 特征索引
        angle_idx = slice(0, 5)      # 0-4: 手指角度
        distance_idx = slice(5, 13)  # 5-12: 指尖距离  
        curl_idx = slice(13, 18)     # 13-17: 手指弯曲
        palm_idx = slice(18, 20)     # 18-19: 手掌方向
        state_idx = slice(20, 25)    # 20-24: 手指状态

        # 3. 改进的归一化策略
        
        # 角度特征：使用非线性变换保留区分度
        angles = processed[angle_idx]
        # 使用sigmoid-like变换：保留原始比例，但限制在合理范围
        angles_norm = angles / 120.0  # 120度作为参考，而不是180
        angles_norm = 1.0 / (1.0 + np.exp(-angles_norm * 3))  # 平滑变换
        processed[angle_idx] = np.clip(angles_norm, 0.1, 0.9)  # 保留边界空间
        
        # 距离特征：动态归一化，允许超出范围
        distances = processed[distance_idx]
        # 计算动态范围（基于训练数据的经验值）
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        if dist_std > 0:
            # Z-score归一化，保留超出范围的值
            distances_norm = (distances - dist_mean) / (dist_std * 2)
        else:
            distances_norm = distances / 2.0  # 备选方案
        
        processed[distance_idx] = distances_norm
        
        # 弯曲特征：保持原样，轻微归一化
        curls = processed[curl_idx]
        # 弯曲程度通常在0-2之间，0=直，2=非常弯曲
        curls_norm = curls / 1.5
        processed[curl_idx] = np.clip(curls_norm, 0.0, 1.5)  # 允许超过1
        
        # 手掌方向：保持原逻辑（合理）
        palm = processed[palm_idx]
        processed[palm_idx] = (palm + 1.0) / 2.0
        
        # 状态特征：不要强制二值化！
        states = processed[state_idx]
        # 使用sigmoid保留渐变信息
        states_norm = 1.0 / (1.0 + np.exp(-states * 3))
        processed[state_idx] = states_norm  # 保持在0-1之间，但不是严格的0或1
        
        # 4. 最终处理：轻微缩放，不严格裁剪
        # 整体缩放到合理范围，但不强制在[0,1]
        processed = processed / 2.0  # 缩放到[-0.5, 1.5]左右
        
        # 防止极端值，但不丢失信息
        processed = np.clip(processed, -1.0, 2.0)
        
        return processed
    
    def _get_thumb_angle(self, landmarks):
        """
        计算拇指的角度（相对于手掌）
        拇指角度较小表示拇指与其他手指靠拢（闭合）
        拇指角度较大表示拇指张开
        """
        # 拇指的三个关键点：CMC, MCP, Tip
        cmc = self._landmark_to_array(landmarks[1])  # CMC
        mcp = self._landmark_to_array(landmarks[2])  # MCP  
        tip = self._landmark_to_array(landmarks[4])  # Tip
        
        # 计算两个向量
        v1 = mcp - cmc  # CMC到MCP
        v2 = tip - mcp  # MCP到Tip
        
        # 计算夹角
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > self.EPSILON and v2_norm > self.EPSILON:
            v1_normalized = v1 / v1_norm
            v2_normalized = v2 / v2_norm
            
            dot_product = np.dot(v1_normalized, v2_normalized)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
        
        return 90.0  # 默认值

    def get_feature_names(self):
        """
        获取特征名称列表（更新版本，匹配新的特征维度）
        """
        feature_names = []
        # 手指角度 (5个)
        for finger in self.finger_names:
            feature_names.append(f'angle_{finger}')

        # 指尖距离 (8个：5个原有 + 3个新增)
        distance_keys = ['thumb_index', 'index_middle', 'avg_tip_distance', 
                        'max_tip_distance', 'min_tip_distance',
                        'thumb_middle', 'ring_pinky', 'std_tip_distance']
        for key in distance_keys:
            feature_names.append(f'distance_{key}')

        # 手指弯曲程度 (5个)
        for finger in self.finger_names:
            feature_names.append(f'curl_{finger}')

        # 手掌朝向 (2个)
        feature_names.extend(['palm_dir_x', 'palm_dir_y'])

        # 手指状态 (5个)
        for finger in self.finger_names:
            feature_names.append(f'state_{finger}')

        return feature_names

    def debug_features(self, landmarks):
        """
        调试方法：打印所有特征的详细信息（兼容版本）
        """
        features = {}
        palm_size = self._get_palm_size(landmarks)

        features['finger_angles'] = self._get_finger_angles(landmarks)
        features['tip_distances'] = self._get_normalized_tip_distances(landmarks, palm_size)
        features['finger_curls'] = self._get_finger_curl_ratios(landmarks)
        features['palm_orientation'] = self._get_palm_orientation(landmarks)
        features['finger_states'] = self._get_finger_states(landmarks, palm_size)

        print("=" * 50)
        print("特征提取调试信息（改进版）")
        print("=" * 50)

        print(f"手掌参考尺寸: {palm_size:.4f}")

        print("\n1. 手指角度 (度):")
        for finger, angle in features['finger_angles'].items():
            print(f"  {finger:10s}: {angle:6.1f}°")

        print("\n2. 指尖距离 (归一化):")
        for key, dist in features['tip_distances'].items():
            print(f"  {key:20s}: {dist:.4f}")

        print("\n3. 手指弯曲比例 (0=直, 1=弯):")
        for finger, curl in features['finger_curls'].items():
            print(f"  {finger:10s}: {curl:.3f}")
        
        # 新增：计算并显示长度比例
        print("\n3.5 手指长度比例:")
        wrist = self._landmark_to_array(landmarks[0])
        for finger in self.finger_names:
            mcp_idx = self.landmark_indices[finger][0]
            tip_idx = self.landmark_indices[finger][3]
            
            mcp_pos = self._landmark_to_array(landmarks[mcp_idx])
            tip_pos = self._landmark_to_array(landmarks[tip_idx])
            
            tip_to_wrist = np.linalg.norm(tip_pos - wrist)
            mcp_to_wrist = np.linalg.norm(mcp_pos - wrist)
            length_ratio = tip_to_wrist / max(mcp_to_wrist, self.MIN_PALM_SIZE)
            
            print(f"  {finger:10s}: {length_ratio:.3f}")

        print("\n4. 手掌方向 (单位向量):")
        print(f"  方向X: {features['palm_orientation']['direction_x']:.3f}")
        print(f"  方向Y: {features['palm_orientation']['direction_y']:.3f}")

        print("\n5. 手指状态 (1=张开, 0=闭合):")
        for finger, state in features['finger_states'].items():
            print(f"  {finger:10s}: {state}")

        # 扁平化特征向量
        flattened = self._flatten_features(features)
        processed = self._post_process_features(flattened)

        print(f"\n6. 特征向量信息:")
        print(f"  原始特征维度: {len(flattened)}")
        print(f"  处理后维度: {len(processed)}")
        print(f"  数据类型: {processed.dtype}")

        # 显示特征摘要
        print(f"\n7. 特征值范围:")
        print(f"  最小值: {processed.min():.3f}")
        print(f"  最大值: {processed.max():.3f}")
        print(f"  平均值: {processed.mean():.3f}")

        return processed