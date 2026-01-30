"""
分类器效果测试框架，请修改
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) #添加引用路径

from feature.classifier import GestureClassifier
from feature.extractor import GestureFeatureExtractor


class GestureRecognitionSystem:
    def __init__(self):
        self.feature_extractor = GestureFeatureExtractor()
        self.classifier = GestureClassifier()
        
        # 状态管理
        self.current_gesture = None
        self.gesture_history = []
        self.is_training_mode = False
        self.training_data = []\
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 1. 使用MediaPipe检测手部
        results = self.detect_hands(frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        # 2. 为每只手处理
        gestures = []
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取特征并分类
            gesture_result = self.classifier.predict(hand_landmarks.landmark)
            
            # 应用时间平滑（减少抖动）
            smoothed_gesture = self._apply_temporal_smoothing(gesture_result)
            
            gestures.append(smoothed_gesture)
        
        return gestures
    
    def start_training_mode(self, gesture_name):
        """进入训练模式，收集样本"""
        self.is_training_mode = True
        self.current_training_label = gesture_name
        self.training_samples = []
    
    def collect_sample(self, landmarks):
        """收集训练样本"""
        if self.is_training_mode:
            self.training_samples.append(landmarks)
    
    def finish_training(self):
        """完成训练"""
        if self.training_samples:
            # 为每个样本添加标签
            training_data = [
                (sample, self.current_training_label) 
                for sample in self.training_samples
            ]
            self.classifier.train(training_data)
        
        self.is_training_mode = False
    
    def _apply_temporal_smoothing(self, current_prediction):
        """使用滑动窗口平滑预测结果"""
        self.gesture_history.append(current_prediction)
        
        # 只保留最近N个预测
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)
        
        # 投票决定最终结果
        if len(self.gesture_history) >= 5:
            recent_predictions = self.gesture_history[-5:]
            # 选择出现次数最多的手势
            # 实现细节...
        
        return current_prediction