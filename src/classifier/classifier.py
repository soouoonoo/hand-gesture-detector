"""
分类器简单框架（以SVM为例），请修改；
要基于写好的feature/extractor.py(做完可以在那个test_classifier用来测试)；
随便用AI，实现思路简单些，只要保证性能好、检测准确就行，需要反复调整；
做好之后就push，或者直接发给我，我再改改
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) #添加引用路径

from feature.extractor import GestureFeatureExtractor 

class GestureClassifier:
    def __init__(self, classifier_type='svm'):
        self.feature_extractor = GestureFeatureExtractor()
        
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100)
        
        self.label_encoder = {}
        self.labels = []
    
    def train(self, training_data):
        """
        training_data格式: [(landmarks_list, label_str), ...]
        """
        X, y = [], []
        
        for landmarks, label in training_data:
            # 提取特征
            features = self.feature_extractor.extract_features(landmarks)
            X.append(features)
            
            # 编码标签
            if label not in self.label_encoder:
                self.label_encoder[label] = len(self.label_encoder)
                self.labels.append(label)
            y.append(self.label_encoder[label])
        
        # 训练分类器
        self.classifier.fit(X, y)
    
    def predict(self, landmarks):
        """预测手势"""
        features = self.feature_extractor.extract_features(landmarks)
        pred_idx = self.classifier.predict([features])[0]
        confidence = np.max(self.classifier.predict_proba([features]))
        
        return {
            'gesture': self.labels[pred_idx],
            'confidence': confidence,
            'all_probabilities': dict(zip(
                self.labels, 
                self.classifier.predict_proba([features])[0]
            ))
        }
    
    def save_model(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'labels': self.labels
            }, f)