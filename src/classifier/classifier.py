from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import numpy as np
import sys
import os
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
#添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#导入特征提取器
try:
    from feature.extractor import GestureFeatureExtractor
except ImportError as e:
    print(f"导入特征提取器失败: {e}")
    print("请确保项目结构正确，extractor.py在feature目录中")
    raise
class GestureClassifier:
    """
    手势分类器 - 专门识别5种游戏控制手势
    5种手势定义：
    1. 'point_one'    - 比"一"：食指指向（用于左右移动）
    2. 'palm_up'      - 五指并拢向上（用于跳跃）
    3. 'victory_eight'- 比"八"：剪刀手（用于左右攻击）
    4. 'fist'         - 握拳（用于技能）
    5. 'hand_open'    - 张开五指（用于交互物件）
    """

    # 手势定义和对应的游戏动作
    GESTURE_DEFINITIONS = {
        'point_one': {
            'name': '食指指向',
            'description': '食指伸直，其他手指弯曲',
            'game_action': '根据食指指向方向决定左右移动',
            'key_features': ['食指伸直', '其他手指弯曲', '拇指可能弯曲']
        },
        'palm_up': {
            'name': '手掌向上',
            'description': '五指并拢，手掌向上',
            'game_action': '向上跳跃',
            'key_features': ['所有手指并拢', '手掌朝上', '手指较直']
        },
        'victory_eight': {
            'name': '剪刀手',
            'description': '食指和中指伸直分开，其余手指弯曲',
            'game_action': '根据食指指向方向决定左右攻击',
            'key_features': ['食指和中指伸直', '两指分开', '其余手指弯曲']
        },
        'fist': {
            'name': '握拳',
            'description': '所有手指弯曲握成拳头',
            'game_action': '释放技能',
            'key_features': ['所有手指弯曲', '指尖靠近手掌', '拇指在外或在内']
        },
        'hand_open': {
            'name': '张开手',
            'description': '所有手指伸直张开',
            'game_action': '交互物件',
            'key_features': ['所有手指伸直', '手指分开', '手掌可能朝前']
        }
    }

    def __init__(self, classifier_type='svm', model_path=None):
        """
        初始化手势分类器

        参数：
        classifier_type: 分类器类型，可选 'svm' 或 'random_forest'
        model_path: 预训练模型路径，如果提供则加载模型
        """
        # 特征提取器
        self.feature_extractor = GestureFeatureExtractor()

        # 特征标准化器
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # 训练状态
        self.is_trained = False

        # 模型信息
        self.model_info = {
            'version': '2.0',
            'classifier_type': classifier_type,
            'feature_extractor': 'GestureFeatureExtractor_v1',
            'feature_dimension': 25,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'trained_at': None,
            'num_classes': 5,
            'gesture_types': list(self.GESTURE_DEFINITIONS.keys()),
            'game_app': '手势控制游戏'
        }

        # 初始化分类器
        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                C=1.0,
                gamma='scale',
                random_state=42
            )
            print("初始化SVM分类器 (专用于5种游戏手势)")

        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=150,  # 增加树的数量以更好区分5种手势
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            print("初始化随机森林分类器 (150棵树，专用于5种游戏手势)")

        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}。请选择 'svm' 或 'random_forest'")

        self.classifier_type = classifier_type

        # 标签编码映射
        self.label_to_index = {}  # 标签 -> 索引
        self.index_to_label = {}  # 索引 -> 标签
        self.class_labels = []    # 标签列表

        # 显示手势定义
        print(f"识别的手势类型 ({len(self.GESTURE_DEFINITIONS)}种):")
        for i, (gesture_id, info) in enumerate(self.GESTURE_DEFINITIONS.items()):
            print(f"  {i+1}. {info['name']} ({gesture_id}): {info['description']}")
            print(f"     游戏动作: {info['game_action']}")

        # 加载预训练模型（如果提供）
        if model_path:
            if os.path.exists(model_path):
                self.load_model(model_path)
                print(f"✓ 已加载预训练模型: {model_path}")
            else:
                print(f"⚠ 模型文件不存在: {model_path}，将创建新模型")
        print(f"特征提取器已初始化，预期特征维度：{self.model_info['feature_dimension']}")
    
    def train(self, training_data, validation_data=None, verbose=True):
        """
        训练手势分类器
        参数：
        training_data: 训练数据，格式 [(landmarks_list, label_str), ...]
                    标签必须是5种手势之一
        validation_data: 验证数据，格式相同（可选）
        verbose: 是否显示训练详细信息

        返回：
        dict: 训练结果信息
        """
        if verbose:
            print("\n" + "="*50)
            print("开始训练游戏手势分类器")
            print("="*50)

        # 检查训练数据
        if not training_data:
            raise ValueError("训练数据为空！请提供有效的训练数据。")

        # 验证标签是否在5种手势范围内
        valid_gestures = set(self.GESTURE_DEFINITIONS.keys())
        labels_in_data = set(label for _, label in training_data)
        invalid_labels = labels_in_data - valid_gestures

        if invalid_labels:
            raise ValueError(f"发现无效手势标签: {invalid_labels}。只允许: {valid_gestures}")

        if verbose:
            print(f"训练样本数量: {len(training_data)}")
            print(f"有效手势类别: {', '.join(valid_gestures)}")

        # 提取特征和标签
        X_train, y_train = [], []
        invalid_samples = 0

        for i, (landmarks, label) in enumerate(training_data):
            try:
                # # 验证landmarks数据
                # if not self._validate_landmarks(landmarks):
                #     if verbose and invalid_samples < 3:
                #         print(f"  样本 {i}: landmarks数据无效，已跳过")
                #     invalid_samples += 1
                #     continue

                # 提取特征（使用extractor.py的方法）
                features = self.feature_extractor.extract_features(landmarks)

                # 验证特征维度（必须与extractor一致）
                if len(features) != self.model_info['feature_dimension']:
                    if verbose and invalid_samples < 3:
                        print(f"  样本 {i}: 特征维度错误 ({len(features)} ≠ {self.model_info['feature_dimension']})")
                    invalid_samples += 1
                    continue

                X_train.append(features)

                # 处理标签编码
                if label not in self.label_to_index:
                    index = len(self.label_to_index)
                    self.label_to_index[label] = index
                    self.index_to_label[index] = label
                    self.class_labels.append(label)
                    if verbose and label in self.GESTURE_DEFINITIONS:
                        info = self.GESTURE_DEFINITIONS[label]
                        print(f"  添加类别: '{info['name']}' ({label})")

                y_train.append(self.label_to_index[label])

            except Exception as e:
                if verbose and invalid_samples < 3:
                    print(f"  样本 {i} 处理失败: {e}")
                invalid_samples += 1
                continue

        if invalid_samples > 0:
            print(f"⚠ 跳过了 {invalid_samples} 个无效样本")

        # 检查是否有有效数据
        if len(X_train) == 0:
            raise ValueError("没有有效的训练样本！请检查数据格式。")

        # 检查是否覆盖了所有5种手势
        trained_gestures = set(self.class_labels)
        missing_gestures = valid_gestures - trained_gestures
        if missing_gestures:
            print(f"警告: 训练数据缺少以下手势: {missing_gestures}")

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

        if verbose:
            print(f"\n有效训练样本: {len(X_train)}")
            print(f"实际训练类别: {len(self.class_labels)}/{len(valid_gestures)}")
            print(f"类别列表: {self.class_labels}")
            print(f"特征维度: {X_train.shape[1]}")

        # 特征标准化
        if verbose:
            print("\n正在进行特征标准化...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.scaler_fitted = True

        # 训练分类器
        if verbose:
            print(f"正在训练 {self.classifier_type.upper()} 分类器...")
            start_time = time.time()

        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True

        if verbose:
            training_time = time.time() - start_time
            print(f"✓ 训练完成，耗时: {training_time:.2f}秒")

        # 计算训练准确率
        train_predictions = self.classifier.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_predictions)

        if verbose:
            print(f"训练准确率: {train_accuracy:.4f}")

        # 更新模型信息
        self.model_info.update({
            'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_classes': len(self.class_labels),
            'trained_gestures': self.class_labels.copy(),
            'train_accuracy': float(train_accuracy),
            'training_samples': len(X_train)
        })

        # 验证集评估（如果提供）
        val_results = {}
        if validation_data:
            val_results = self.evaluate(validation_data, verbose=verbose)

        # 返回训练结果
        result = {
            'status': 'success',
            'classifier_type': self.classifier_type,
            'num_classes': len(self.class_labels),
            'class_labels': self.class_labels,
            'training_samples': len(X_train),
            'train_accuracy': train_accuracy,
            'invalid_samples': invalid_samples,
            'model_info': self.model_info.copy()
        }

        if validation_data:
            result['validation_accuracy'] = val_results.get('accuracy', 0.0)

        if verbose:
            print("\n" + "="*50)
            print("训练完成！")
            print("="*50)
            print(f"已训练类别: {', '.join(self.class_labels)}")
            print(f"训练准确率: {train_accuracy:.2%}")
            if validation_data:
                print(f"验证准确率: {result.get('validation_accuracy', 0.0):.2%}")
            print("可以使用 save_model() 方法保存模型")

        return result

    def predict(self, landmarks, confidence_threshold=0.6, return_details=False):
        """
        预测手势
        参数：
        landmarks: MediaPipe landmarks列表
        confidence_threshold: 置信度阈值，低于此值返回'unknown'
        return_details: 是否返回详细的手势分析信息

        返回：
        dict: 包含预测结果的字典
        """
        if not self.is_trained:
            return {
                'gesture': 'unknown',
                'confidence': 0.0,
                'error': 'Model not trained',
                'all_probabilities': {}
            }

        # 验证landmarks
        if not self._validate_landmarks(landmarks):
            return {
                'gesture': 'unknown',
                'confidence': 0.0,
                'error': 'Invalid landmarks',
                'all_probabilities': {}
            }

        try:
            # 提取特征
            features = self.feature_extractor.extract_features(landmarks)

            # 检查特征维度
            if len(features) != self.model_info['feature_dimension']:
                return {
                    'gesture': 'unknown',
                    'confidence': 0.0,
                    'error': f'Feature dimension mismatch: {len(features)}',
                    'all_probabilities': {}
                }

            # 标准化特征
            if self.scaler_fitted:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]

            # 预测
            prediction_idx = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]

            # 获取最高置信度
            confidence = np.max(probabilities)
            gesture_id = self.index_to_label.get(prediction_idx, 'unknown')

            # 创建所有类别的概率字典
            all_probs = {}
            for idx, prob in enumerate(probabilities):
                label = self.index_to_label.get(idx, f'class_{idx}')
                all_probs[label] = float(prob)

            # 应用置信度阈值
            if confidence < confidence_threshold:
                gesture_id = 'unknown'

            # 基础结果
            result = {
                'gesture': gesture_id,
                'confidence': float(confidence),
                'all_probabilities': all_probs,
                'feature_vector': features.tolist()
            }

            # 如果要求详细信息，添加手势描述和游戏动作
            if return_details and gesture_id in self.GESTURE_DEFINITIONS:
                gesture_info = self.GESTURE_DEFINITIONS[gesture_id]
                result.update({
                    'gesture_name': gesture_info['name'],
                    'description': gesture_info['description'],
                    'game_action': gesture_info['game_action'],
                    'key_features': gesture_info['key_features']
                })

            return result

        except Exception as e:
            return {
                'gesture': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'all_probabilities': {}
            }
    
    def save_model(self, filepath, save_metadata=True):
        """
        保存训练好的模型

        参数：
        filepath: 模型保存路径
        save_metadata: 是否保存模型元数据为单独文件
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法保存")

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # 准备保存数据
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'scaler_fitted': self.scaler_fitted,
            'label_to_index': self.label_to_index,
            'index_to_label': self.index_to_label,
            'class_labels': self.class_labels,
            'classifier_type': self.classifier_type,
            'is_trained': self.is_trained,
            'model_info': self.model_info
        }

        # 保存主模型文件
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ 模型已保存到: {filepath}")

        # 保存元数据为JSON文件（可选）
        if save_metadata:
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            metadata = {
                'model_info': self.model_info,
                'class_labels': self.class_labels,
                'classifier_type': self.classifier_type,
                'num_classes': len(self.class_labels),
                'feature_dimension': self.model_info['feature_dimension'],
                'gesture_definitions': self.GESTURE_DEFINITIONS,
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"✓ 模型元数据已保存到: {metadata_path}")

    def load_model(self, filepath):
        """
        加载预训练模型

        参数：
        filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # 加载模型数据
            self.classifier = model_data['classifier']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.scaler_fitted = model_data.get('scaler_fitted', False)
            self.label_to_index = model_data['label_to_index']
            self.index_to_label = model_data['index_to_label']
            self.class_labels = model_data['class_labels']
            self.classifier_type = model_data.get('classifier_type', 'svm')
            self.is_trained = model_data['is_trained']
            self.model_info = model_data.get('model_info', self.model_info)

            print(f"✓ 模型加载成功: {filepath}")
            print(f"  分类器类型: {self.classifier_type}")
            print(f"  类别数量: {len(self.class_labels)}")
            print(f"  手势类别: {', '.join(self.class_labels)}")

            return True

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise

    def evaluate(self, test_data, verbose=True):
        """
        评估模型性能

        参数：
        test_data: 测试数据，格式 [(landmarks, label), ...]
        verbose: 是否显示详细信息

        返回：
        dict: 评估结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法评估")

        if verbose:
            print("\n" + "="*50)
            print("模型评估")
            print("="*50)

        X_test, y_true = [], []
        invalid_samples = 0

        for landmarks, true_label in test_data:
            try:
                if not self._validate_landmarks(landmarks):
                    invalid_samples += 1
                    continue

                features = self.feature_extractor.extract_features(landmarks)

                if len(features) != self.model_info['feature_dimension']:
                    invalid_samples += 1
                    continue

                X_test.append(features)

                if true_label in self.label_to_index:
                    y_true.append(self.label_to_index[true_label])
                else:
                    y_true.append(-1)  # 未知标签

            except Exception:
                invalid_samples += 1
                continue

        if len(X_test) == 0:
            raise ValueError("没有有效的测试样本")

        X_test = np.array(X_test)
        y_true = np.array(y_true)

        # 标准化特征
        if self.scaler_fitted:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # 预测
        y_pred = self.classifier.predict(X_test_scaled)

        # 计算准确率（忽略未知标签）
        valid_indices = y_true != -1
        if np.sum(valid_indices) > 0:
            accuracy = accuracy_score(y_true[valid_indices], y_pred[valid_indices])
        else:
            accuracy = 0.0

        # 生成分类报告
        if verbose and len(self.class_labels) > 0:
            print(f"\n测试样本数: {len(X_test)}")
            print(f"无效样本数: {invalid_samples}")
            print(f"准确率: {accuracy:.4f}")

            if len(self.class_labels) > 1:
                print("\n分类报告:")
                print(classification_report(
                    y_true[valid_indices], 
                    y_pred[valid_indices],
                    target_names=self.class_labels,
                    zero_division=0
                ))

            # 显示混淆矩阵（如果类别较少）
            if len(self.class_labels) <= 5:
                print("\n混淆矩阵:")
                cm = confusion_matrix(y_true[valid_indices], y_pred[valid_indices])
                print("预测→")
                for i, true_label in enumerate(self.class_labels):
                    row = f"{true_label:15s} |"
                    for j, pred_label in enumerate(self.class_labels):
                        row += f" {cm[i,j]:3d}"
                    print(row)

        return {
            'accuracy': float(accuracy),
            'test_samples': len(X_test),
            'invalid_samples': invalid_samples,
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist()
        }
    
    def get_gesture_info(self, gesture_id=None):
        """
        获取手势信息

        参数：
        gesture_id: 手势ID，如果为None则返回所有手势信息

        返回：
        dict: 手势信息
        """
        if gesture_id is None:
            return self.GESTURE_DEFINITIONS.copy()
        elif gesture_id in self.GESTURE_DEFINITIONS:
            return self.GESTURE_DEFINITIONS[gesture_id]
        else:
            return {'error': f'未知手势: {gesture_id}'}

    def get_feature_names(self):
        """获取特征名称列表（与extractor.py一致）"""
        return self.feature_extractor.get_feature_names()

    def get_model_info(self):
        """获取模型信息"""
        return self.model_info.copy()

    def _validate_landmarks(self, landmarks):
        """
        验证landmarks数据格式 - 修复版
        
        真实MediaPipe landmarks的验证需要更宽松
        """
        if landmarks is None:
            return False
        
        if not isinstance(landmarks, list):
            return False
        
        # MediaPipe landmarks是21个点
        if len(landmarks) != 21:
            print(f"⚠ 警告: landmarks数量{len(landmarks)} != 21")
            return False
        
        try:
            # 检查前几个landmark是否有必要的属性
            # MediaPipe的landmark有x, y, z属性
            for i in range(min(3, len(landmarks))):
                lm = landmarks[i]
                
                # 检查是否有x, y属性
                if not hasattr(lm, 'x') or not hasattr(lm, 'y'):
                    print(f"⚠ 警告: landmark[{i}]缺少x或y属性")
                    return False
                
                # 获取值
                x = getattr(lm, 'x', None)
                y = getattr(lm, 'y', None)
                
                # 检查是否为数字
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    print(f"⚠ 警告: landmark[{i}]的x或y不是数字")
                    return False
                
                # MediaPipe的坐标通常在0-1之间，但允许稍微超出
                # 真实数据可能在-0.5到1.5之间
                if x < -1.0 or x > 2.0 or y < -1.0 or y > 2.0:
                    print(f"⚠ 警告: landmark[{i}]坐标超出范围: x={x}, y={y}")
                    # 不直接返回False，可能只是极端情况
            
            return True
            
        except Exception as e:
            print(f"⚠ landmarks验证异常: {e}")
            # 为了兼容性，即使有异常也返回True
            # 因为真实数据可能有各种情况
            return True  # 改为True，更宽松

    def get_class_distribution(self, data):
        """
        获取数据集中各类别的分布

        参数：
        data: 数据集，格式 [(landmarks, label), ...]

        返回：
        dict: 类别分布统计
        """
        distribution = {}
        for _, label in data:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def analyze_gesture_features(self, landmarks):
        """
        分析手势特征，用于调试和特征理解

        参数：
        landmarks: 手部关键点

        返回：
        dict: 特征分析结果
        """
        try:
            # 提取原始特征
            features = self.feature_extractor.extract_features(landmarks)
            feature_names = self.feature_extractor.get_feature_names()

            # 获取手指状态
            finger_states = {}
            for i, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
                state_idx = 20 + i  # 手指状态在特征向量中的位置
                if state_idx < len(features):
                    finger_states[finger] = '张开' if features[state_idx] > 0.5 else '闭合'

            # 获取手指角度
            finger_angles = {}
            for i, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
                angle_idx = i  # 手指角度在特征向量中的位置
                if angle_idx < len(features):
                    angle = features[angle_idx] * 180  # 转换为角度
                    finger_angles[finger] = f"{angle:.1f}°"

            # 分析可能的手势类型
            potential_gestures = []
            for gesture_id, info in self.GESTURE_DEFINITIONS.items():
                # 简单的规则判断（可以根据需要扩展）
                score = 0

                if gesture_id == 'point_one':
                    # 食指指向：食指张开，其他手指闭合
                    if finger_states.get('index') == '张开':
                        score += 2
                    if finger_states.get('middle') == '闭合':
                        score += 1

                elif gesture_id == 'victory_eight':
                    # 剪刀手：食指和中指张开
                    if finger_states.get('index') == '张开':
                        score += 1
                    if finger_states.get('middle') == '张开':
                        score += 1

                elif gesture_id == 'fist':
                    # 握拳：所有手指闭合
                    if all(state == '闭合' for state in finger_states.values()):
                        score += 3

                elif gesture_id == 'hand_open':
                    # 张开手：所有手指张开
                    if all(state == '张开' for state in finger_states.values()):
                        score += 3

                potential_gestures.append({
                    'gesture': gesture_id,
                    'name': info['name'],
                    'score': score,
                    'action': info['game_action']
                })

            # 按分数排序
            potential_gestures.sort(key=lambda x: x['score'], reverse=True)

            return {
                'finger_states': finger_states,
                'finger_angles': finger_angles,
                'potential_gestures': potential_gestures[:3],  # 只返回前3个
                'feature_count': len(features),
                'is_valid': True
            }

        except Exception as e:
            return {
                'error': str(e),
                'is_valid': False
            }

    def predict_with_analysis(self, landmarks, confidence_threshold=0.6):
        """
        预测手势并返回分析结果

        参数：
        landmarks: 手部关键点
        confidence_threshold: 置信度阈值

        返回：
        dict: 包含预测和分析结果的字典
        """
        # 获取预测结果
        prediction = self.predict(landmarks, confidence_threshold, return_details=True)

        # 获取特征分析
        analysis = self.analyze_gesture_features(landmarks)

        # 合并结果
        result = {
            'prediction': prediction,
            'analysis': analysis,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return result