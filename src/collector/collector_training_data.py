"""
手势训练数据收集脚本 - Part 1/6 (修复版)
专门用于收集5种游戏手势的训练数据
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
import time
from datetime import datetime
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feature.extractor import GestureFeatureExtractor
from classifier.classifier import GestureClassifier


class GestureDataCollector:
    """
    手势数据收集器 - 专门收集5种游戏手势数据
    """
    
    def __init__(self, output_dir="training_data"):
        """
        初始化数据收集器
        
        参数：
        output_dir: 数据保存目录
        """
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化特征提取器
        self.feature_extractor = GestureFeatureExtractor()
        
        # 数据存储
        self.output_dir = output_dir
        self.dataset = []  # 存储所有收集的数据 [(landmarks_array, label), ...]
        self.current_gesture = None
        self.current_samples = []
        
        # 收集参数
        self.collection_speed = 0.1  # 收集间隔（秒）
        self.last_collection_time = 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义5种游戏手势
        self.gestures = [
            ("point_one", "pose 'one'"),
            ("palm_up", "put up your closed palm"),
            ("victory_eight", "pose 'eight'"),
            ("fist", "fasten your fist"),
            ("hand_open", "open your hand")
        ]
        
        # 收集状态
        self.is_collecting = False
        self.collection_count = 0
        self.total_collected = 0
        
        print("=" * 60)
        print("手势数据收集器初始化完成")
        print("=" * 60)
        print("将收集以下5种游戏手势：")
        for i, (gesture_id, desc) in enumerate(self.gestures):
            print(f"  {i+1}. {gesture_id}: {desc}")
        print("=" * 60)

    def start_collection(self, gesture_id, gesture_name, num_samples=150):  # 改为150
        """
        开始收集特定手势的数据
        参数：
        gesture_id: 手势ID
        gesture_name: 手势名称
        num_samples: 需要收集的样本数量
        """
        self.current_gesture = gesture_id
        self.current_samples = []
        self.collection_count = 0
        self.target_samples = num_samples
        self.last_collection_time = time.time()

        print(f"\n{'='*60}")
        print(f"开始收集手势: {gesture_name} ({gesture_id})")
        print(f"目标样本数: {num_samples}")
        print(f"收集速度: {1/self.collection_speed:.1f} 样本/秒")
        print(f"{'='*60}")
        print("提示:")
        print("  1. 请保持手势稳定")
        print("  2. 可以稍微改变手的角度和位置")
        print("  3. 按 's' 键开始/停止收集")
        print("  4. 按 'c' 键取消当前收集")
        print("  5. 按 'q' 键退出程序")
        print(f"{'='*60}")

        self.is_collecting = True

    def _convert_landmarks_to_array(self, landmarks):
        """
        将MediaPipe landmarks转换为可序列化的numpy数组
        参数：
        landmarks: MediaPipe landmarks对象

        返回：
        numpy.ndarray: 形状为(21, 3)的数组
        """
        if landmarks is None:
            return None

        landmarks_array = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            landmarks_array[i] = [lm.x, lm.y, lm.z]
        return landmarks_array

    def stop_collection(self):
        """停止当前手势的数据收集"""
        if self.current_gesture and len(self.current_samples) > 0:
            # 保存当前收集的数据
            self._save_current_batch()
            print(f"✓ 已保存 {len(self.current_samples)} 个样本")
        # 添加到总数据集
        for landmarks_array in self.current_samples:
            self.dataset.append((landmarks_array, self.current_gesture))

        self.total_collected += len(self.current_samples)

        self.current_gesture = None
        self.current_samples = []
        self.is_collecting = False

    def add_sample(self, landmarks):
        """
        添加一个手势样本（控制收集速度）
        参数：
        landmarks: MediaPipe检测到的关键点
        """
        if not self.is_collecting or not self.current_gesture:
            return

        # 控制收集速度
        current_time = time.time()
        if current_time - self.last_collection_time < self.collection_speed:
            return

        # 验证landmarks
        if not self._validate_landmarks(landmarks):
            return

        # 转换为可序列化的数组
        landmarks_array = self._convert_landmarks_to_array(landmarks)
        if landmarks_array is None:
            return

        # 添加样本
        self.current_samples.append(landmarks_array)
        self.collection_count += 1
        self.last_collection_time = current_time

        # 显示进度
        if self.collection_count % 10 == 0:
            print(f"  已收集: {self.collection_count}/{self.target_samples}")

        # 达到目标数量
        if self.collection_count >= self.target_samples:
            print(f"✓ 已完成 {self.current_gesture} 的数据收集")
            self.stop_collection()


    def _validate_landmarks(self, landmarks):
        """
        验证landmarks数据的有效性
        参数：
        landmarks: MediaPipe关键点

        返回：
        bool: 数据是否有效
        """
        if landmarks is None or len(landmarks) != 21:
            return False

        try:
            # 检查关键点坐标是否在合理范围内
            valid_count = 0
            for lm in landmarks:
                if hasattr(lm, 'x') and hasattr(lm, 'y'):
                    if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                        valid_count += 1

            # 至少要有一定比例的关键点是有效的
            return valid_count >= 15  # 至少15/21个点是有效的
        except:
            return False
            
    def _save_current_batch(self):
        """保存当前批次的数据"""
        if not self.current_gesture or len(self.current_samples) == 0:
            return
        
        # 创建手势专属目录
        gesture_dir = os.path.join(self.output_dir, self.current_gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_gesture}_{timestamp}_{len(self.current_samples)}samples.pkl"
        filepath = os.path.join(gesture_dir, filename)
        
        # 保存数据 - 只保存可序列化的数据
        data_to_save = {
            'gesture_id': self.current_gesture,
            'samples': self.current_samples,  # 已经是numpy数组
            'count': len(self.current_samples),
            'timestamp': timestamp,
            'feature_extractor': 'GestureFeatureExtractor_v1',
            'landmark_format': 'numpy_array_21x3'
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ 数据已保存到: {filepath}")
        except Exception as e:
            print(f"✗ 保存数据失败: {e}")
            return
        
        # 同时保存为JSON格式用于检查
        json_filename = f"{self.current_gesture}_{timestamp}_metadata.json"
        json_path = os.path.join(gesture_dir, json_filename)
        
        metadata = {
            'gesture_id': self.current_gesture,
            'sample_count': len(self.current_samples),
            'timestamp': timestamp,
            'feature_dimension': 25,
            'landmarks_count': 21,
            'data_format': 'numpy_array_21x3',
            'description': next((desc for gid, desc in self.gestures if gid == self.current_gesture), "")
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    def save_complete_dataset(self, filename="complete_dataset.pkl"):
        """
        保存完整的数据集
        
        参数：
        filename: 保存文件名
        """
        if len(self.dataset) == 0:
            print("警告: 没有数据可保存")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备数据 - 只保存可序列化的数据
        dataset_info = {
            'total_samples': len(self.dataset),
            'gestures': [gesture[0] for gesture in self.gestures],
            'samples_per_gesture': {},
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_extractor': 'GestureFeatureExtractor_v1',
            'landmark_format': 'numpy_array_21x3',
            'dataset': self.dataset  # 已经是可序列化的格式
        }
        
        # 统计每个手势的样本数
        for gesture_id, _ in self.gestures:
            count = sum(1 for _, label in self.dataset if label == gesture_id)
            dataset_info['samples_per_gesture'][gesture_id] = count
        
        # 保存数据
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(dataset_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"保存数据集失败: {e}")
            return
        
        print(f"\n{'='*60}")
        print("数据集保存完成")
        print(f"{'='*60}")
        print(f"文件路径: {filepath}")
        print(f"总样本数: {len(self.dataset)}")
        for gesture_id, count in dataset_info['samples_per_gesture'].items():
            gesture_name = next((name for gid, name in self.gestures if gid == gesture_id), gesture_id)
            print(f"  {gesture_name}: {count} 个样本")
        print(f"{'='*60}")
        
        return filepath

    def load_dataset(self, filepath):
        """
        加载已保存的数据集
        
        参数：
        filepath: 数据集文件路径
        """
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在 {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                dataset_info = pickle.load(f)
            
            # 检查数据格式
            if 'dataset' not in dataset_info:
                print("错误: 数据集格式不正确")
                return False
            
            self.dataset = dataset_info['dataset']
            self.total_collected = len(self.dataset)
            
            print(f"\n{'='*60}")
            print("数据集加载成功")
            print(f"{'='*60}")
            print(f"总样本数: {len(self.dataset)}")
            
            if 'samples_per_gesture' in dataset_info:
                for gesture_id, count in dataset_info['samples_per_gesture'].items():
                    gesture_name = next((name for gid, name in self.gestures if gid == gesture_id), gesture_id)
                    print(f"  {gesture_name}: {count} 个样本")
            
            print(f"数据格式: {dataset_info.get('landmark_format', 'unknown')}")
            print(f"{'='*60}")
            return True
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return False

    def get_dataset_statistics(self):
        """
        获取数据集的统计信息
        
        返回：
        dict: 统计信息
        """
        if len(self.dataset) == 0:
            return {"error": "数据集为空"}
        
        stats = {
            'total_samples': len(self.dataset),
            'gestures': {},
            'feature_dimension': 25,
            'collected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'recommended_minimum': 150  # 建议每个手势最少150个样本
        }
        
        # 统计每个手势
        for gesture_id, gesture_name in self.gestures:
            count = sum(1 for _, label in self.dataset if label == gesture_id)
            stats['gestures'][gesture_id] = {
                'name': gesture_name,
                'count': count,
                'percentage': f"{(count / len(self.dataset) * 100):.1f}%",
                'recommended': '✓' if count >= 150 else f'需要{150-count}更多'
            }
        
        # 显示统计信息
        print(f"\n{'='*60}")
        print("数据集统计信息")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"建议每个手势最少150个样本")
        
        for gesture_id, info in stats['gestures'].items():
            status = "✓ 足够" if info['count'] >= 150 else f"⚠ 不足 ({info['recommended']})"
            print(f"  {info['name']}: {info['count']} 样本 - {status}")
        
        print(f"{'='*60}")
        
        return stats

    def split_dataset(self, train_ratio=0.8):
        """
        分割数据集为训练集和测试集（修复版）
        """
        if len(self.dataset) == 0:
            return [], []
        
        # 按手势分组
        gesture_groups = {}
        for landmarks_array, label in self.dataset:
            # 将numpy数组转换为landmarks对象
            landmarks = self._convert_array_to_landmarks(landmarks_array)
            if landmarks is None:
                continue
                
            if label not in gesture_groups:
                gesture_groups[label] = []
            gesture_groups[label].append((landmarks, label))
        
        # 分割每个手势的数据
        train_data = []
        test_data = []
        
        for gesture_id, samples in gesture_groups.items():
            np.random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            
            train_data.extend(samples[:split_idx])
            test_data.extend(samples[split_idx:])
        
        # 打乱顺序
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        
        print(f"\n数据集分割完成:")
        print(f"  训练集: {len(train_data)} 个样本")
        print(f"  测试集: {len(test_data)} 个样本")
        print(f"  分割比例: {train_ratio}:{1-train_ratio}")
        
        return train_data, test_data

    def collect_data_interactive(self, camera_id=0):
        """
        交互式数据收集
        """
        print("\n" + "="*60)
        print("交互式手势数据收集模式")
        print("="*60)
        print("操作指南:")
        print("  1. 按数字键 1-5 选择手势")
        print("  2. 按 's' 键开始/停止收集")
        print("  3. 按 'c' 键取消当前收集")
        print("  4. 按 'd' 键显示数据集统计")
        print("  5. 按 't' 键训练模型（SVM分类器）")
        print("  6. 按 '+' 键增加收集速度")
        print("  7. 按 '-' 键降低收集速度")
        print("  8. 按 'q' 键退出")
        print("="*60)
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        collecting_gesture = None
        collecting_name = ""
        frame_count = 0
        
        while True:
            success, image = cap.read()
            if not success:
                print("错误: 无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 水平翻转图像以获得镜像视图
            image = cv2.flip(image, 1)
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # 检测手部
            results = self.hands.process(image_rgb)
            
            # 转换回BGR
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 绘制手部关键点
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # 如果正在收集数据，则添加样本
                    if self.is_collecting and collecting_gesture:
                        self.add_sample(hand_landmarks.landmark)
            
            # 在图像上绘制信息面板（英文显示）
            self._draw_info_panel(image, collecting_gesture, collecting_name, hand_detected)
            
            # 显示图像
            cv2.imshow('Gesture Data Collection', image)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出
                break
                
            elif ord('1') <= key <= ord('5'):  # 选择手势
                gesture_idx = key - ord('1')
                if 0 <= gesture_idx < len(self.gestures):
                    collecting_gesture, collecting_name = self.gestures[gesture_idx]
                    print(f"\n已选择手势: {collecting_name} ({collecting_gesture})")
                    print("按 's' 键开始收集数据")
                    
            elif key == ord('s'):  # 开始/停止收集
                if collecting_gesture:
                    if not self.is_collecting:
                        # 开始收集150个样本
                        self.start_collection(collecting_gesture, collecting_name, 150)
                    else:
                        print(f"提前停止收集，已收集 {self.collection_count} 个样本")
                        self.stop_collection()
                else:
                    print("请先选择手势 (按数字键 1-5)")
                    
            elif key == ord('c'):  # 清除当前数据
                if self.is_collecting:
                    print("已停止并清除当前收集的数据")
                    self.stop_collection()
                    
            elif key == ord('d'):  # 显示统计
                self.get_dataset_statistics()
                
            elif key == ord('t'):  # 训练模型
                if len(self.dataset) > 0:
                    print("\n将使用SVM分类器进行训练...")
                    self._train_model()
                else:
                    print("错误: 没有数据可以训练")
            
            elif key == ord('+'):  # 增加收集速度
                if self.collection_speed > 0.02:  # 最快0.02秒/样本
                    self.collection_speed = max(0.02, self.collection_speed * 0.8)
                    print(f"收集速度增加: {1/self.collection_speed:.1f} 样本/秒")
                else:
                    print("已达到最大收集速度")
            
            elif key == ord('-'):  # 降低收集速度
                if self.collection_speed < 0.5:  # 最慢0.5秒/样本
                    self.collection_speed = min(0.5, self.collection_speed * 1.2)
                    print(f"收集速度降低: {1/self.collection_speed:.1f} 样本/秒")
                else:
                    print("已达到最小收集速度")
        
        cap.release()
        cv2.destroyAllWindows()
    def _draw_info_panel(self, image, current_gesture, gesture_name, hand_detected):
        """
        在图像上绘制信息面板（英文显示）
        """
        height, width = image.shape[:2]
        
        # 绘制半透明背景
        info_panel_height = 180
        info_panel = np.zeros((info_panel_height, width, 3), dtype=np.uint8)
        info_panel[:] = (50, 50, 50)
        
        # 添加文本信息（英文）
        y_offset = 30
        line_height = 25
        
        # 标题
        cv2.putText(info_panel, "Gesture Data Collector", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # 手部检测状态
        hand_status = "Hand Detected" if hand_detected else "No Hand"
        hand_color = (0, 255, 0) if hand_detected else (255, 100, 100)
        cv2.putText(info_panel, hand_status, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 1)
        y_offset += line_height
        
        # 当前手势
        if current_gesture:
            status_color = (0, 255, 0) if self.is_collecting else (255, 255, 0)
            status_text = "Collecting..." if self.is_collecting else "Selected"
            
            # 显示手势名称（英文简化版）
            gesture_display_name = gesture_name.split(' ')[0]  # 只显示第一个词
            cv2.putText(info_panel, f"Gesture: {gesture_display_name}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(info_panel, f"Status: {status_text}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            y_offset += line_height
            
            if self.is_collecting:
                progress = f"{self.collection_count}/{self.target_samples}"
                cv2.putText(info_panel, f"Progress: {progress}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                y_offset += line_height
                
                # 进度条
                bar_width = 300
                bar_height = 10
                bar_x = 10
                bar_y = y_offset
                progress_ratio = self.collection_count / self.target_samples
                
                # 背景条
                cv2.rectangle(info_panel, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), 
                            (100, 100, 100), -1)
                
                # 进度条
                progress_width = int(bar_width * progress_ratio)
                progress_color = (0, 200, 255) if progress_ratio < 1.0 else (0, 255, 0)
                cv2.rectangle(info_panel, (bar_x, bar_y), 
                            (bar_x + progress_width, bar_y + bar_height), 
                            progress_color, -1)
                
                y_offset += bar_height + 5
        
        # 收集速度显示
        speed_text = f"Speed: {1/self.collection_speed:.1f} samples/sec"
        cv2.putText(info_panel, speed_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y_offset += 20
        
        # 总样本数
        total_text = f"Total Samples: {self.total_collected}"
        cv2.putText(info_panel, total_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        y_offset += 20
        
        # 操作提示（英文）
        controls_text = "Controls: 1-5 Select, s Start/Stop, q Quit"
        cv2.putText(info_panel, controls_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
        
        # 将信息面板合并到原图像
        image[height-info_panel_height:height, 0:width] = cv2.addWeighted(
            image[height-info_panel_height:height, 0:width], 0.3, 
            info_panel, 0.7, 0
        )

    def _train_model(self):
        """训练手势分类模型（使用SVM）"""
        if len(self.dataset) < 300:  # 总样本至少300个
            print(f"错误: 数据量不足 ({len(self.dataset)}个样本)")
            print("建议至少收集300个样本（每个手势约60个）")
            print("或者确保每个手势都有足够的数据")
            
            # 检查每个手势的数据量
            stats = self.get_dataset_statistics()
            return
        
        print("\n" + "="*60)
        print("开始训练手势分类模型（SVM）")
        print("="*60)
        
        # 分割数据集
        train_data, test_data = self.split_dataset(train_ratio=0.8)
        
        # 初始化SVM分类器
        classifier = GestureClassifier(classifier_type='svm')
        
        try:
            # 训练模型
            print("正在训练SVM模型...")
            train_result = classifier.train(train_data, test_data, verbose=True)
            
            # 评估模型
            print("\n模型评估结果:")
            eval_result = classifier.evaluate(test_data, verbose=True)
            
            # 保存模型
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"gesture_model_{timestamp}.pkl")
            
            classifier.save_model(model_path, save_metadata=True)
            
            print(f"\n✓ 模型训练完成并保存到: {model_path}")
            print("="*60)
            
            return model_path
            
        except Exception as e:
            print(f"模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _convert_array_to_landmarks(self, landmarks_array):
        """
        将numpy数组转换回MediaPipe landmarks格式（修复版）
        
        参数：
        landmarks_array: numpy数组，形状为(21, 3)
        
        返回：
        list: 格式正确的MediaPipe landmarks列表
        """
        if landmarks_array is None or landmarks_array.shape != (21, 3):
            return None
        
        # 创建一个更接近MediaPipe格式的landmark类
        class MediaPipeLandmark:
            def __init__(self, idx, x, y, z):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)
                # MediaPipe landmarks还有一些其他属性
                self.visibility = 0.0  # 默认值
        
        landmarks = []
        for i in range(21):
            if i < len(landmarks_array):
                x, y, z = landmarks_array[i][:3]
                # 确保坐标在合理范围内
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    # 使用默认值
                    x, y, z = 0.5, 0.5, 0.0
                
                landmark = MediaPipeLandmark(i, x, y, z)
                landmarks.append(landmark)
            else:
                # 如果数据不够，创建默认landmark
                landmark = MediaPipeLandmark(i, 0.5, 0.5, 0.0)
                landmarks.append(landmark)
        
        return landmarks
    def load_dataset_for_training(self, filepath):
        """
        加载数据集并转换为classifier可用的格式
        
        参数：
        filepath: 数据集文件路径
        
        返回：
        list: 格式为[(landmarks_list, label_str), ...]
        """
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在 {filepath}")
            return []
        
        try:
            with open(filepath, 'rb') as f:
                dataset_info = pickle.load(f)
            
            if 'dataset' not in dataset_info:
                print("错误: 数据集格式不正确")
                return []
            
            training_data = []
            for landmarks_array, label in dataset_info['dataset']:
                # 转换回landmarks格式
                landmarks = self._convert_array_to_landmarks(landmarks_array)
                if landmarks is not None:
                    training_data.append((landmarks, label))
            
            print(f"已加载 {len(training_data)} 个有效样本")
            return training_data
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return []


def main():
    """主函数"""
    print("手势训练数据收集系统")
    print("版本: 1.0")
    print("用途: 收集5种游戏手势的训练数据")
    
    # 创建数据收集器
    collector = GestureDataCollector(output_dir="training_data")
    
    # 检查是否加载现有数据
    dataset_file = os.path.join("training_data", "complete_dataset.pkl")
    if os.path.exists(dataset_file):
        print(f"\n发现现有数据集: {dataset_file}")
        load_choice = input("是否加载现有数据? (y/n): ").lower()
        if load_choice == 'y':
            collector.load_dataset(dataset_file)
    
    # 开始交互式数据收集
    collector.collect_data_interactive(camera_id=0)
    
    # 询问是否保存数据
    if len(collector.dataset) > 0:
        save_choice = input("\n是否保存数据集? (y/n): ").lower()
        if save_choice == 'y':
            collector.save_complete_dataset()
            
        train_choice = input("是否训练模型? (y/n): ").lower()
        if train_choice == 'y':
            collector._train_model()
    
    print("\n程序结束")


if __name__ == "__main__":
    main()