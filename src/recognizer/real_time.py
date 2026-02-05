"""
real_time.py - 基于real_test.py的改进版实时识别系统
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature.extractor import GestureFeatureExtractor


class RealTimeGestureRecognizer:
    """
    实时手势识别器 - 改进版
    """
    
    def __init__(self, model_path=None, camera_id=0):
        """
        初始化实时手势识别器
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
        
        # 特征提取器
        self.feature_extractor = GestureFeatureExtractor()
        
        # 加载模型
        self.classifier = None
        self.scaler = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.class_labels = []
        self.scaler_fitted = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✓ 模型加载成功: {model_path}")
        else:
            print("⚠ 未提供模型路径")
        
        # 摄像头
        self.camera_id = camera_id
        self.cap = None
        
        # 识别状态
        self.is_running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
        # 手势历史
        self.gesture_history = []
        self.history_size = 5
        
        # 显示设置
        self.show_features = False
        self.show_debug = True
        
        # 动作映射
        self.action_mapping = {
            'point_one': 'MOVE',
            'palm_up': 'JUMP',
            'victory_eight': 'ATTACK',
            'fist': 'SKILL',
            'hand_open': 'INTERACT',
            'unknown': 'NONE'
        }
        
        print("\n" + "="*60)
        print("实时手势识别系统初始化完成")
        print("="*60)

    def load_model(self, model_path):
        """加载模型"""
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.scaler_fitted = model_data.get('scaler_fitted', False)
        self.label_to_index = model_data['label_to_index']
        self.index_to_label = model_data['index_to_label']
        self.class_labels = model_data['class_labels']
        
        print(f"  类别数量: {len(self.class_labels)}")
        print(f"  手势类别: {', '.join(self.class_labels)}")

    def start(self):
        """启动实时识别"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("错误: 无法打开摄像头")
            return False
        
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("实时识别系统启动")
        print("控制按键:")
        print("  'q' - 退出程序")
        print("  's' - 显示/隐藏详细信息")
        print("  'd' - 显示/隐藏特征调试")
        print("  'r' - 重置识别历史")
        print("  'p' - 手动预测当前帧")
        print("  '1-5' - 测试特定手势")
        print("="*60)
        
        return True

    def stop(self):
        """停止实时识别"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n识别系统已停止")

    def predict(self, landmarks, confidence_threshold=0.82):  
        """
        预测手势 - 简单版：低于阈值显示为'other'
        """
        if not self.classifier:
            return {'gesture': 'other', 'confidence': 0.0}
        
        try:
            # 1. 提取特征
            features = self.feature_extractor.extract_features(landmarks)
            
            # 2. 标准化
            if self.scaler_fitted and hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = np.array([features])
            
            # 3. 预测
            prediction_idx = self.classifier.predict(features_scaled)[0]
            
            # 4. 获取概率
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities))
                
                # 所有类别概率
                all_probs = {}
                for idx, prob in enumerate(probabilities):
                    gesture_name = self.index_to_label.get(idx, f'class_{idx}')
                    all_probs[gesture_name] = float(prob)
            else:
                confidence = 1.0
                all_probs = {}
            
            # 5. 获取手势
            gesture_id = self.index_to_label.get(int(prediction_idx), 'other')
            
            # 6. 关键修改：如果置信度低于阈值，显示为'other'
            if confidence < confidence_threshold:
                gesture_id = 'other'
            
            # 7. 手势信息
            gesture_name = self._get_gesture_display_name(gesture_id)
            game_action = 'NONE' if gesture_id == 'other' else self.action_mapping.get(gesture_id, 'NONE')
            
            return {
                'gesture': gesture_id,
                'gesture_name': gesture_name,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'prediction_idx': int(prediction_idx),
                'game_action': game_action,
                'is_valid': gesture_id != 'other'
            }
            
        except Exception as e:
            print(f"预测错误: {e}")
            return {'gesture': 'other', 'confidence': 0.0, 'game_action': 'NONE', 'is_valid': False}

    def _get_gesture_display_name(self, gesture_id):
        """获取手势显示名称"""
        gesture_names = {
            'point_one': 'pose \'1\'',
            'palm_up': 'palm_up',
            'victory_eight': 'pose \'8\'',
            'fist': 'fastern fist',
            'hand_open': 'open hand',
            'other': 'illegal gesture',
            'unknown': 'no gesture'
        }
        return gesture_names.get(gesture_id, gesture_id)

    def process_frame(self, image):
        """
        处理单帧图像
        """
        # 水平翻转
        image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 检测手部
        results = self.hands.process(image_rgb)
        
        # 转换回BGR
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        recognition_result = None
        
        # 如果检测到手部
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 识别手势
                recognition_result = self.predict(hand_landmarks.landmark)
                
                # 更新手势历史
                if recognition_result['gesture'] != 'unknown':
                    self.gesture_history.append(recognition_result['gesture'])
                    if len(self.gesture_history) > self.history_size:
                        self.gesture_history.pop(0)
        
        return image, recognition_result

    def draw_result(self, image, result):
        """
        在图像上绘制识别结果
        """
        if result is None:
            return image
        
        height, width = image.shape[:2]
        y_offset = 50
        
        # 手势名称
        gesture = result['gesture']
        gesture_name = result.get('gesture_name', gesture)
        confidence = result['confidence']
        game_action = result['game_action']
        
        # 颜色设置
        if gesture == 'unknown':
            gesture_color = (255, 100, 100)  # 红色
            action_color = (200, 200, 200)   # 灰色
        else:
            gesture_color = (0, 255, 100)    # 绿色
            action_color = (0, 200, 255)     # 橙色
        
        # 手势名称
        cv2.putText(image, f"Gesture: {gesture_name}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        y_offset += 40
        
        # 置信度
        cv2.putText(image, f"Confidence: {confidence:.2%}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y_offset += 30
        
        # 游戏动作
        cv2.putText(image, f"Action: {game_action}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        y_offset += 40
        
        # 显示详细信息
        if self.show_debug and gesture != 'unknown':
            # 平滑后的手势
            if len(self.gesture_history) > 1:
                from collections import Counter
                gesture_counts = Counter(self.gesture_history)
                smoothed = gesture_counts.most_common(1)[0][0]
                cv2.putText(image, f"Smoothed: {smoothed}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                y_offset += 25
            
            # 显示所有概率
            all_probs = result.get('all_probabilities', {})
            if all_probs:
                # 显示前3个
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                for i, (gest, prob) in enumerate(sorted_probs):
                    if i == 0:
                        color = (0, 255, 0)
                    else:
                        color = (200, 200, 200)
                    
                    text = f"{gest}: {prob:.2%}"
                    cv2.putText(image, text, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20
        
        return image

    def draw_system_info(self, image):
        """绘制系统信息"""
        height, width = image.shape[:2]
        
        # 底部信息栏
        info_height = 60
        info_y = height - info_height
        
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, info_y), (width, height), (40, 40, 40), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # 信息
        y_offset = info_y + 20
        
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(image, fps_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 帧数
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(image, frame_text, (width // 2 - 50, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 时间
        current_time = datetime.now().strftime("%H:%M:%S")
        time_text = f"Time: {current_time}"
        cv2.putText(image, time_text, (width - 120, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 操作提示
        y_offset += 20
        controls_text = "q-Quit, s-Info, d-Debug, r-Reset, p-Predict"
        cv2.putText(image, controls_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
        
        return image

    def run(self):
        """运行实时识别主循环"""
        if not self.start():
            return
        
        last_fps_time = time.time()
        fps_frame_count = 0
        
        while self.is_running:
            # 读取帧
            success, frame = self.cap.read()
            if not success:
                print("错误: 无法读取摄像头帧")
                break
            
            # 处理帧
            processed_frame, result = self.process_frame(frame)
            
            # 绘制结果
            if result is not None:
                processed_frame = self.draw_result(processed_frame, result)
            
            # 更新FPS
            fps_frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                self.fps = fps_frame_count / (current_time - last_fps_time)
                fps_frame_count = 0
                last_fps_time = current_time
            
            # 绘制系统信息
            processed_frame = self.draw_system_info(processed_frame)
            
            # 显示
            cv2.imshow('Gesture Recognition - Real Time', processed_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_debug = not self.show_debug
                print(f"详细信息: {'开启' if self.show_debug else '关闭'}")
            elif key == ord('d'):
                self.show_features = not self.show_features
                print(f"特征显示: {'开启' if self.show_features else '关闭'}")
            elif key == ord('r'):
                self.gesture_history = []
                print("识别历史已重置")
            elif key == ord('p'):
                # 手动触发预测
                if result:
                    print(f"\n当前预测: {result['gesture']}")
                    print(f"置信度: {result['confidence']:.2%}")
                    print(f"游戏动作: {result['game_action']}")
            
            self.frame_count += 1
        
        self.stop()


def main():
    """主函数"""
    print("实时手势识别系统 - 改进版")
    print("版本: 2.0")
    print("用途: 实时识别5种游戏手势并进行动作映射")
    
    # 查找最新的模型
    model_dir = "models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(model_dir, x)), reverse=True)
            model_path = os.path.join(model_dir, model_files[0])
            print(f"\n使用最新模型: {model_path}")
        else:
            model_path = input("\n请输入模型文件路径: ").strip()
    else:
        model_path = input("\n请输入模型文件路径: ").strip()
    
    # 运行识别器
    try:
        recognizer = RealTimeGestureRecognizer(model_path=model_path, camera_id=0)
        recognizer.run()
    except Exception as e:
        print(f"运行识别系统时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()