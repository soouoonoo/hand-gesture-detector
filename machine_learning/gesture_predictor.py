import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

class GesturePredictor:
    def __init__(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        # 加载模型
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # 手势标签名称
        self.label_names = ["zero", "one", "two", "three", "four", "five", 
                           "six", "seven", "eight", "nine", "ten"]

        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        print("✅ 模型加载完成，准备开始识别...")

    def extract_features(self, hand_landmarks):
        """从MediaPipe关键点提取特征"""
        landmarks = hand_landmarks.landmark
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y])  # 只使用x,y坐标
        return np.array(features).reshape(1, -1)

    def predict_gesture(self, features):
        """预测手势"""
        # 标准化特征
        features_scaled = self.scaler.transform(features)

        # 预测
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        # 获取Top-3预测
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = [(self.label_names[i], probabilities[i]) for i in top3_idx]

        return int(prediction), top3

    def run_realtime(self):
        """实时手势识别"""
        cap = cv2.VideoCapture(0)

        with self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1
        ) as hands:

            fps_time = 0
            predictions_history = []

            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape

                # 计算FPS
                fps = 1.0 / (time.time() - fps_time) if fps_time > 0 else 0
                fps_time = time.time()

                # 转换为RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测手势
                results = hands.process(rgb_frame)

                # 显示FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if results.multi_hand_landmarks:
                    # 只处理第一只手
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # 绘制关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 提取特征并预测
                    features = self.extract_features(hand_landmarks)
                    prediction, top3 = self.predict_gesture(features)

                    # 添加到历史（用于平滑）
                    predictions_history.append(prediction)
                    if len(predictions_history) > 5:
                        predictions_history.pop(0)

                    # 使用众数平滑预测结果
                    smoothed_prediction = max(set(predictions_history), 
                                            key=predictions_history.count)

                    # 显示预测结果
                    cv2.putText(frame, 
                              f"Gesture: {self.label_names[smoothed_prediction]} ({smoothed_prediction})",
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 显示置信度
                    cv2.putText(frame, 
                              f"Confidence: {top3[0][1]:.2f}",
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 显示Top-3预测
                    for i, (name, prob) in enumerate(top3[:3]):
                        y_pos = 150 + i * 30
                        color = (0, 200, 0) if i == 0 else (200, 200, 0)
                        cv2.putText(frame, 
                                  f"{name}: {prob:.3f}",
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                else:
                    cv2.putText(frame, "No hand detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 显示帮助信息
                cv2.putText(frame, "Press 'q' to quit", (width-200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # 显示窗口
                cv2.imshow("Gesture Recognition 1-10", frame)

                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎮 手势1-10实时识别系统")
    print("请确保已经运行 train_model.py 训练了模型")

    predictor = GesturePredictor()
    predictor.run_realtime()