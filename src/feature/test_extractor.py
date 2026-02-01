import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

# 添加路径，确保可以导入extractor
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from feature.extractor import GestureFeatureExtractor

def test_basic_extraction():
    """测试基础特征提取功能"""
    print("=" * 60)
    print("基础特征提取测试")
    print("=" * 60)

    # 初始化特征提取器
    extractor = GestureFeatureExtractor()

    # 创建模拟的手部关键点
    print("\n1. 测试模拟数据:")
    mock_landmarks = create_mock_landmarks()

    try:
        features = extractor.extract_features(mock_landmarks)
        print(f"✓ 模拟数据特征提取成功")
        print(f"  特征维度: {features.shape}")
        print(f"  特征类型: {features.dtype}")
        print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")

        # 显示特征名称
        feature_names = extractor.get_feature_names()
        print(f"  特征总数: {len(feature_names)}")
        print(f"  前5个特征名: {feature_names[:5]}")

    except Exception as e:
        print(f"✗ 模拟数据特征提取失败: {e}")
        import traceback
        traceback.print_exc()

    # 从摄像头测试
    print("\n2. 测试真实手部数据:")
    test_with_camera(extractor)

def create_mock_landmarks():
    """创建模拟的手部关键点"""
    mock_landmarks = []
    for i in range(21):
        class MockLandmark:
            def __init__(self, idx):
                # 创建一些有规律的数据
                self.x = 0.5 + 0.05 * np.sin(idx * 0.3)
                self.y = 0.5 + 0.05 * np.cos(idx * 0.3)
                self.z = 0.0 + 0.02 * np.sin(idx * 0.5)
        mock_landmarks.append(MockLandmark(i))
    return mock_landmarks

def save_features(extractor, landmarks, frame_id):
    """保存特征到文件"""
    try:
        features = extractor.extract_features(landmarks)
        feature_names = extractor.get_feature_names()

        # 保存为文本文件
        filename = f"features_frame_{frame_id}.txt"
        with open(filename, 'w') as f:
            f.write("特征名称,特征值\n")
            for name, value in zip(feature_names, features):
                f.write(f"{name},{value:.6f}\n")

        # 保存为numpy文件
        np.save(f"features_frame_{frame_id}.npy", features)

        print(f"✓ 特征已保存到 {filename}")

    except Exception as e:
        print(f"✗ 保存特征失败: {e}")

def test_with_camera(extractor):
    """使用摄像头测试特征提取"""
    # 初始化MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ 无法打开摄像头")
        test_with_images(extractor, hands)
        return

    print("摄像头已打开，请展示手势...")
    print("按 's' 键保存当前帧特征")
    print("按 'd' 键显示详细特征信息")
    print("按 'q' 键退出")

    frame_count = 0
    last_save_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像显示
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # 转换颜色空间
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测手部
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

                try:
                    # 提取特征
                    features = extractor.extract_features(hand_landmarks.landmark)

                    # 显示基本信息
                    cv2.putText(display_frame, f"Features: {len(features)} dim", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Frame: {frame_count}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 每隔一段时间打印特征统计
                    current_time = time.time()
                    if current_time - last_save_time > 2.0:
                        print(f"\n帧 {frame_count} - 特征统计:")
                        print(f"  维度: {features.shape}")
                        print(f"  均值: {np.mean(features):.4f}")
                        print(f"  标准差: {np.std(features):.4f}")
                        print(f"  范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
                        last_save_time = current_time

                except Exception as e:
                    cv2.putText(display_frame, f"Error: {str(e)[:30]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print(f"特征提取错误: {e}")

        else:
            cv2.putText(display_frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('test_extractor', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and results.multi_hand_landmarks:
            # 保存当前特征
            save_features(extractor, hand_landmarks.landmark, frame_count)
        elif key == ord('d') and results.multi_hand_landmarks:
            # 显示详细特征
            extractor.debug_features(hand_landmarks.landmark)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def test_with_images(extractor, hands):
    """使用测试图片测试"""
    print("使用测试图片进行测试...")

    # 创建几个简单的测试图像
    for i in range(3):
        height, width = 480, 640
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 在图像上画一些手部示意点
        center_x, center_y = width // 2, height // 2

        # 画"手掌"
        cv2.circle(image, (center_x, center_y), 50, (200, 200, 200), -1)

        # 画"手指"
        for j in range(5):
            angle = -np.pi/2 + j * np.pi/8
            finger_x = int(center_x + 100 * np.cos(angle))
            finger_y = int(center_y + 100 * np.sin(angle))
            cv2.circle(image, (finger_x, finger_y), 10, (100, 100, 255), -1)

        # 测试特征提取
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    features = extractor.extract_features(hand_landmarks.landmark)
                    print(f"图片 {i+1} 特征提取成功: {features.shape}")
                except Exception as e:
                    print(f"图片 {i+1} 特征提取失败: {e}")
        else:
            print(f"图片 {i+1} 未检测到手部")

    print("图片测试完成")

if __name__=="__main__":
    test_basic_extraction()