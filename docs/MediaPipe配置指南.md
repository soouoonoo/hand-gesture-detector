# MediaPipe 配置指南：

## 1. 基础配置检查

```python
import mediapipe as mp
print(f"MediaPipe {mp.__version__} 已就绪")

# 所有可用模块
print("可用模块:")
print("- mp.solutions.hands      # 手部检测")
print("- mp.solutions.pose       # 姿态检测")
print("- mp.solutions.face_mesh  # 面部网格")
print("- mp.solutions.face_detection # 人脸检测")
```

## 2. 手部检测配置（手势识别核心）

```python
import cv2
import mediapipe as mp

# 初始化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 配置参数
hands = mp_hands.Hands(
    static_image_mode=False,      # False:视频流, True:静态图
    max_num_hands=2,              # 最多检测手数
    model_complexity=1,           # 0:轻量, 1:完整
    min_detection_confidence=0.5, # 检测置信度阈值
    min_tracking_confidence=0.5   # 跟踪置信度阈值
)
```

## 3. 摄像头测试配置

```python
def test_mediapipe():
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # 处理图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # 绘制手部关键点
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            cv2.imshow('MediaPipe测试', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC退出
                break
    
    cap.release()
    cv2.destroyAllWindows()

# 运行测试
# test_mediapipe()
```

## 4. 完整配置脚本

创建 setup_mediapipe.py：

```python
import cv2
import mediapipe as mp

def check_config():
    print("=== MediaPipe 配置检查 ===")
    
    # 1. 版本检查
    print(f"1. MediaPipe版本: {mp.__version__}")
    
    # 2. 模块检查
    print("2. 模块检查:")
    modules = ['hands', 'pose', 'face_mesh', 'face_detection']
    for module in modules:
        try:
            getattr(mp.solutions, module)
            print(f"   ✅ {module}")
        except:
            print(f"   ❌ {module}")
    
    # 3. 摄像头检查
    print("3. 摄像头检查:")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   ✅ 摄像头可用")
        cap.release()
    else:
        print("   ❌ 摄像头不可用")
    
    # 4. 手部检测测试
    print("4. 手部检测初始化:")
    try:
        hands = mp.solutions.hands.Hands()
        print("   ✅ 手部检测器创建成功")
        hands.close()
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    print("\n✅ 配置完成！")

if __name__ == "__main__":
    check_config()
```

运行：

```cmd
python setup_mediapipe.py
```

## 5. 性能优化配置

```python
# 根据设备性能调整
config = {
    'low_performance': {
        'model_complexity': 0,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.7,
        'max_num_hands': 1
    },
    'balanced': {
        'model_complexity': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'max_num_hands': 2
    },
    'high_accuracy': {
        'model_complexity': 1,
        'min_detection_confidence': 0.3,
        'min_tracking_confidence': 0.3,
        'max_num_hands': 2
    }
}

# 使用配置
hands = mp_hands.Hands(**config['balanced'])
```

## 6. 常见问题解决

```python
# 如果报错，尝试：
# 1. 降低模型复杂度
hands = mp_hands.Hands(model_complexity=0)

# 2. 提高置信度阈值
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. 确保图像格式正确
# BGR → RGB（必须）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

## 7. 最终验证

运行这个完整测试：

```cmd
python -c "
import cv2
import mediapipe as mp

print('1. 导入检查...')
print(f'   OpenCV: {cv2.__version__}')
print(f'   MediaPipe: {mp.__version__}')

print('2. 初始化手部检测...')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

print('3. 创建测试图像...')
import numpy as np
test_img = np.zeros((480, 640, 3), dtype=np.uint8)

print('4. 处理测试...')
results = hands.process(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

print('✅ 所有测试通过！')
hands.close()
"
```

配置完成标准：

1. ✅ 无导入错误
2. ✅ 能创建检测器
3. ✅ 能处理图像
4. ✅ 摄像头能打开
