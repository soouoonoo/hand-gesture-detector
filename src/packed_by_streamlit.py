"""
网页程序，例：在终端执行streamlit run /.../packed_by_streamlit.py
请先在test_specific_method调试，做好单一方法再从这里使用
"""
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) #添加引用路径

# 直接从utils模块导入检测方法
from multiple_hand_gestures import MultipleHandGestures

# 页面配置
st.set_page_config(
    page_title="手势检测",
    page_icon="👆",
    layout="centered"
)

st.title("👆 简单手势检测")
st.markdown("检测 **'比一'** 手势（食指伸直，其他手指弯曲）")

# 初始化MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 创建占位符
status_text = st.empty()
image_placeholder = st.empty()

# 简单的检测函数
def detect_pointing_gesture():
    """检测'比一'手势的主函数"""

    # 创建摄像头对象
    cap = cv2.VideoCapture(0)

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化手势检测器
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        detecting = True
        while detecting:
            # 读取帧
            success, frame = cap.read()
            if not success:
                status_text.warning("无法读取摄像头")
                break

            # 镜像翻转
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 检测手势
            results = hands.process(rgb_frame)

            # 检测状态
            gesture_detected = False
            score_info = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制关键点
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )

                    # 使用您的DetectNumberOne方法进行检测
                    detected, score, total = MultipleHandGestures.DetectNumberOne(hand_landmarks, (height, width), debug=False)

                    if detected:
                        gesture_detected = True
                        score_info = f"✅ 检测到手势 (分数: {score}/{total})"

                        # 绘制边界框
                        landmarks = hand_landmarks.landmark
                        x_coords = [int(lm.x * width) for lm in landmarks]
                        y_coords = [int(lm.y * height) for lm in landmarks]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        cv2.rectangle(frame, 
                                     (x_min-20, y_min-20), 
                                     (x_max+20, y_max+20), 
                                     (0, 255, 0), 3)

                        # 显示检测结果
                        cv2.putText(frame, "POINTING DETECTED", 
                                   (x_min, y_min-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # 显示分数
                        cv2.putText(frame, f"Score: {score}/{total}", 
                                   (x_min, y_min-60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        status_text.success(score_info)
                    else:
                        score_info = f"👋 未检测到 (分数: {score}/{total})"
                        status_text.info(score_info)

            # 显示状态
            if not results.multi_hand_landmarks:
                status_text.info("🖐️ 未检测到手部")

            # 转换为RGB显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 显示图像
            image_placeholder.image(frame_rgb, channels="RGB")

            # 检查是否需要停止
            detecting = True  # 这里可以通过外部变量控制

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 主界面
if st.button("🎥 开始检测", type="primary"):
    # 开始检测
    detect_pointing_gesture()

st.markdown("---")
st.markdown("""
### 📝 使用说明
1. 点击 **开始检测** 按钮
2. 面对摄像头
3. 做出 **'比一'** 手势：
   - 食指完全伸直
   - 其他手指弯曲
4. 检测到手势会有绿色边框提示
5. 左上角显示检测分数
""")