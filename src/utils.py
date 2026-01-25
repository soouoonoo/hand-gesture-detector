"""
手部检测核心功能包 - 基于MediaPipe
"""
import cv2
import mediapipe as mp
import time
import numpy as np


class HandDetector:
    '''手部位置获取器'''
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        初始化手部检测器
        
        参数:
            mode: 是否静态图像模式
            max_hands: 最大检测手数
            detection_con: 检测置信度阈值
            track_con: 跟踪置信度阈值
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        # 初始化MediaPipe手部模块
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        
        # 绘制工具
        self.mp_draw = mp.solutions.drawing_utils
        
        # 性能统计
        self.fps = 0 #帧率
        self.frame_count = 0 #总帧数
        self.start_time = time.time() 
        self.hand_count = 0
        
        print(f"手部检测器初始化完成 (max_hands={max_hands})")
    
    def detect_hands(self, img, draw=True):
        """
        检测图像中的手部
        
        参数:
            img: 输入图像 (BGR格式)
            draw: 是否绘制关键点和连接线
        
        返回:
            处理后的图像
        """
        # 转换为RGB格式（MediaPipe需要）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 处理图像，检测手部
        self.results = self.hands.process(img_rgb)
        
        # 重置手部计数
        self.hand_count = 0
        
        # 如果检测到手部
        if self.results.multi_hand_landmarks:
            self.hand_count = len(self.results.multi_hand_landmarks)
            
            if draw:
                # 为每只手绘制关键点和连接线
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
        
        return img
    
    def get_landmark_positions(self, img, hand_no=0):
        """
        获取手部关键点的像素坐标
        
        参数:
            img: 图像
            hand_no: 手部索引（如果检测到多只手）
        
        返回:
            关键点列表 [[id, x, y], ...]
        """
        landmarks_list = []
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            
            for id, landmark in enumerate(hand.landmark):
                # 获取图像尺寸
                h, w, c = img.shape
                
                # 将归一化坐标转换为像素坐标
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                landmarks_list.append([id, cx, cy])
        
        return landmarks_list
    
    def get_finger_tip_positions(self, img):
        """
        获取指尖位置（简化版）
        指尖对应的索引: 4(拇指), 8(食指), 12(中指), 16(无名指), 20(小指)
        """
        fingertips = []
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                h, w, c = img.shape
                
                # 获取五个指尖位置
                tip_ids = [4, 8, 12, 16, 20]
                for tip_id in tip_ids:
                    landmark = hand_landmarks.landmark[tip_id]
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    fingertips.append((tip_id, cx, cy))
        
        return fingertips
    
    def update_fps(self):
        current_time = time.time()
        
        # 获取上一次调用update_fps的时间
        last_time = getattr(self, '_last_fps_time', self.start_time)
        
        # 计算时间间隔：当前时间 - 上次调用时间
        elapsed = current_time - last_time  
        
        # 更新记录，供下次使用
        self._last_fps_time = current_time
        
        # 计算FPS
        if elapsed > 0:
            self.fps = 10.0 / elapsed  # 因为是每10帧调用一次
        else:
            self.fps = 0.0

class HandFeatureCalculator:
    '''手部特征计算器'''
    def __init__(self):
        pass
    
    def calculate_finger_lengths(self, landmarks_array):
        '''计算手指长度'''
        # 手部关键点索引
        finger_indices = {
            'thumb': [1, 2, 3, 4],        # 拇指
            'index': [5, 6, 7, 8],        # 食指
            'middle': [9, 10, 11, 12],    # 中指
            'ring': [13, 14, 15, 16],     # 无名指
            'pinky': [17, 18, 19, 20]     # 小指
        }

        lengths = {}

        for finger_name, indices in finger_indices.items():
            total_length = 0
            for i in range(len(indices) - 1):
                p1 = landmarks_array[indices[i]]
                p2 = landmarks_array[indices[i + 1]]
                segment_length = np.linalg.norm(p2 - p1)
                total_length += segment_length

            lengths[finger_name] = total_length
        
        return lengths

    def calculate_finger_angles(self, landmarks_array):
        '''计算手指关节角度'''
        angles = {}

        joint_angles = {
            'thumb_mcp': [1, 2, 3], 'thumb_ip': [2, 3, 4],
            'index_pip': [5, 6, 7], 'index_dip': [6, 7, 8],
            'middle_pip': [9, 10, 11], 'middle_dip': [10, 11, 12],
            'ring_pip': [13, 14, 15], 'ring_dip': [14, 15, 16],
            'pinky_pip': [17, 18, 19], 'pinky_dip': [18, 19, 20],  # 修正：thumb_pip -> pinky_pip
        }

        for joint_name, indices in joint_angles.items():
            p1 = landmarks_array[indices[0]]
            p2 = landmarks_array[indices[1]]
            p3 = landmarks_array[indices[2]]

            angle = self.calculate_angle(p1, p2, p3)
            angles[joint_name] = angle

        return angles
    
    def calculate_angle(self, p1, p2, p3):
        '''计算三个点之间的角度'''

        # 计算向量
        v1 = p1 - p2  # 从 p2 指向 p1
        v2 = p3 - p2  # 从 p2 指向 p3

        # 计算向量夹角
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        # 避免除以零
        if norm_product == 0:
            return 0.0
            
        cos_angle = dot_product / norm_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在有效范围内
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def get_hand_orientation(self, landmarks_array):
        '''判断手部朝向'''
        # 使用手腕和手掌点判断
        wrist = landmarks_array[0]
        palm_center = (landmarks_array[5] + landmarks_array[17]) / 2

        # 计算方向向量
        direction = palm_center - wrist

        # 判断主要方向
        if abs(direction[0]) > abs(direction[1]):
            return 'horizontal' if direction[0] > 0 else 'horizontal'
        else:
            return 'vertical' if direction[1] > 0 else 'vertical'

    def is_hand_open(self, landmarks_array, threshold=2):
        '''判断手是否张开'''
        # 计算指尖到手腕的平均距离
        wrist = landmarks_array[0]
        finger_tips = landmarks_array[[4, 8, 12, 16, 20]]

        distances = [np.linalg.norm(tip - wrist) for tip in finger_tips]
        avg_distance = np.mean(distances)

        # 计算手掌宽度作为参考
        palm_width = np.linalg.norm(landmarks_array[5] - landmarks_array[17])

        ratio = avg_distance / palm_width

        return ratio > threshold

def cornerText(img, text, y_offset=0):
    '''在左上角显示文本'''
    cv2.putText(img, text, (10, 25 + y_offset*30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def create_landmarks_array(hand_landmarks, image_shape):
    '''将 MediaPipe Landmark 对象转换为 numpy 数组'''
    h, w = image_shape[:2]
    
    landmarks_array = []
    for landmark in hand_landmarks.landmark:
        # 转换为像素坐标
        x_px = landmark.x * w
        y_px = landmark.y * h
        z_px = landmark.z  # z坐标通常保持原值
        landmarks_array.append([x_px, y_px, z_px])
    
    return np.array(landmarks_array)

def test_detector():
    """HandDetecor测试函数"""
    print("测试手部检测器...")
    
    detector = HandDetector()
    
    # 测试摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 退出测试")
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.detect_hands(img)
        
        # 显示手部数量
        cv2.putText(img, f"Hands: {detector.hand_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Hand Detector Test", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap
