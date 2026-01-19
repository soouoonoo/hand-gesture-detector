"""
手部检测器 - 基于MediaPipe
"""
import cv2
import mediapipe as mp
import time

class HandDetector:
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
        self.fps = 0
        self.frame_count = 0
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
        """更新FPS计算"""
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0

def test_detector():
    """测试函数"""
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
