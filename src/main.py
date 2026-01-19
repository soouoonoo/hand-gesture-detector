"""
手势识别系统主程序
"""
import cv2
if __name__=="__main__":#解决导入路径问题
    import sys
    import os
    #获取项目根目录
    project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,project_root)#加到开头
    from src.utils.hand_detector import HandDetector
else:#如果作为模块导入，使用相对导入
    from .utils.hand_detector import HandDetector

def main():
    print("手势识别系统启动...")
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("按'q'退出")
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.find_hands(img)
        cv2.imshow("Gesture Recognition", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("系统关闭")

if __name__ == "__main__":
    main()
