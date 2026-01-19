"""
手势识别主程序 - 基础版本
"""
import cv2
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.hand_detector import HandDetector

def main():
    """主函数"""
    print("=== 手势识别系统 v0.1 ===")
    print("按 'q' 键退出程序")
    print()
    
    # 初始化手部检测器
    detector = HandDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        print("请检查：")
        print("1. 摄像头是否正确连接")
        print("2. 是否有其他程序占用摄像头")
        return
    
    print("摄像头已打开，开始检测...")
    
    frame_count = 0
    
    while True:
        # 读取一帧
        success, frame = cap.read()
        
        if not success:
            print("错误：无法读取摄像头画面")
            break
        
        frame_count += 1
        
        # 检测手部
        frame = detector.detect_hands(frame)
        
        # 显示手部数量
        if detector.hand_count > 0:
            cv2.putText(frame, f"检测到手部: {detector.hand_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示帧率（每30帧计算一次）
        if frame_count % 30 == 0:
            detector.update_fps()
        
        cv2.putText(frame, f"FPS: {detector.fps:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 显示帮助信息
        cv2.putText(frame, "按 'q' 退出", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow("手势识别", frame)
        
        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户退出")
            break
        elif key == ord('s'):
            # 保存当前帧
            cv2.imwrite(f"frame_{frame_count}.jpg", frame)
            print(f"已保存帧: frame_{frame_count}.jpg")
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("程序结束")

if __name__ == "__main__":
    main()
