"""
摄像头测试
"""
import cv2
import sys

def test_camera(index=0):
    print(f"测试摄像头 #{index}")
    
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 #{index}")
        return False
    
    print(f"✅ 摄像头 #{index} 可用")
    print(f"分辨率: {int(cap.get(3))}x{int(cap.get(4))}")
    print("按'q'退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow(f"Camera #{index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("测试完成")
    return True

if __name__ == "__main__":
    success = test_camera(0)
    sys.exit(0 if success else 1)
