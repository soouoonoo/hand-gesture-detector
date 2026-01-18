"""
测试环境文件
"""

def test_environment():
    """测试环境是否正常"""
    try:
        import cv2, mediapipe, numpy
        print("✅ 所有包已安装")
        print(f"OpenCV: {cv2.__version__}")
        print(f"MediaPipe: {mediapipe.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    test_environment()
