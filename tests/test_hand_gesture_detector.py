"""
摄像头测试脚本
"""
import cv2
import sys
import os

# 添加src到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_camera_basic():
    """基础摄像头测试"""
    print("=== 摄像头基础测试 ===")
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 测试失败：无法打开摄像头")
        return False
    
    print("✅ 摄像头已打开")
    
    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.1f} FPS")
    
    # 测试读取几帧
    print("  测试读取画面...")
    frames_read = 0
    
    for i in range(30):  # 尝试读取30帧
        ret, frame = cap.read()
        
        if ret:
            frames_read += 1
            
            # 每10帧显示一次进度
            if i % 10 == 0:
                print(f"    已读取 {i+1}/30 帧")
        else:
            print(f"    ❌ 第 {i+1} 帧读取失败")
            break
    
    # 释放摄像头
    cap.release()
    
    if frames_read > 0:
        print(f"✅ 测试通过：成功读取 {frames_read} 帧")
        return True
    else:
        print("❌ 测试失败：无法读取任何画面")
        return False

def test_opencv_installation():
    """测试OpenCV安装"""
    print("\n=== OpenCV安装测试 ===")
    
    try:
        import cv2
        version = cv2.__version__
        print(f"✅ OpenCV版本: {version}")
        return True
    except ImportError as e:
        print(f"❌ OpenCV导入失败: {e}")
        return False

def test_mediapipe_installation():
    """测试MediaPipe安装"""
    print("\n=== MediaPipe安装测试 ===")
    
    try:
        import mediapipe
        print(f"✅ MediaPipe已安装")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe导入失败: {e}")
        return False

def test_all():
    """运行所有测试"""
    print("开始运行环境测试...")
    print("-" * 40)
    
    tests = [
        ("OpenCV安装", test_opencv_installation),
        ("MediaPipe安装", test_mediapipe_installation),
        ("摄像头测试", test_camera_basic),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("测试结果总结:")
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n通过 {success_count}/{len(tests)} 个测试")
    
    if success_count == len(tests):
        print("\n🎉 所有测试通过！可以开始项目开发。")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
