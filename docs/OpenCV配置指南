# Windows 配置 OpenCV：

## 1. 基础验证
   ```cmd
   python -c "import cv2; print('OpenCV配置成功')"
   ```
## 2. 测试摄像头（手势识别需要）
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   if cap.isOpened():
       print("✅ 摄像头可用")
       cap.release()
   else:
       print("❌ 摄像头不可用")
   ```
## 3. 创建测试脚本 test_opencv.py
   ```python
   import cv2
   import numpy as np
   
   # 测试基本功能
   print(f"OpenCV版本: {cv2.__version__}")
   
   # 创建测试图像
   img = np.zeros((100, 100, 3), dtype=np.uint8)
   cv2.rectangle(img, (20, 20), (80, 80), (0, 255, 0), 2)
   
   # 保存测试
   cv2.imwrite('test_output.jpg', img)
   print("✅ 图像处理功能正常")
   
   # 摄像头测试
   cap = cv2.VideoCapture(0)
   if cap.isOpened():
       ret, frame = cap.read()
       if ret:
           print("✅ 摄像头读取正常")
       cap.release()
   ```
## 4. 运行测试
   ```cmd
   python test_opencv.py
   ```
## 5. 虚拟环境配置（如果使用）
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install opencv-python
   ```
