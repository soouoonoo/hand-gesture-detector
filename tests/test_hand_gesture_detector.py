"""
手势识别系统测试
"""
import unittest
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.hand_detector import HandDetector
    HAS_DEPS = True
except:
    HAS_DEPS = False

class TestHandDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_DEPS:
            cls.detector = HandDetector()
        cls.test_img = np.ones((480,640,3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        if not HAS_DEPS:
            self.skipTest("缺少依赖")
        self.assertIsNotNone(self.detector.hands)
    
    def test_find_hands(self):
        if not HAS_DEPS:
            self.skipTest("缺少依赖")
        result = self.detector.find_hands(self.test_img, draw=False)
        self.assertEqual(result.shape, (480,640,3))

class TestCamera(unittest.TestCase):
    def test_camera(self):
        cap = cv2.VideoCapture(0)
        available = cap.isOpened()
        cap.release()
        if not available:
            self.skipTest("摄像头不可用")

def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in [TestHandDetector, TestCamera]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTest(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n测试结果: {result.testsRun}个测试")
    if result.wasSuccessful():
        print("✅ 所有测试通过")
    else:
        print("❌ 有测试失败")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
