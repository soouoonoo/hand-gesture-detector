"""
基础环境测试
"""
import sys

def test_python():
    ver = sys.version_info
    if ver.major == 3 and ver.minor >= 9:
        print(f"✅ Python {ver.major}.{ver.minor}")
        return True
    else:
        print(f"❌ 需要Python 3.9+")
        return False

def test_imports():
    modules = [('cv2','OpenCV'), ('numpy','NumPy'), 
               ('mediapipe','MediaPipe'), ('sklearn','scikit-learn')]
    
    all_ok = True
    for mod, name in modules:
        try:
            __import__(mod)
            print(f"✅ {name}")
        except:
            print(f"❌ {name}")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("环境测试")
    print("-" * 20)
    
    results = []
    for name, func in [("Python版本", test_python), ("依赖包", test_imports)]:
        print(f"\n{name}:")
        results.append(func())
    
    print("\n" + "=" * 20)
    if all(results):
        print("🎉 环境测试通过")
        sys.exit(0)
    else:
        print("⚠️  环境测试失败")
        sys.exit(1)
