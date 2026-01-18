# MediaPipe 安装步骤：

## 1. 确保 Python 已安装
   ```cmd
   python --version
   ```
## 2. 安装 MediaPipe
   ```cmd
   pip install mediapipe
   ```
## 3. 验证安装
   ```cmd
   python -c "import mediapipe; print(f'MediaPipe版本: {mediapipe.__version__}')"
   ```

额外功能（可选）：

```cmd
# GPU 支持（需要 CUDA）
pip install mediapipe-gpu

# 或指定版本
pip install mediapipe==0.10.0
```

国内镜像加速：

```cmd
pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple
```

测试手势检测：

```python
import mediapipe as mp
mp_hands = mp.solutions.hands
print("✅ MediaPipe 手势模块可用")
```

安装完成。
