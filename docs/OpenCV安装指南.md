# Windows 安装 OpenCV for Python：

## 1. 安装 Python
   · 从 python.org 下载 Python 3.9+
   · 安装时必须勾选 "Add Python to PATH"
## 2. 打开 CMD 或 PowerShell
   ```cmd
   Win + R → 输入 cmd → 回车
   ```
## 3. 升级 pip
   ```cmd
   python -m pip install --upgrade pip
   ```
## 4. 安装 OpenCV
   ```cmd
   pip install opencv-python
   ```
## 5. 验证安装
   ```cmd
   python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
   ```
   显示版本号即成功。

可选：安装完整版（包含额外模块）

```cmd
pip install opencv-contrib-python
```

注意：如果网络慢，使用国内镜像：

```cmd
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

