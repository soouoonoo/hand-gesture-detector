Windows-Git安装指南

# 一、安装准备（按顺序）

## 第1步：安装Python(安过VS Code可忽略)

1. 浏览器打开 python.org
2. 下载 Python 3.9+ Windows安装包
3. 双击安装，必须勾选 "Add Python to PATH"
4. 点 Install Now

## 第2步：安装Git

1. 浏览器打开 git-scm.com/download/win
2. 下载 Git for Windows
3. 双击安装，全部点 Next（用默认设置）

## 第3步：验证安装

```cmd
打开CMD（按Win+R，输入cmd，回车）
输入：
python --version   （应显示Python 3.9.x）
（code --version ，如果用VS Code）
git --version      （应显示git version 2.x.x）
```

# 二、获取项目代码

## 方法A：使用Git Bash（推荐）

1. 桌面右键 → Git Bash Here
2. 输入：

```bash
git clone https://github.com/soouoonoo/hand-gesture-detector.git
cd hand-gesture-detector
```

## 方法B：使用CMD

```cmd
cd Desktop
git clone https://github.com/soouoonoo/hand-gesture-detector.git
cd hand-gesture-detector
```

# 三、环境配置

## 第1步：创建虚拟环境

```cmd
python -m venv venv
```

## 第2步：激活环境

```cmd
venv\Scripts\activate.bat
```

看到 (venv) 前缀表示成功

## 第3步：安装依赖

```cmd
pip install -r requirements.txt
```

如果慢，用：

```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 第4步：测试

```cmd
python tests\test_basic.py
python tests\test_camera.py
```

# 四、每日操作流程

## 获取更新

```cmd
cd hand-gesture-detector
venv\Scripts\activate.bat
git checkout main
git pull origin main
```

## 开发功能

```cmd
# 1. 创建分支
git checkout -b feature/你的名字-功能

# 2. 写代码（在src目录）

# 3. 提交
git add .
git commit -m "feat: 做了什么"

# 4. 推送
git push origin feature/你的名字-功能
```

## 功能完成(可选)

1. 浏览器打开GitHub项目页
2. 点 Pull requests → New pull request
3. base选 main，compare选你的分支
4. 写描述，@队友
5. 点 Create pull request

# 五、常见问题解决

## 摄像头打不开：

1. Windows设置 → 隐私 → 相机 → 开启权限
2. 关闭微信、QQ等可能占用摄像头的软件

## pip安装失败：

```cmd
python -m pip install --upgrade pip
pip install opencv-python mediapipe numpy
```

## Git克隆失败：

1. 下载ZIP：GitHub项目页 → Code → Download ZIP
2. 解压到桌面

环境激活失败：
直接运行：

```cmd
venv\Scripts\python.exe src\main.py
```
