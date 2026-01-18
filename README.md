# 手势识别项目

基于OpenCV、MediaPipe和机器学习的手势识别系统。

## 项目简介
本项目是一个实时手势识别系统，能够识别多种静态和动态手势，并应用于实际场景。

## 功能特性
- 实时手部关键点检测（MediaPipe）
- 手势数据采集与标注
- 手势特征提取
- 实时手势识别
- 图形用户界面（PyQt/Tkinter）
- 应用场景演示

## 快速开始

### Windows用户
```bash
# 克隆项目
git clone https://github.com/soouoonoo/hand-gesture-detector.git
cd hand-gesture-detector

# 一键安装
.\setup-windows.bat
```

## 快速验证

### 测试摄像头
python tests/test_camera.py

### 运行手势检测
python src/main.py

## 项目结构
hand-gesture-detector/
├── src/                    # 源代码
│   ├── data/              # 数据处理
│   ├── model/             # 机器学习模型
│   ├── utils/             # 工具函数
│   └── gui/               # 用户界面
├── docs/                  # 项目文档
├── tests/                 # 测试代码
├── notebooks/             # Jupyter实验
├── requirements.txt       # Python依赖
└── README.md             # 项目说明
