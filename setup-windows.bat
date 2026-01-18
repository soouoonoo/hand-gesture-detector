@echo off
chcp 65001 >nul
echo ========================================
echo     手势识别项目 - Windows一键安装
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python
    pause
    exit /b 1
)

echo [1/7] 升级pip...
python -m pip install --upgrade pip

echo [2/7] 创建虚拟环境...
if exist venv (
    echo 虚拟环境已存在
) else (
    python -m venv venv
)

echo [3/7] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [4/7] 安装依赖包...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [5/7] 测试环境...
python -c "import cv2, mediapipe, numpy; print('✓ 所有包安装成功')"

echo [6/7] 运行测试...
python tests\test_basic.py

echo [7/7] 完成！
echo.
echo 常用命令:
echo   venv\Scripts\activate.bat  - 激活环境
echo   python src\main.py         - 运行程序
echo.
pause
