Write-Host "========================================" -ForegroundColor Cyan
Write-Host "     手势识别项目安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    $ver = python --version 2>&1
    Write-Host "[1/7] Python版本: $ver" -ForegroundColor Green
} catch {
    Write-Host "[错误] 未找到Python" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[2/7] 升级pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "[3/7] 创建虚拟环境..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "虚拟环境已存在" -ForegroundColor Green
} else {
    python -m venv venv
}

Write-Host "[4/7] 激活虚拟环境..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

Write-Host "[5/7] 安装依赖包..." -ForegroundColor Yellow
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "[6/7] 测试环境..." -ForegroundColor Yellow
python -c "import cv2, mediapipe, numpy; print('✓ 环境配置成功')"

Write-Host "[7/7] 完成！" -ForegroundColor Green
Write-Host ""
Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
