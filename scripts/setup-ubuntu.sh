echo "Ubuntu环境配置脚本"

sudo apt update
sudo apt install -y python3-venv python3-pip git

echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

echo "安装依赖..."
pip install -r requirements.txt

echo "✅ 环境配置完成"
echo "激活环境: source venv/bin/activate"
