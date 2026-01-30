#!/usr/bin/env python3
"""
手势识别系统一键运行脚本
使用方法：
1. python run_all.py collect    # 收集数据
2. python run_all.py train      # 训练模型  
3. python run_all.py predict    # 实时识别
"""

import sys
import subprocess
import os

def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python run_all.py collect   # 收集手势数据")
        print("  python run_all.py train     # 训练模型")
        print("  python run_all.py predict   # 实时识别")
        return
    
    command = sys.argv[1]
    
    if command == "collect":
        print("📸 开始收集手势数据...")
        print("请依次做出0-10的手势，按空格键保存每个样本")
        subprocess.run([sys.executable, "data_collector.py"])
        
    elif command == "train":
        print("🧠 开始训练模型...")
        if not os.path.exists("gesture_data/features.npy"):
            print("❌ 没有找到数据，请先运行: python run_all.py collect")
            return
        subprocess.run([sys.executable, "train_model.py"])
        
    elif command == "predict":
        print("🎮 开始实时手势识别...")
        if not os.path.exists("model.pkl"):
            print("❌ 没有找到模型，请先运行: python run_all.py train")
            return
        subprocess.run([sys.executable, "gesture_predictor.py"])
        
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main()