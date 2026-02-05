"""
check_model.py - 检查模型文件
"""

import pickle
import numpy as np

# 加载模型
with open("models/gesture_model_20260205_175617.pkl", "rb") as f:
    model_data = pickle.load(f)

print("="*60)
print("模型检查")
print("="*60)

# 检查所有键
print("模型包含的键:")
for key in model_data.keys():
    print(f"  {key}")

# 检查分类器
classifier = model_data.get('classifier')
print(f"\n分类器类型: {type(classifier)}")
if hasattr(classifier, 'classes_'):
    print(f"类别: {classifier.classes_}")

# 检查标签映射
print(f"\n标签映射:")
print(f"label_to_index: {model_data.get('label_to_index', {})}")
print(f"index_to_label: {model_data.get('index_to_label', {})}")
print(f"class_labels: {model_data.get('class_labels', [])}")

# 检查scaler
scaler = model_data.get('scaler')
print(f"\nScaler类型: {type(scaler)}")
print(f"Scaler已训练: {model_data.get('scaler_fitted', False)}")

# 检查模型信息
model_info = model_data.get('model_info', {})
print(f"\n模型信息:")
for key, value in model_info.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)