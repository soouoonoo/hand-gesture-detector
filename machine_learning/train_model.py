import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def load_data():
    """加载收集的数据"""
    features = np.load("gesture_data/features.npy")
    labels = np.load("gesture_data/labels.npy")
    print(f"📊 加载数据: {features.shape[0]} 个样本, {features.shape[1]} 个特征")
    print(f"🔢 标签分布: {np.bincount(labels.astype(int))}")
    return features, labels

def train_simple_model():
    """训练简单的机器学习模型"""
    
    # 1. 加载数据
    X, y = load_data()
    
    # 2. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 3. 特征标准化（重要！）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 训练多个模型比较
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"  训练准确率: {train_score:.3f}")
        print(f"  测试准确率: {test_score:.3f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_model_name = name
    
    print(f"\n🏆 最佳模型: {best_model_name} (准确率: {best_score:.3f})")
    
    # 5. 保存模型和标准化器
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # 6. 评估每个类别的准确率
    print("\n📈 每个手势的准确率:")
    y_pred = best_model.predict(X_test_scaled)
    unique_labels = np.unique(y_test)
    
    label_names = ["zero", "one", "two", "three", "four", "five", 
                  "six", "seven", "eight", "nine", "ten"]
    
    for label in unique_labels:
        idx = y_test == label
        accuracy = (y_pred[idx] == label).mean()
        print(f"  {label_names[int(label)]}({int(label)}): {accuracy:.3f}")
    
    return best_model, scaler

if __name__ == "__main__":
    print("🚀 开始训练手势识别模型...")
    model, scaler = train_simple_model()
    print("✅ 模型训练完成！保存为 model.pkl 和 scaler.pkl")