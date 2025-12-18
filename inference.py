"""
运动分类模型推理脚本
用于加载训练好的模型并进行预测
"""

import numpy as np
import pandas as pd
import torch
from main import MambaClassifier

class ActivityPredictor:
    """运动类型预测器"""
    
    def __init__(self, model_path='best_model.pth'):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型检查点
        print(f"正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.label_encoder = checkpoint['label_encoder']
        self.scaler = checkpoint['scaler']
        
        # 重建模型
        self.model = MambaClassifier(
            input_dim=len(self.scaler.mean_),
            num_classes=len(self.label_encoder.classes_),
            d_model=self.config['d_model'],
            n_layers=self.config['n_layers'],
            d_state=self.config['d_state'],
            d_conv=self.config['d_conv'],
            expand=self.config['expand'],
            dropout=0.0  # 推理时不使用dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型加载成功!")
        print(f"支持的活动类型: {list(self.label_encoder.classes_)}")
    
    def predict(self, features):
        """
        预测单个或多个样本
        
        Args:
            features: numpy array, shape (n_features,) 或 (n_samples, n_features)
        
        Returns:
            predictions: 预测的活动类型
            probabilities: 各类别的概率
        """
        # 确保输入是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # 转换回numpy
        probabilities = probabilities.cpu().numpy()
        predicted_indices = probabilities.argmax(axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
        
        return predicted_labels, probabilities
    
    def predict_from_csv(self, csv_path, output_path='predictions.csv'):
        """
        从CSV文件读取数据并预测
        
        Args:
            csv_path: 输入CSV文件路径
            output_path: 输出CSV文件路径
        """
        print(f"正在读取数据: {csv_path}")
        df = pd.read_excel(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
        
        # 提取特征列(前196列)
        feature_columns = df.columns[:-3] if 'activity_type' in df.columns else df.columns
        features = df[feature_columns].values
        
        # 预测
        print("正在进行预测...")
        predictions, probabilities = self.predict(features)
        
        # 保存结果
        results_df = df.copy()
        results_df['predicted_activity'] = predictions
        
        # 添加各类别概率
        for i, class_name in enumerate(self.label_encoder.classes_):
            results_df[f'prob_{class_name}'] = probabilities[:, i]
        
        results_df.to_csv(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")
        
        # 显示统计
        print(f"\n预测统计:")
        print(results_df['predicted_activity'].value_counts())
        
        # 如果有真实标签,计算准确率
        if 'activity_type' in df.columns:
            accuracy = (results_df['activity_type'] == results_df['predicted_activity']).mean()
            print(f"\n准确率: {accuracy*100:.2f}%")
        
        return results_df

def example_usage():
    """使用示例"""
    
    # 初始化预测器
    predictor = ActivityPredictor('best_model.pth')
    
    # 方法1: 从文件预测
    print("\n" + "="*60)
    print("示例1: 从文件预测")
    print("="*60)
    results = predictor.predict_from_csv('data.csv', 'predictions.csv')
    print(f"预测了 {len(results)} 个样本")
    
    # 方法2: 单个样本预测
    print("\n" + "="*60)
    print("示例2: 单个样本预测")
    print("="*60)
    
    # 读取一个样本用于演示
    df = pd.read_excel('data.csv', nrows=1)
    sample_features = df.iloc[0, :-3].values
    
    predictions, probabilities = predictor.predict(sample_features)
    
    print(f"预测结果: {predictions[0]}")
    print(f"\n各类别概率:")
    for i, class_name in enumerate(predictor.label_encoder.classes_):
        print(f"  {class_name}: {probabilities[0, i]*100:.2f}%")

if __name__ == '__main__':
    example_usage()
