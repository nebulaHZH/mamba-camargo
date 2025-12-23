"""
多模型性能对比脚本
对比模型：Mamba, Transformer, XGBoost, SVM, LDA, MLP
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入Mamba模型组件
from main import MambaClassifier, ActivityDataset, set_seed, load_and_preprocess_data

set_seed(42)

# ==================== Transformer模型 ====================
class TransformerClassifier(nn.Module):
    """
    基于Transformer的分类器
    使用多头自注意力机制进行序列建模
    """
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # 输入嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        """
        x: (batch, input_dim) 或 (batch, seq_len, input_dim)
        """
        # 如果输入是2D，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # 嵌入
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==================== 传统机器学习模型训练 ====================
def train_sklearn_model(model, X_train, y_train, X_val, y_val, model_name):
    """训练sklearn类型的模型"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*60}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 训练集评估
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # 验证集评估
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"训练时间: {train_time:.2f}秒")
    print(f"训练准确率: {train_acc*100:.2f}%")
    print(f"验证准确率: {val_acc*100:.2f}%")
    
    return model, train_time

def evaluate_model(model, X_test, y_test, model_name, label_encoder, is_torch=False, device=None):
    """评估模型性能"""
    print(f"\n{'='*60}")
    print(f"评估 {model_name} 模型...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if is_torch:
        # PyTorch模型
        model.eval()
        test_dataset = ActivityDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
    else:
        # sklearn模型
        y_pred = model.predict(X_test)
        y_true = y_test
    
    inference_time = time.time() - start_time
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} 测试集结果:")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"推理时间:  {inference_time:.4f}秒")
    
    print(f"\n详细分类报告:")
    print(classification_report(y_true, y_pred, 
                              target_names=label_encoder.classes_,
                              digits=4))
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time': inference_time,
        'predictions': y_pred,
        'true_labels': y_true
    }
    
    return results

# ==================== 训练Mamba模型 ====================
def train_mamba_model(X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    """训练Mamba模型"""
    print(f"\n{'='*60}")
    print(f"训练 Mamba 模型...")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_dataset = ActivityDataset(X_train, y_train)
    val_dataset = ActivityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 创建模型
    model = MambaClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 训练循环
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    num_epochs = 50
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - 训练准确率: {train_acc:.2f}%, 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停触发! {patience} 个epoch验证准确率未提升")
            break
    
    train_time = time.time() - start_time
    
    print(f"\n训练时间: {train_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return model, train_time

# ==================== 训练Transformer模型 ====================
def train_transformer_model(X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    """训练Transformer模型"""
    print(f"\n{'='*60}")
    print(f"训练 Transformer 模型...")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_dataset = ActivityDataset(X_train, y_train)
    val_dataset = ActivityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 创建模型
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 训练循环
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    num_epochs = 50
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - 训练准确率: {train_acc:.2f}%, 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停触发! {patience} 个epoch验证准确率未提升")
            break
    
    train_time = time.time() - start_time
    
    print(f"\n训练时间: {train_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return model, train_time

# ==================== 可视化对比结果 ====================
def plot_model_comparison(results_list, save_path='model_comparison.png'):
    """绘制模型对比图"""
    models = [r['model'] for r in results_list]
    accuracies = [r['accuracy'] * 100 for r in results_list]
    precisions = [r['precision'] * 100 for r in results_list]
    recalls = [r['recall'] * 100 for r in results_list]
    f1_scores = [r['f1_score'] * 100 for r in results_list]
    
    # 6个模型的颜色
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8338EC']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 准确率对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, accuracies, color=colors[:len(models)])
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{accuracies[i]:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    # F1分数对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, f1_scores, color=colors[:len(models)])
    ax2.set_ylabel('F1-Score (%)', fontsize=12)
    ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1_scores[i]:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Precision对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, precisions, color=colors[:len(models)])
    ax3.set_ylabel('Precision (%)', fontsize=12)
    ax3.set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{precisions[i]:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Recall对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, recalls, color=colors[:len(models)])
    ax4.set_ylabel('Recall (%)', fontsize=12)
    ax4.set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{recalls[i]:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n模型对比图已保存至: {save_path}")
    plt.close()

def plot_confusion_matrices(results_list, label_encoder, save_path='confusion_matrices.png'):
    """绘制所有模型的混淆矩阵"""
    n_models = len(results_list)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results_list):
        cm = confusion_matrix(result['true_labels'], result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_, 
                   yticklabels=label_encoder.classes_,
                   ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{result["model"]} - Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close()

def save_results_table(results_list, save_path='model_comparison_results.csv'):
    """保存对比结果到CSV"""
    df = pd.DataFrame([{
        'Model': r['model'],
        'Accuracy (%)': f"{r['accuracy']*100:.2f}",
        'Precision (%)': f"{r['precision']*100:.2f}",
        'Recall (%)': f"{r['recall']*100:.2f}",
        'F1-Score (%)': f"{r['f1_score']*100:.2f}",
        'Inference Time (s)': f"{r['inference_time']:.4f}"
    } for r in results_list])
    
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n结果表格已保存至: {save_path}")
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)

# ==================== 主程序 ====================
def main():
    print("="*80)
    print("多模型性能对比实验")
    print("对比模型: Mamba, Transformer, XGBoost, SVM, LDA, MLP")
    print("="*80)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, scaler = \
        load_and_preprocess_data('all_fused_features.csv', test_size=0.2, val_size=0.1)
    
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    # 存储所有模型的结果
    results_list = []
    
    # ==================== 1. Mamba模型 ====================
    mamba_model, mamba_train_time = train_mamba_model(
        X_train, y_train, X_val, y_val, input_dim, num_classes, device
    )
    mamba_results = evaluate_model(
        mamba_model, X_test, y_test, 'Mamba', label_encoder, 
        is_torch=True, device=device
    )
    results_list.append(mamba_results)
    
    # ==================== 2. Transformer ====================
    transformer_model, transformer_train_time = train_transformer_model(
        X_train, y_train, X_val, y_val, input_dim, num_classes, device
    )
    transformer_results = evaluate_model(
        transformer_model, X_test, y_test, 'Transformer', label_encoder,
        is_torch=True, device=device
    )
    results_list.append(transformer_results)
    
    # ==================== 3. XGBoost ====================
    print(f"\n{'='*80}")
    print("训练 XGBoost 模型...")
    print(f"{'='*80}")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_model, xgb_train_time = train_sklearn_model(
        xgb_model, X_train, y_train, X_val, y_val, 'XGBoost'
    )
    xgb_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost', label_encoder)
    results_list.append(xgb_results)
    
    # ==================== 4. SVM ====================
    print(f"\n{'='*80}")
    print("训练 SVM 模型...")
    print(f"{'='*80}")
    
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        random_state=42,
        max_iter=1000
    )
    
    svm_model, svm_train_time = train_sklearn_model(
        svm_model, X_train, y_train, X_val, y_val, 'SVM'
    )
    svm_results = evaluate_model(svm_model, X_test, y_test, 'SVM', label_encoder)
    results_list.append(svm_results)
    
    # ==================== 5. LDA ====================
    print(f"\n{'='*80}")
    print("训练 LDA 模型...")
    print(f"{'='*80}")
    
    lda_model = LinearDiscriminantAnalysis(
        solver='svd'
    )
    
    lda_model, lda_train_time = train_sklearn_model(
        lda_model, X_train, y_train, X_val, y_val, 'LDA'
    )
    lda_results = evaluate_model(lda_model, X_test, y_test, 'LDA', label_encoder)
    results_list.append(lda_results)
    
    # ==================== 6. MLP ====================
    print(f"\n{'='*80}")
    print("训练 MLP 模型...")
    print(f"{'='*80}")
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp_model, mlp_train_time = train_sklearn_model(
        mlp_model, X_train, y_train, X_val, y_val, 'MLP'
    )
    mlp_results = evaluate_model(mlp_model, X_test, y_test, 'MLP', label_encoder)
    results_list.append(mlp_results)
    
    # ==================== 结果汇总 ====================
    print("\n" + "="*80)
    print("生成对比结果...")
    print("="*80)
    
    # 保存结果表格
    save_results_table(results_list)
    
    # 绘制对比图
    plot_model_comparison(results_list)
    
    # 绘制混淆矩阵
    plot_confusion_matrices(results_list, label_encoder)
    
    # 找出最佳模型
    best_model = max(results_list, key=lambda x: x['accuracy'])
    print(f"\n{'='*80}")
    print(f"最佳模型: {best_model['model']}")
    print(f"准确率: {best_model['accuracy']*100:.2f}%")
    print(f"F1分数: {best_model['f1_score']*100:.2f}%")
    print("="*80)
    
    print("\n实验完成! 所有结果已保存。")

if __name__ == '__main__':
    main()
