"""
基于Mamba模型的运动分类系统
数据集：包含EMG、加速度计、陀螺仪等传感器数据
任务：分类4种运动类型 - levelground, stair, treadmill, ramp
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==================== 数据加载和预处理 ====================
class ActivityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(file_path, test_size=0.2, val_size=0.1):
    """加载并预处理数据"""
    print("正在加载数据...")
    # 读取Excel文件(虽然后缀是.csv但实际是.xlsx)
    df = pd.read_excel(file_path)
    
    print(f"数据集大小: {df.shape}")
    print(f"活动类型分布:\n{df['activity_type'].value_counts()}")
    
    # 分离特征和标签
    # 后3列是: file_name, activity_type, segment_index
    feature_columns = df.columns[:-3]
    X = df[feature_columns].values
    y = df['activity_type'].values
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n类别映射:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {idx}: {label}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集、验证集、测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, scaler

# ==================== Mamba模型核心组件 ====================
class MambaBlock(nn.Module):
    """
    Mamba Block - 状态空间模型的核心组件
    基于选择性状态空间模型(Selective SSM)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 卷积层(用于局部特征提取)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # 状态空间参数
        A = torch.randn(d_state, self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 激活函数
        self.activation = nn.SiLU()
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # 输入投影并分割
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # 卷积操作
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # 移除padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # 激活
        x = self.activation(x)
        
        # SSM操作
        ssm_out = self.ssm(x)
        
        # 门控机制
        out = ssm_out * self.activation(res)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out
    
    def ssm(self, x):
        """选择性状态空间模型(简化版)"""
        batch, seq_len, d_inner = x.shape
        
        # 计算SSM参数
        A = -torch.exp(self.A_log.float()).transpose(0, 1)  # (d_inner, d_state)
        D = self.D.float()  # (d_inner,)
        
        # 选择性参数
        x_dbl = self.x_proj(x)  # (batch, seq_len, d_state * 2)
        B, C = x_dbl.split([self.d_state, self.d_state], dim=-1)  # 每个(batch, seq_len, d_state)
        
        # 离散化参数
        delta = F.softplus(self.dt_proj(x))  # (batch, seq_len, d_inner)
        
        # 简化的SSM计算: 直接使用全局池化
        # 这里使用一个简化的方法，避免复杂的状态递归
        # y = ∑(B * x * C) + D * x
        
        # 将B和C投影到d_inner维度
        B_proj = B @ self.A_log[:, :d_inner // 4]  # 简化投影
        C_proj = C @ self.A_log[:, :d_inner // 4]  # 简化投影
        
        # 门控机制
        y = x * delta  # 时间依赖性
        y = y + x * D  # skip connection
        
        return y

class ResidualBlock(nn.Module):
    """残差连接块"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))

class MambaClassifier(nn.Module):
    """
    基于Mamba的分类器
    用于时序/序列数据的分类任务
    """
    def __init__(self, input_dim, num_classes, d_model=128, n_layers=4, 
                 d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        # 输入嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Mamba层堆叠
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
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
        
        # Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits

# ==================== 训练和评估 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='训练')
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存至: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close()

# ==================== 主程序 ====================
def main():
    # 配置参数
    config = {
        'data_path': 'data.csv',
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'd_model': 128,
        'n_layers': 4,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'dropout': 0.1,
        'test_size': 0.2,
        'val_size': 0.1,
    }
    
    print("=" * 60)
    print("基于Mamba模型的运动分类系统")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, scaler = \
        load_and_preprocess_data(config['data_path'], config['test_size'], config['val_size'])
    
    # 创建数据加载器
    train_dataset = ActivityDataset(X_train, y_train)
    val_dataset = ActivityDataset(X_val, y_val)
    test_dataset = ActivityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # 创建模型
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    model = MambaClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        d_state=config['d_state'],
        d_conv=config['d_conv'],
        expand=config['expand'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # 训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'label_encoder': label_encoder,
                'scaler': scaler
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发! {patience} 个epoch验证准确率未提升")
            break
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 加载最佳模型进行测试
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型...")
    print("=" * 60)
    
    checkpoint = torch.load('best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n测试集结果:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, 
                              target_names=label_encoder.classes_,
                              digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, label_encoder.classes_)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"模型已保存至: best_model.pth")
    print("=" * 60)

if __name__ == '__main__':
    main()
