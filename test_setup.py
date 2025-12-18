"""
快速测试脚本 - 验证数据加载和模型初始化
"""

import torch
import pandas as pd
from main import load_and_preprocess_data, MambaClassifier

print("="*60)
print("快速测试 - 验证系统配置")
print("="*60)

# 测试数据加载
print("\n1. 测试数据加载...")
try:
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, scaler = \
        load_and_preprocess_data('data.csv', test_size=0.2, val_size=0.1)
    print("✓ 数据加载成功!")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    exit(1)

# 测试模型初始化
print("\n2. 测试模型初始化...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(label_encoder.classes_),
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ).to(device)
    
    print(f"✓ 模型初始化成功!")
    print(f"  设备: {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    exit(1)

# 测试前向传播
print("\n3. 测试前向传播...")
try:
    sample_input = torch.randn(4, X_train.shape[1]).to(device)
    output = model(sample_input)
    print(f"✓ 前向传播成功!")
    print(f"  输入形状: {sample_input.shape}")
    print(f"  输出形状: {output.shape}")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ 所有测试通过! 系统配置正确,可以开始训练。")
print("="*60)
print("\n运行以下命令开始训练:")
print("  python main.py")
