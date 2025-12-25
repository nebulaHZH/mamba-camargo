"""
快速测试数据加载是否正常
"""
import sys
sys.path.append('.')
from train_mamba_sequence import load_dataset

print("测试数据加载...")
print("=" * 60)

# 只加载每个类别的前2个文件进行测试
X, y, file_sources = load_dataset(
    data_root=r'E:\0_yao\dataset\carmago_struct',
    window_size=500,
    stride=250,
    max_files_per_class=2  # 只加载每类前2个文件
)

print("\n" + "=" * 60)
print("数据加载测试成功!")
print(f"总样本数: {len(X)}")
print(f"数据形状: {X.shape}")
print(f"标签形状: {y.shape}")
print("=" * 60)
