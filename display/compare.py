import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 准备数据
data = {
    'Model': ['Mamba', 'Transformer', 'XGBoost', 'SVM', 'LDA', 'MLP'],
    'Accuracy': [97.61,97.21,95.02,83.60,79.75,96.07],
    'F1-Score': [97.61,97.21,95.00,83.90,79.49,96.08],
    'Precision': [97.61,97.21,95.02,85.24,79.53,96.09],
    'Recall': [97.61,97.21,95.02,83.60,79.75,96.07]
}
df = pd.DataFrame(data)

# 2. 学术风格配置 (字号微调以适应小图)
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 9,             # 全局字号稍微调小
    "axes.titlesize": 11,       # 标题字号
    "axes.linewidth": 0.8,      # 边框变细
    "grid.color": "#E0E0E0",    
    "grid.linewidth": 0.8,
}
plt.rcParams.update(config)

# 3. 创建画布 - 关键修改
# figsize=(7, 5): 更加紧凑，适合直接插入文档
# layout="constrained": 自动计算最佳间距，防止小图时标签被切掉
fig, axes = plt.subplots(2, 2, figsize=(6, 4), dpi=300, layout="constrained")
axes = axes.flatten()
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

# 样式
line_color = '#28536B'
marker_style = 'o'

for i, metric in enumerate(metrics):
    ax = axes[i]
    x_data = df['Model']
    y_data = df[metric]
    
    # 绘制折线
    ax.plot(x_data, y_data, marker=marker_style, color=line_color, 
            linewidth=1.2, markersize=4, linestyle='-', zorder=3)
    
    # 添加数值标签
    for j, val in enumerate(y_data):
        # 动态调整标签位置
        offset = 2.0 if val < 90 else 1.2
        ax.text(j, val + offset, f'{val:.1f}', # 保留1位小数更简洁，如果需要2位改为 .2f
                ha='center', va='bottom', fontsize=7, color='#333333')

    # 标题设置
    ax.set_title(metric, fontweight='bold', pad=8)
    ax.set_xlabel('')
    
    # Y轴范围动态调整
    y_min = y_data.min()
    ax.set_ylim(y_min - 8, 108) # 留出更多上下空间，防止文字顶格
    
    # X轴设置 (关键：防止重叠)
    ax.set_xticks(range(len(x_data)))
    # 因为图变小了，让文字稍微倾斜20度，防止 Transformer 和 XGBoost 挤在一起
    ax.set_xticklabels(x_data, rotation=20, ha='right')
    
    # 美化
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()

fig.savefig('small_chart.pdf')