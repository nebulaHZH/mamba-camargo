# 基于Mamba模型的运动分类系统

## 项目简介

本项目使用最先进的Mamba(状态空间模型)架构对人体运动进行分类。系统可以识别4种不同的运动类型:
- **levelground**: 平地行走
- **stair**: 楼梯行走
- **treadmill**: 跑步机
- **ramp**: 斜坡行走

## 数据集说明

- **特征维度**: 196维
- **特征类型**: EMG肌电信号、加速度计、陀螺仪等传感器数据
- **标签列**: `activity_type` (运动类型)
- **其他列**: `file_name` (文件名), `segment_index` (片段索引)

## 环境要求

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm openpyxl
```

## 项目结构

```
mamba/
├── data.csv                 # 数据集(Excel格式)
├── main.py                  # 主训练脚本
├── inference.py             # 推理脚本
├── best_model.pth          # 训练好的最佳模型
├── training_history.png    # 训练历史图
└── confusion_matrix.png    # 混淆矩阵
```

## 快速开始

### 1. 训练模型

```bash
python main.py
```

训练过程会:
- 自动加载和预处理数据
- 划分训练集(70%)、验证集(10%)、测试集(20%)
- 训练Mamba分类模型
- 保存最佳模型到 `best_model.pth`
- 生成训练历史图和混淆矩阵

### 2. 使用模型预测

```python
from inference import ActivityPredictor

# 初始化预测器
predictor = ActivityPredictor('best_model.pth')

# 从文件预测
results = predictor.predict_from_csv('data.csv', 'predictions.csv')

# 单个样本预测
import pandas as pd
df = pd.read_excel('data.csv', nrows=1)
sample = df.iloc[0, :-3].values
predictions, probabilities = predictor.predict(sample)
print(f"预测结果: {predictions[0]}")
```

## Mamba模型架构

### 核心组件

1. **选择性状态空间模型(Selective SSM)**
   - 动态选择重要信息
   - 高效处理长序列
   - 线性时间复杂度

2. **MambaBlock**
   - 输入投影
   - 1D卷积(局部特征提取)
   - SSM状态更新
   - 门控机制
   - 输出投影

3. **分类器架构**
   - 输入嵌入层: Linear(196 → 128)
   - 4层Mamba残差块
   - 全局平均池化
   - 分类头: Linear(128 → 64 → 4)

### 模型参数

- `d_model`: 128 (模型维度)
- `n_layers`: 4 (Mamba层数)
- `d_state`: 16 (状态空间维度)
- `d_conv`: 4 (卷积核大小)
- `expand`: 2 (扩展因子)
- `dropout`: 0.1 (dropout率)

## 训练配置

```python
config = {
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'test_size': 0.2,
    'val_size': 0.1,
}
```

## 性能指标

训练后会输出:
- 训练/验证损失和准确率曲线
- 测试集准确率
- 详细分类报告(精确率、召回率、F1分数)
- 混淆矩阵

## 模型优势

1. **高效性**: Mamba模型具有线性时间复杂度,比Transformer更快
2. **长序列建模**: 能够有效处理长距离依赖
3. **选择性机制**: 动态选择重要特征,提高分类准确率
4. **泛化能力**: 通过残差连接和dropout提高模型泛化性能

## 自定义配置

如需修改模型参数,可编辑 `main.py` 中的 `config` 字典:

```python
config = {
    'd_model': 256,      # 增加模型容量
    'n_layers': 6,       # 增加深度
    'batch_size': 128,   # 调整批次大小
    'learning_rate': 0.0005,  # 调整学习率
}
```

## 注意事项

1. 数据文件 `data.csv` 实际是Excel格式(.xlsx),代码已自动处理
2. 如果使用GPU训练,确保已安装CUDA版本的PyTorch
3. 训练过程支持早停机制,防止过拟合
4. 模型检查点包含完整的预处理器(scaler和label_encoder),方便部署

## 可视化结果

训练完成后会生成:
- `training_history.png`: 展示训练和验证的损失/准确率变化
- `confusion_matrix.png`: 展示各类别的分类效果

## 许可证

MIT License
