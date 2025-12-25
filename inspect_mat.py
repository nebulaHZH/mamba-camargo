"""
检查mat文件的结构
"""
import scipy.io as sio
import glob
import os

# 读取第一个mat文件看看结构
mat_files = glob.glob(r'E:\0_yao\dataset\carmago_struct\AB06\**\*.mat', recursive=True)

if mat_files:
    print(f"找到 {len(mat_files)} 个mat文件")
    print(f"\n检查第一个文件: {mat_files[0]}")
    
    # 加载mat文件
    mat_data = sio.loadmat(mat_files[0])
    
    print("\n=== Mat文件内容 ===")
    for key in mat_data.keys():
        if not key.startswith('__'):  # 忽略元数据
            print(f"\n变量名: {key}")
            print(f"  类型: {type(mat_data[key])}")
            if hasattr(mat_data[key], 'shape'):
                print(f"  形状: {mat_data[key].shape}")
            if hasattr(mat_data[key], 'dtype'):
                print(f"  数据类型: {mat_data[key].dtype}")
    
    # 检查几个常见的可能变量名
    possible_keys = ['data', 'signals', 'emg', 'imu', 'label', 'activity', 'X', 'Y']
    print("\n=== 检查常见变量 ===")
    for key in possible_keys:
        if key in mat_data:
            print(f"✓ 找到 '{key}': shape={mat_data[key].shape}")
    
    # 如果是struct，可能需要特殊处理
    print("\n=== 完整键列表 ===")
    print([k for k in mat_data.keys() if not k.startswith('__')])
    
    # 详细查看data_struct
    if 'data_struct' in mat_data:
        print("\n=== data_struct 详细信息 ===")
        ds = mat_data['data_struct']
        print(f"形状: {ds.shape}")
        print(f"dtype字段: {ds.dtype.names}")
        
        # 查看第一个样本
        print("\n第一个时间步数据:")
        for field in ds.dtype.names:
            val = ds[field][0, 0]
            if hasattr(val, 'shape'):
                print(f"  {field}: shape={val.shape}, dtype={val.dtype}")
                if val.size < 10:
                    print(f"    值: {val}")
            else:
                print(f"  {field}: {val}")
        
        # 尝试提取一个肌肉通道的完整时序数据
        print("\n=== 提取完整时序数据示例 ===")
        muscle_name = 'gastrocmed'  # 腓肠肌
        if muscle_name in ds.dtype.names:
            # 提取所有时间步的该肌肉数据
            muscle_data = [ds[muscle_name][i, 0] for i in range(len(ds))]
            print(f"提取 {muscle_name} 数据:")
            print(f"  时间步数: {len(muscle_data)}")
            print(f"  第一个值: {muscle_data[0]}")
            print(f"  数据类型: {type(muscle_data[0])}")
    
    # 查看col_names
    if 'col_names' in mat_data:
        print("\n=== 列名信息 ===")
        col_names = mat_data['col_names']
        print(f"列名: {col_names}")
    
else:
    print("未找到mat文件！")
