#一个测试dataset.py的测试程序

import os
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from torch.utils.data import Dataset
import random
from dataset import TimeSeriesForecastingDataset




data_file_path = 'data/data_in_12_out_12_rescale_False.pkl'  # 替换为你的数据文件路径
index_file_path = 'data/index_in_12_out_12_rescale_False.pkl'  # 替换为你的索引文件路径
    
# 创建数据集实例
dataset = TimeSeriesForecastingDataset(data_file_path, index_file_path, mode='train')
    
# 创建 DataLoader
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
# 遍历 DataLoader 进行整个 epoch 的数据处理
for batch in data_loader:
    x ,y,z= batch  # 假设batch直接是数据，没有其他信息
    print("Original Data Shape:", x.shape)
    print("Masked Data Shape:", y.shape)
    print("Mask Shape:", z.shape)
    break