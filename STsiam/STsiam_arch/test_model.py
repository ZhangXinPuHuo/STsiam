#一个测试model.py的测试程序


import torch
import torch.nn as nn
import numpy as np      
import torch.nn.functional as F  

from model import Model

class Config:
    def __init__(self):
        self.d_model = 128       # 模型的维度
        self.d_ff = 512          # 解码器内前馈网络的维度
        self.d_layers = 4        # 解码器的层数
        self.n_heads = 8         # 注意力机制中的头数
        self.dropout = 0.15      # Dropout比率
        self.factor = 2          # 注意力中的缩放因子
        self.output_attention = False  # 是否输出注意力权重
        self.seq_len = 20        # 输入序列的长度
        self.mask_rate = 0.3     # 掩码率
        self.max_distance = 50   # 最大距离嵌入索引
        self.head_dropout = 0.1  # Flatten head的dropout
        self.activation = "relu"

# 实例化模型
configs = Config()
model = Model(configs)
model.eval()

# 生成随机数据
B, L, N, C = 10, 20, 3, configs.d_model  # 批次大小, 序列长度, 序列数量, 特征数
history_data = torch.randn(B, L, N, C)
future_data = torch.randn(B, L, N, C)
distances = torch.randint(0, configs.max_distance + 1, (B,))

# 运行模型
with torch.no_grad():  # 在评估模式下不追踪梯度
    processed_future = model(history_data, future_data, distances)

# 打印输出数据的形状

print("Processed Future Data Shape:", processed_future.shape)