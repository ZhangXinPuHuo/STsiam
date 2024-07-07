import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset import TimeSeriesForecastingDataset
from model import Model
# 定义配置类
class Config:
    def __init__(self):
        self.d_model = 3
        self.d_ff = 512
        self.d_layers = 2
        self.n_heads = 4
        self.dropout = 0.1
        self.factor = 2
        self.output_attention = False
        self.seq_len = 12
        self.mask_rate = 0.3
        self.max_distance = 16992
        self.head_dropout = 0.1
        self.learning_rate = 0.001
        self.epochs = 5
        self.batch_size = 32
        self.activation = "relu"

# 初始化配置
configs = Config()

# 初始化模型
model = Model(configs)
model.train()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

data_file_path = 'data/data_in_12_out_12_rescale_False.pkl'  # 替换为你的数据文件路径
index_file_path = 'data/index_in_12_out_12_rescale_False.pkl'  # 替换为你的索引文件路径
# 数据加载器
dataset = TimeSeriesForecastingDataset(data_file_path, index_file_path, mode='train')
dataloader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

# 训练循环
for epoch in range(configs.epochs):
    total_loss = 0
    for past_data, current_data, distances in dataloader:
        optimizer.zero_grad()
        outputs = model(past_data, current_data, distances)
        loss = criterion(outputs, current_data)  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

print("Training completed.")
