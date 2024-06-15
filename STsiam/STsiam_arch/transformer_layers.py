import torch
from torch import nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + src2
        src2 = self.norm2(src)
        src2 = F.relu(self.linear1(src2))
        src2 = self.dropout(self.linear2(src2))
        return src + src2

class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, dropout)
        self.transformer_encoder = nn.ModuleList([encoder_layers for _ in range(nlayers)])

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B * N, L, D).transpose(0, 1)
        for layer in self.transformer_encoder:
            src = layer(src)
        output = src.transpose(0, 1).view(B, N, L, D)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, target, memory):
        target2 = self.norm1(target)
        target2, _ = self.self_attn(target2, target2, target2)
        target = target + target2
        target2 = self.norm2(target)
        target2, _ = self.multihead_attn(target2, memory, memory)
        target = target + target2
        target2 = self.norm3(target)
        target2 = F.relu(self.linear1(target2))
        target2 = self.dropout(self.linear2(target2))
        return target + target2

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, dropout)
        self.transformer_decoder = nn.ModuleList([decoder_layers for _ in range(nlayers)])

    def forward(self, target, memory):
        B, N, L, D = target.shape
        target = target * math.sqrt(self.d_model)
        target = target.view(B * N, L, D).transpose(0, 1)
        memory = memory.view(B * N, L, D).transpose(0, 1)
        for layer in self.transformer_decoder:
            target = layer(target, memory)
        output = target.transpose(0, 1).view(B, N, L, D)
        return output