import torch
import torch.nn as nn
import numpy as np      
import torch.nn.functional as F  
from utils.Layers import Siamese_DecoderLayer, Decoder, AttentionLayer, FullAttention


class SimpleMLPEncoder(nn.Module):
    """A simple MLP Encoder with shared weights for both past and future data."""
    def __init__(self, input_dim, output_dim):
        super(SimpleMLPEncoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.lineage_embeddings = nn.Embedding(configs.max_distance + 1, configs.d_model)
        self.encoder = SimpleMLPEncoder(configs.d_model, configs.d_model)  # 使用与嵌入同样维度的MLP
        self.decoder = Decoder(
            [
                Siamese_DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def generate_binomial_mask(self, B, L, N, C, p=0.5):
        return torch.from_numpy(np.random.binomial(1, 1 - p, size=(B, L, N, C))).to(torch.bool)

    def apply_mask(self, data, mask):
        return mask * data

    def apply_lineage_embeddings(self, data, distances):
        B, L, N, _ = data.shape
        embeddings = self.lineage_embeddings(distances)
        embeddings = embeddings.unsqueeze(1).unsqueeze(2).repeat(1, L, N, 1)
        return data + embeddings

    def forward(self, past_data, current_data, distances):
        B, L, N, C = past_data.shape

        # Masking past data
        mask = self.generate_binomial_mask(B, L, N, C, p=self.configs.mask_rate).to(past_data.device)
        masked_past = self.apply_mask(past_data, mask)
        masked_past = self.apply_lineage_embeddings(masked_past, distances)
        masked_past = self.encoder(masked_past.view(B * L * N, C)).view(B, L, N, C)

        # Current data processing
        zero_distances = torch.zeros(B, dtype=torch.long, device=current_data.device)
        current_data = self.apply_lineage_embeddings(current_data, zero_distances)
        current_data = self.encoder(current_data.view(B * L * N, C)).view(B, L, N, C)
        
        
        # Decode - combining history and future data
        # 重新组织数据以适应解码器的输入需求
        current_data_reshaped = current_data.view(B, N, L, C).permute(0, 1, 2, 3).contiguous().view(B * N, L, C)
        masked_past_reshaped = masked_past.view(B, N, L, C).permute(0, 1, 2, 3).contiguous().view(B * N, L, C)

        # 将重组后的数据输入到解码器
        dec_out = self.decoder(current_data_reshaped, masked_past_reshaped)
        pred = dec_out.view(B, L, N, C)

        return pred