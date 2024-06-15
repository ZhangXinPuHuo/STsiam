import torch
from torch import nn

from .transformer_layers import TransformerLayers,TransformerDecoder

class TimeSiam(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder_layers, decoder_layers, mlp_ratio, dropout=0.1, mask_ratio=0.25, lineage_size=100):
        super().__init__()
        self.encoder = TransformerLayers(embed_dim, encoder_layers, mlp_ratio, num_heads, dropout)
        self.decoder = TransformerDecoder(embed_dim, decoder_layers, mlp_ratio, num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.lineage_embeddings = nn.Embedding(lineage_size, embed_dim)
        self.mask_ratio = mask_ratio

    def apply_mask(self, x):
        mask = (torch.rand(x.size()) < self.mask_ratio).float().to(x.device)
        return x * mask

    def forward(self, past, curr, lineage_d):
        B, _, L, C = past.shape
        
        # Apply mask to curr
        curr = self.apply_mask(curr)
        
        # Lineage embedding
        lineage_embedding_curr = self.lineage_embeddings(torch.zeros(B, dtype=torch.long, device=curr.device))
        lineage_embedding_past = self.lineage_embeddings(lineage_d)
        
        # Reshape and add lineage embedding
        lineage_embedding_curr = lineage_embedding_curr.view(B, 1, 1, C)  # Shape: (B, 1, 1, D)
        lineage_embedding_past = lineage_embedding_past.view(B, 1, 1, C)  # Shape: (B, 1, 1, D)
        
        curr = curr + lineage_embedding_curr
        past = past + lineage_embedding_past

        # Encoding
        past_encoded = self.encoder(past)
        curr_encoded = self.encoder(curr)

        # Decoding
        reconstructed_curr = self.decoder(curr_encoded, past_encoded)

        reconstructed_curr = self.output_layer(reconstructed_curr)
        return reconstructed_curr, curr
    