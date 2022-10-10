from models.self_attention_pool import PreNorm, Attention, FeedForward
import math
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import nn, einsum

class SeqAttention(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, depth=1, lstm_dim=256, lstm_layers=1, bidirectional=False, mlp_dim=256, proj_dim=-1, dropout = 0.):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        
        lstm_dropout = dropout if lstm_layers > 1 else 0
        self.lstm = nn.LSTM(dim, lstm_dim, num_layers=lstm_layers, dropout=lstm_dropout, bidirectional=bidirectional, batch_first=True)
        
        if bidirectional:
            lstm_dim *= 2

        if proj_dim > 0:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(lstm_dim),
                nn.Linear(lstm_dim, proj_dim)
            )
        else:
            self.mlp_head = nn.Identity()

    def forward(self, x, masks=None):
        x = x.type(torch.float32)
        x += self.pos_encoding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x, _ = self.lstm(x)
        x = self.mlp_head(x[:,-1])
        
        return x