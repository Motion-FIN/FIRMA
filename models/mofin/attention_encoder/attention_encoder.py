import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class SelfAttentionLayer(nn.Module):
#     def __init__(self, hidden_dim, n_heads):
#         super(SelfAttentionLayer, self).__init__()
#         self.self_attention = nn.MultiheadAttention(hidden_dim, n_heads)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x):
#         # Inputs x: (sequence_length, batch_size, d_model)
#         # Outputs output: (sequence_length, batch_size, d_model)
#         # print(f'x.shape : {x.shape}')
#         output, _ = self.self_attention(x, x, x)
#         output = self.layer_norm(x + (output * x))
#         return output

# # Transformer Encoder
# class TransformerEncoder(nn.Module):
#     def __init__(self, cfg):
#         super(TransformerEncoder, self).__init__()
#         self.cfg = cfg
#         input_dim = cfg.input_dim
#         hidden_dim = cfg.hidden_dim
#         num_heads = cfg.num_heads
#         num_layers = cfg.num_layers
#         self.layers = nn.ModuleList([SelfAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)])

#     def forward(self, x):
#         # Inputs x: (sequence_length, batch_size, d_model)
#         # Outputs output: (sequence_length, batch_size, d_model)
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        input_dim = cfg.input_dim
        hidden_dim = cfg.hidden_dim
        num_heads = cfg.num_heads
        num_layers = cfg.num_layers

        self.embedding = nn.Linear(input_dim, hidden_dim)
        # self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        # print(f'xx : {x.shape}')
        x = self.embedding(x)
        # x = self.positional_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=65536):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # print(f'x.shape : {x.shape}')
        residual = x
        x = self.self_attention(x, x, x)[0]
        x = self.dropout(x)
        x = x + residual
        x = self.norm1(x)
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)

        return x
