import torch, math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding, dropout = 0.1, max_seq_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        positional_encoding = torch.zeros(max_seq_len, dim_embedding)
        position = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1)
        denom_term = torch.exp(torch.arange(0, dim_embedding, 2).float() * (-math.log(10000.0) / dim_embedding))
        positional_encoding[:, 0::2] = torch.sin(position * denom_term)
        positional_encoding[:, 1::2] = torch.cos(position * denom_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0),:]
        return self.dropout(x)