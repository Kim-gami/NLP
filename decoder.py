import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, dropout):
        super().__init__()

        self.memory_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.memory_pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tgt_pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model = embedding_dim, nhead = 8, dim_feedforward = 2048, dropout = dropout),
        num_layers = num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.d_model = embedding_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform_(self.memory_embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.tgt_embedding.weight, -initrange, initrange)

        for param in self.decoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        nn.init.uniform_(self.fc.weight, -initrange, initrange)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tgt, memory = None, tgt_mask = None, memory_mask = None, memory_key_padding_mask = None, tgt_key_padding_mask = None):
        tgt = self.tgt_embedding(tgt) * self.d_model ** 0.5
        tgt = self.tgt_pos_encoder(tgt)
        print(tgt)
        memory = self.memory_embedding(memory) * self.d_model ** 0.5
        memory = self.memory_pos_encoder(memory)
        print(memory)
        output = self.decoder(tgt = tgt, memory = memory, tgt_mask = tgt_mask, memory_mask = memory_mask, memory_key_padding_mask = memory_key_padding_mask, tgt_key_padding_mask = tgt_key_padding_mask)
        print(output)
        output = self.fc(output)
        return output