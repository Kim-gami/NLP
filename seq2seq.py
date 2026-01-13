import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model,
                 nhead, src_vocab_size = tokenizer_src.vocab_size,
                 tgt_vocab_size = tokenizer_tgt.vocab_size,
                 dim_feedforward = 512, dropout = 0.1):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pos_encoder = PositionalEncoding(d_model, dropout)
        self.trg_pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model = d_model, nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward, dropout = dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, trg, src_mask = None,
                src_padding_mask = None, trg_mask = None,
                trg_padding_mask = None, memory_key_padding_mask = None):
        src = self.src_embedding(src) * (self.d_model ** 0.5)
        src = self.src_pos_encoder(src)
        trg = self.trg_embedding(trg) * (self.d_model ** 0.5)
        trg = self.trg_pos_encoder(trg)
        output = self.transformer(src, trg, src_mask, trg_mask, None,
                                  src_padding_mask, trg_padding_mask,
                                  memory_key_padding_mask)
        output = self.fc(self.dropout(output))
        return output