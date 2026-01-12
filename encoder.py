import torch, math
import torch.nn as nn
from positional_encoding import PositionalEncoding

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, num_classes):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, key_padding_mask = None):
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask = key_padding_mask)

        x = x.mean(dim = 0)

        x = self.fc(x)
        x = torch.sigmoid(x)
        return x