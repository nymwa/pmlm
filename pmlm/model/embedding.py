import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, d_vocab, d_model, dropout, padding_idx = 0):
        super().__init__()
        self.token_embedding = nn.Embedding(d_vocab, d_model, padding_idx = padding_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.dropout(x)
        return x

