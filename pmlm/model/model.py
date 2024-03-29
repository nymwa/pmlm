import torch.nn as nn
from .embedding import Embedding
from .layer import Layer


class Model(nn.Module):

    def __init__(
            self,
            d_vocab,
            d_model,
            d_rnn,
            dropout,
            num_layers,
            padding_idx = 0):

        super().__init__()

        self.embedding = Embedding(
                d_vocab,
                d_model,
                dropout,
                padding_idx)

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Layer(d_model, d_rnn, dropout)
            for _ in range(num_layers)])

        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_vocab, bias = False)

    def forward(
            self,
            batch,
            enforce_sorted = True,
            hidden_list = None):

        x = self.embedding(batch.inputs)

        next_hidden_list = []
        for i in range(self.num_layers):
            if hidden_list is None:
                h = None
            else:
                h = hidden_list[i]
            layer = self.layers[i]

            x, h = layer(
                    x,
                    batch.lengths,
                    h = h,
                    enforce_sorted = enforce_sorted)
            next_hidden_list.append(h)

        x = self.layernorm(x)
        x = self.fc(x)
        return x, next_hidden_list

