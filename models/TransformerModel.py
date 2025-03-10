import numpy as np
import torch
from torch import nn

from models.PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int,
        dropout: float
    ):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src: torch.Tensor):
        src = src.permute(1, 0)

        src = self.embedding(src) * np.sqrt(self.d_model)

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        output = output.permute(1, 0, 2)

        output = torch.mean(output, dim=1)

        output = self.fc(output)

        output = torch.sigmoid(output)

        return output
