import torch
from torch import nn

from models.PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0
    ):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(d_input, d_hidden)

        self.pos_encoder = PositionalEncoding(d_hidden, dropout, max_len)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_hidden, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_hidden, d_output)
        self.d_model = d_hidden

    def forward(self, src: torch.Tensor):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, device=src.device))

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        output = torch.mean(output, dim=1)

        output = self.fc(output)

        return output
