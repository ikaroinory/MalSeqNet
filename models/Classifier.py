import torch
from torch import nn

from models.TransformerModel import TransformerModel


class Classifier(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_embedding: int,
        d_hidden: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        normal_sequence_max_len: int,
        abnormal_sequence_max_len: int,
        x_sequence_max_len: int,
        dropout: float = 0
    ):
        super(Classifier, self).__init__()
        self.normal_encoder = TransformerModel(
            d_input=d_input,
            d_output=d_embedding,
            d_hidden=d_hidden,
            num_heads=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=normal_sequence_max_len,
            dropout=dropout
        )
        self.normal_embedding_relu = nn.LeakyReLU()
        self.abnormal_encoder = TransformerModel(
            d_input=d_input,
            d_output=d_embedding,
            d_hidden=d_hidden,
            num_heads=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=abnormal_sequence_max_len,
            dropout=dropout
        )
        self.abnormal_embedding_relu = nn.LeakyReLU()
        self.x_encoder = TransformerModel(
            d_input=d_input,
            d_output=d_embedding,
            d_hidden=d_hidden,
            num_heads=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=x_sequence_max_len,
            dropout=dropout
        )
        self.x_embedding_relu = nn.LeakyReLU()

        self.relu = nn.LeakyReLU()

        self.fc = nn.Linear(d_embedding * 3, 1)

    def forward(self, x: torch.Tensor, normal_key_api_sequence: torch.Tensor, abnormal_key_api_sequence: torch.Tensor):
        normal_embedding = self.normal_encoder(normal_key_api_sequence)
        normal_embedding = self.normal_embedding_relu(normal_embedding)

        abnormal_embedding = self.abnormal_encoder(abnormal_key_api_sequence)
        abnormal_embedding = self.abnormal_embedding_relu(abnormal_embedding)

        x_embedding = self.x_encoder(x)
        x_embedding = self.x_embedding_relu(x_embedding)

        x_embedding = torch.cat([x_embedding, normal_embedding, abnormal_embedding], dim=1)
        x_embedding = self.relu(x_embedding)

        return self.fc(x_embedding)
