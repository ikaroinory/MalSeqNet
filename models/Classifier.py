import torch
from torch import nn

from models.TransformerModel import TransformerModel


class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        normal_sequence_max_len: int,
        abnormal_sequence_max_len: int,
        x_sequence_max_len: int,
        dropout: float
    ):
        super(Classifier, self).__init__()
        self.normal_encoder = TransformerModel(
            input_dim=input_dim,
            output_dim=embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=normal_sequence_max_len,
            dropout=dropout
        )
        self.normal_embedding_relu = nn.LeakyReLU()
        self.abnormal_encoder = TransformerModel(
            input_dim=input_dim,
            output_dim=embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=abnormal_sequence_max_len,
            dropout=dropout
        )
        self.abnormal_embedding_relu = nn.LeakyReLU()
        self.x_encoder = TransformerModel(
            input_dim=input_dim,
            output_dim=embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=x_sequence_max_len,
            dropout=dropout
        )
        self.x_embedding_relu = nn.LeakyReLU()

        self.relu = nn.LeakyReLU()

        self.fc = nn.Linear(embedding_dim * 3, 1)

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
