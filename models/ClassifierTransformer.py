import numpy as np
import torch
from torch import nn

from models.TransformerModel import TransformerModel
from models.PositionalEncoding import PositionalEncoding


class ClassifierTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, max_len, dropout):
        super(ClassifierTransformer, self).__init__()
        # 创建一个线性变换层，维度input_dim4→d_model
        self.embedding_x = nn.Embedding(input_dim, d_model)  # 使用嵌入层
        self.embedding_normal_key_api_sequence = nn.Embedding(input_dim, d_model)
        self.embedding_abnormal_key_api_sequence = nn.Embedding(input_dim, d_model)

        self.transformer = TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )

        # 生成pe
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        # 生成一层encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # 多层encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # 维度d_model→output_dim
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, x, normal_key_api_sequence, abnormal_key_api_sequence):
        x = x.permute(1, 0)
        normal_key_api_sequence = normal_key_api_sequence.reshape(normal_key_api_sequence.shape[0], -1).permute(1, 0)
        abnormal_key_api_sequence = abnormal_key_api_sequence.reshape(abnormal_key_api_sequence.shape[0], -1).permute(1, 0)

        # 缩放
        x = self.embedding_x(x) * np.sqrt(self.d_model)
        normal_key_api_sequence = self.embedding_normal_key_api_sequence(normal_key_api_sequence) * np.sqrt(self.d_model)
        abnormal_key_api_sequence = self.embedding_abnormal_key_api_sequence(abnormal_key_api_sequence) * np.sqrt(self.d_model)

        x = torch.cat([x, normal_key_api_sequence, abnormal_key_api_sequence], dim=0)

        # 加上位置嵌入
        x = self.pos_encoder(x)

        output = self.transformer_encoder(x)

        # 调整输出形状为(batch, seq_len, d_model)
        output = output.permute(1, 0, 2)
        # 对所有位置的表示取平均
        output = torch.mean(output, dim=1)
        # 线性变换
        output = self.fc(output)
        # 使用sigmoid激活函数
        output = torch.sigmoid(output)

        return output
