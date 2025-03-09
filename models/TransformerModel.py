import numpy as np
import torch
from torch import nn

from models.PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, max_len, dropout):
        super(TransformerModel, self).__init__()
        # 创建一个线性变换层，维度input_dim4→d_model
        self.embedding = nn.Embedding(input_dim, d_model)  # 使用嵌入层
        # 生成pe
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        # 生成一层encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # 多层encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # 维度d_model→output_dim
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(1, 0)

        # 缩放
        src = self.embedding(src) * np.sqrt(self.d_model)

        # 加上位置嵌入
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        # 调整输出形状为(batch, seq_len, d_model)
        output = output.permute(1, 0, 2)
        # 对所有位置的表示取平均
        output = torch.mean(output, dim=1)
        # 线性变换
        output = self.fc(output)
        # 使用sigmoid激活函数
        output = torch.sigmoid(output)

        return output
