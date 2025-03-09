from torch import Tensor, nn


class VectorEncoder(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, d_output: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.attention = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.Softmax(dim=1)
        )

        self.output_layer = nn.Linear(d_hidden, d_output)

    def forward(self, x: Tensor):
        # Input: [batch_size, sequence_length, input_size]
        # Output: [batch_size, 1, output_size]

        features = self.feature_extractor(x)
        attn_weights = self.attention(features)
        weighted_features = (features * attn_weights).sum(dim=1, keepdim=True)
        output_vector = self.output_layer(weighted_features)
        return output_vector
