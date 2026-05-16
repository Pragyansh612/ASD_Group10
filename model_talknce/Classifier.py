import torch
from torch import nn
import math

class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(channel, momentum=0.01, eps=0.001)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        identity = x.transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        x = (identity * w).transpose(1, 2)
        return x

class Detector(nn.Module):
    def __init__(self, channel):
        super(Detector, self).__init__()
        # Replace bidirectional GRU with single transformer encoder layer
        # input: (B, T, channel) — same as GRU input
        # output: (B, T, channel//2) — same as GRU output to match Fusion input
        self.input_proj = nn.Linear(channel, channel//2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channel//2,
            nhead=4,
            dim_feedforward=channel,
            dropout=0.5,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, T, channel)
        x = self.drop(x)
        x = self.input_proj(x)          # (B, T, channel//2)
        x = self.transformer(x)          # (B, T, channel//2)
        x = x.transpose(1, 2)           # (B, channel//2, T) to match reshape in Model.py
        return x
