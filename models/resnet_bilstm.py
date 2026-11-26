# models/resnet_bilstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Residual Block
# --------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


# --------------------------------------------------
# Hybrid ResNet + BiLSTM Model (Regression)
# --------------------------------------------------

class HybridResNetBiLSTM(nn.Module):
    def __init__(self, input_channels=11, stat_dim=56):
        super().__init__()

        # ----- CNN Backbone -----
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)

        self.dropout = nn.Dropout(0.3)

        # ----- BiLSTM Branch -----
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # ----- Statistical + Previous Cycles Branch -----
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # ----- Final Regression Head -----
        self.fc = nn.Linear(64 * 2 + 32, 1)

    def forward(self, x, stats):

        # CNN
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.dropout(x)

        # LSTM expects (batch, seq, features)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        temporal_embedding = torch.mean(lstm_out, dim=1)

        # Statistical branch
        stat_embedding = self.stat_mlp(stats)

        # Fusion
        combined = torch.cat([temporal_embedding, stat_embedding], dim=1)

        out = self.fc(combined)

        return torch.sigmoid(out)  # RUL between 0–1
