"""
AlphaZero neural network for 6x6 Othello.

Exact replica of alpha-zero-general's OthelloNNet.py:
- Input: 1-channel canonical board {-1, 0, +1}
- 4 plain Conv layers (3×3), first 2 with padding, last 2 without
- Shared FC (2048 → 1024 → 512) with BN + Dropout
- Output: log_softmax policy + tanh value

See docs/network-depth-receptive-field.md for the rationale.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device import get_device


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero neural network — exact alpha-zero-general replica.

    Input: (batch, board_x, board_y) — canonical board with {-1, 0, +1}
           Reshaped to (batch, 1, board_x, board_y) internally.

    Output:
        - log_prob_policy: (batch, action_size) — log_softmax over actions
        - value: (batch, 1), range [-1, 1] via tanh
    """

    def __init__(self, game_size=6, num_channels=512, dropout=0.3, device=None):
        super().__init__()

        self.device = device if device is not None else get_device()
        self.game_size = game_size
        self.num_channels = num_channels
        self.dropout_rate = dropout
        self.action_size = game_size * game_size + 1

        # ----- Convolutional trunk (1-channel input) -----
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        # After 4 conv layers: 6→6→6→4→2
        self.flattened_size = num_channels * (game_size - 4) * (game_size - 4)

        # ----- Shared FC -----
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # ----- Heads -----
        self.fc_policy = nn.Linear(512, self.action_size)
        self.fc_value = nn.Linear(512, 1)

        self.to(self.device)

    def forward(self, x):
        """
        Args:
            x: (batch, board_x, board_y) — canonical board

        Returns:
            log_prob_policy: (batch, action_size) — log_softmax
            value: (batch, 1) — tanh
        """
        # Reshape: (B, board_x, board_y) → (B, 1, board_x, board_y)
        s = x.view(-1, 1, self.game_size, self.game_size)

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        s = s.view(-1, self.flattened_size)

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.dropout_rate,
            training=self.training,
        )
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.dropout_rate,
            training=self.training,
        )

        pi = self.fc_policy(s)
        v = self.fc_value(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, x):
        """
        Predict with proper device handling. Returns (probs, value).

        Matches alpha-zero-general NNet.predict():
        returns torch.exp(log_pi) as probability vector.
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(np.array(x).astype(np.float64))
            if x.device != self.device:
                x = x.to(self.device)
            pi, v = self.forward(x)
            return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()
