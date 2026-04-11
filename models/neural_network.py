"""
AlphaZero neural network for 6x6 Othello.

Architecture follows alpha-zero-general's proven design:
- 4 plain Conv layers (no ResBlock — unnecessary for small board)
- First 2 convs with padding (preserve 6x6 spatial)
- Last 2 convs without padding (spatial compression 6→4→2)
- Shared FC layers (2048 → 1024 → 512)
- Split policy and value heads from shared 512 representation

Receptive field after 4 convs: 9x9 (covers the 6x6 board with margin).
Parameter count: ~10M (close to alpha-zero-general's 512-channel network).

See docs/network-depth-receptive-field.md for the rationale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device import get_device


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero neural network — alpha-zero-general style.

    Input: (batch, 3, 6, 6)
        - Channel 0: current player's pieces (0/1)
        - Channel 1: opponent's pieces (0/1)
        - Channel 2: current player indicator (all 1s in canonical form)

    Output:
        - policy_logits: (batch, 37)  — 36 board positions + 1 pass action
        - value: (batch, 1), range [-1, 1] from current player's perspective
    """

    def __init__(self, game_size=6, num_channels=512, dropout=0.3, device=None):
        super().__init__()

        # Store configuration
        self.device = device if device is not None else get_device()
        self.game_size = game_size
        self.num_channels = num_channels
        self.dropout_rate = dropout

        # ----- Convolutional trunk -----
        # 4 plain conv layers. First 2 preserve spatial (padding=1).
        # Last 2 have no padding, shrinking 6→4→2, forcing global compression.
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1)  # no padding: 6→4
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1)  # no padding: 4→2

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        # After 4 conv layers, spatial is 2x2 regardless of game_size (as long as game_size >= 4).
        # For game_size > 6, we need to recompute; for 6x6 it's always 2.
        self.post_conv_spatial = 2  # 6 → 6 → 6 → 4 → 2
        self.flattened_size = num_channels * self.post_conv_spatial * self.post_conv_spatial

        # ----- Shared fully-connected layers -----
        # Progressive compression 2048 → 1024 → 512 with BN + Dropout.
        # Both policy and value heads branch from the 512-dim shared representation.
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # ----- Policy head -----
        # Single linear from 512 → (size*size + 1) for action logits
        self.fc_policy = nn.Linear(512, game_size * game_size + 1)

        # ----- Value head -----
        # Single linear from 512 → 1, passed through tanh for [-1, 1] range
        self.fc_value = nn.Linear(512, 1)

        # Move everything to target device
        self.to(self.device)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, 3, game_size, game_size)

        Returns:
            policy_logits: (batch, game_size * game_size + 1)
            value: (batch, 1)
        """
        # Convolutional trunk
        # Shape: (B, 3, 6, 6) → (B, 512, 6, 6)
        s = F.relu(self.bn1(self.conv1(x)))
        # Shape: (B, 512, 6, 6) → (B, 512, 6, 6)
        s = F.relu(self.bn2(self.conv2(s)))
        # Shape: (B, 512, 6, 6) → (B, 512, 4, 4)
        s = F.relu(self.bn3(self.conv3(s)))
        # Shape: (B, 512, 4, 4) → (B, 512, 2, 2)
        s = F.relu(self.bn4(self.conv4(s)))

        # Flatten: (B, 512, 2, 2) → (B, 2048)
        s = s.view(s.size(0), self.flattened_size)

        # Shared FC trunk with BN, ReLU, and dropout
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

        # Split heads from shared 512-dim representation
        policy_logits = self.fc_policy(s)
        value = torch.tanh(self.fc_value(s))

        return policy_logits, value

    def predict(self, x):
        """
        Make a prediction with proper device and dtype handling.
        Puts model in eval mode (disables dropout, BN uses running stats).

        Args:
            x: Input tensor or numpy array (batch, 3, game_size, game_size)

        Returns:
            policy_logits: (batch, game_size * game_size + 1)
            value: (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dtype != next(self.parameters()).dtype:
                x = x.to(dtype=next(self.parameters()).dtype)
            if x.device != self.device:
                x = x.to(self.device)
            return self.forward(x)
