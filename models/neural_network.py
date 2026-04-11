import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device import get_device

class ResBlock(nn.Module):
    """Residual Block for the AlphaZero neural network."""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroNetwork(nn.Module):
    """Neural Network for AlphaZero."""
    def __init__(self, game_size, num_resblocks=4, num_channels=128, dropout=0.3, device=None):
        super(AlphaZeroNetwork, self).__init__()

        # Set device (CUDA, MPS, or CPU)
        self.device = device if device is not None else get_device()

        # Input is 3 channels (current player's pieces, opponent's pieces, player indicator)
        self.input_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_resblocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_dropout = nn.Dropout(p=dropout)
        self.policy_fc = nn.Linear(32 * game_size * game_size, game_size * game_size + 1)  # +1 for pass action

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_dropout1 = nn.Dropout(p=dropout)
        self.value_fc1 = nn.Linear(32 * game_size * game_size, 256)
        self.value_dropout2 = nn.Dropout(p=dropout)
        self.value_fc2 = nn.Linear(256, 1)

        # Move the entire model to the specified device
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor with shape (batch_size, 3, game_size, game_size)
            
        Returns:
            policy_logits: Policy logits with shape (batch_size, game_size * game_size + 1)
            value: Value prediction with shape (batch_size, 1)
        """
        # Move input to device if not already there
        if x.device != self.device:
            x = x.to(self.device)
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.value_dropout1(value)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout2(value)
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def predict(self, x):
        """
        Make a prediction with proper device and dtype handling.
        
        Args:
            x: Input tensor with shape (batch_size, 3, game_size, game_size)
            
        Returns:
            policy_logits: Policy logits with shape (batch_size, game_size * game_size + 1)
            value: Value prediction with shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            # Ensure input is on the correct device and has the correct dtype
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Convert to the same dtype as model parameters
            model_dtype = next(self.parameters()).dtype
            if x.dtype != model_dtype:
                x = x.to(dtype=model_dtype)
            
            # Move to the correct device
            if x.device != self.device:
                x = x.to(self.device)
            
            return self.forward(x)