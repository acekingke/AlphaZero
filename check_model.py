import torch

# Load the checkpoint
checkpoint = torch.load('./models/checkpoint_32.pt')

# Print model structure
for key, value in checkpoint['model_state_dict'].items():
    print(f"{key}: {value.shape}")