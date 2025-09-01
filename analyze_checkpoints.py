#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint to see training progress."""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    print(f"Iteration: {checkpoint.get('iteration', 'Unknown')}")
    
    # Get loss histories
    policy_losses = checkpoint.get('policy_losses', [])
    value_losses = checkpoint.get('value_losses', [])
    total_losses = checkpoint.get('total_losses', [])
    
    print(f"Number of recorded iterations: {len(total_losses)}")
    
    if len(total_losses) > 0:
        print(f"Final losses - Policy: {policy_losses[-1]:.4f}, Value: {value_losses[-1]:.4f}, Total: {total_losses[-1]:.4f}")
        
        if len(total_losses) > 10:
            print(f"Last 10 total losses: {total_losses[-10:]}")
        else:
            print(f"All total losses: {total_losses}")
    
    return policy_losses, value_losses, total_losses

def compare_checkpoints():
    """Compare multiple checkpoints."""
    checkpoints = [
        'models/checkpoint_20.pt',
        'models/checkpoint_25.pt', 
        'models/checkpoint_30.pt',
        'models/checkpoint_32.pt',
        'models/checkpoint_34.pt'
    ]
    
    all_losses = {}
    
    for cp in checkpoints:
        try:
            policy_losses, value_losses, total_losses = analyze_checkpoint(cp)
            all_losses[cp] = {
                'policy': policy_losses,
                'value': value_losses, 
                'total': total_losses
            }
            print("-" * 50)
        except Exception as e:
            print(f"Error loading {cp}: {e}")
    
    # Plot comparison
    if all_losses:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        for cp, losses in all_losses.items():
            if losses['policy']:
                plt.plot(losses['policy'], label=cp.split('/')[-1])
        plt.title('Policy Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        for cp, losses in all_losses.items():
            if losses['value']:
                plt.plot(losses['value'], label=cp.split('/')[-1])
        plt.title('Value Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        for cp, losses in all_losses.items():
            if losses['total']:
                plt.plot(losses['total'], label=cp.split('/')[-1])
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('checkpoint_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved comparison plot as 'checkpoint_comparison.png'")

if __name__ == "__main__":
    compare_checkpoints()