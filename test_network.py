#!/usr/bin/env python3

import torch
import numpy as np
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from utils.device import get_device

def test_network_output():
    """Test neural network output on initial board position."""
    print("Testing neural network output...")
    
    # Load model
    device = get_device(use_mps=True)
    model = AlphaZeroNetwork(game_size=6, device=device)
    
    # Test with checkpoint_30 (best trained model)
    checkpoint = torch.load('models/checkpoint_30.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create initial game state
    env = OthelloEnv(size=6)
    env.reset()
    
    print("Initial board:")
    env.render()
    print(f"Valid moves: {env.board.get_valid_moves()}")
    print(f"Current player: {env.board.current_player}")
    
    # Get observation
    observation = env.board.get_observation()
    print(f"Observation shape: {observation.shape}")
    
    # Test network prediction
    with torch.no_grad():
        tensor_obs = torch.FloatTensor(observation).unsqueeze(0).to(device)
        policy_logits, value = model(tensor_obs)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()
    
    print(f"Network value prediction: {value:.4f}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {np.sum(policy):.4f}")
    
    # Check if policy makes sense for valid moves
    valid_moves_mask = env.get_valid_moves_mask()
    masked_policy = policy * valid_moves_mask
    
    print(f"Valid moves mask: {valid_moves_mask}")
    print(f"Masked policy: {masked_policy}")
    
    # Check top predicted actions
    top_actions = np.argsort(policy)[-5:][::-1]
    print("Top 5 predicted actions:")
    for i, action in enumerate(top_actions):
        if action == env.size * env.size:
            print(f"  {i+1}. Pass (action {action}): prob={policy[action]:.4f}, valid={valid_moves_mask[action]}")
        else:
            row, col = env.get_coords_from_action(action)
            is_valid = valid_moves_mask[action]
            print(f"  {i+1}. ({row},{col}) (action {action}): prob={policy[action]:.4f}, valid={is_valid}")

def test_different_positions():
    """Test network on different board positions."""
    print("\n" + "="*50)
    print("Testing network on different positions...")
    
    device = get_device(use_mps=True)
    model = AlphaZeroNetwork(game_size=6, device=device)
    
    checkpoint = torch.load('models/checkpoint_30.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test on a few different positions
    env = OthelloEnv(size=6)
    env.reset()
    
    # Make a few moves manually
    moves = [(4, 3), (2, 4), (1, 2)]  # Some valid initial moves
    
    for i, (row, col) in enumerate(moves):
        print(f"\n--- After move {i+1}: ({row}, {col}) ---")
        
        action = env.get_action_from_coords(row, col)
        env.step(action)
        
        env.render()
        print(f"Current player: {env.board.current_player}")
        print(f"Valid moves: {env.board.get_valid_moves()}")
        
        # Get network prediction
        observation = env.board.get_observation()
        with torch.no_grad():
            tensor_obs = torch.FloatTensor(observation).unsqueeze(0).to(device)
            policy_logits, value = model(tensor_obs)
            value = value.item()
        
        print(f"Network value: {value:.4f}")
        
        if env.board.is_done():
            winner = env.board.get_winner()
            print(f"Game over! Winner: {winner}")
            break

if __name__ == "__main__":
    test_network_output()
    test_different_positions()