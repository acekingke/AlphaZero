#!/usr/bin/env python3

import torch
import numpy as np
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from utils.device import get_device

def compare_checkpoint_predictions():
    """Compare predictions from different checkpoints on the same position."""
    print("Comparing checkpoint predictions...")
    
    device = get_device(use_mps=True)
    
    # Test environment setup
    env = OthelloEnv(size=6)
    env.reset()
    observation = env.board.get_observation()
    
    print("Initial board:")
    env.render()
    print(f"Valid moves: {env.board.get_valid_moves()}")
    
    checkpoints = [
        'models/checkpoint_10.pt',
        'models/checkpoint_20.pt', 
        'models/checkpoint_30.pt'
    ]
    
    for checkpoint_path in checkpoints:
        print(f"\n--- {checkpoint_path} ---")
        
        model = AlphaZeroNetwork(game_size=6, device=device)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                tensor_obs = torch.FloatTensor(observation).unsqueeze(0).to(device)
                policy_logits, value = model(tensor_obs)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                value = value.item()
            
            print(f"Value prediction: {value:.4f}")
            
            # Show top moves
            valid_moves_mask = env.get_valid_moves_mask()
            masked_policy = policy * valid_moves_mask
            
            top_actions = np.argsort(masked_policy)[-3:][::-1]
            print("Top 3 valid moves:")
            for i, action in enumerate(top_actions):
                if masked_policy[action] > 0:
                    if action == env.size * env.size:
                        print(f"  {i+1}. Pass: {masked_policy[action]:.4f}")
                    else:
                        row, col = env.get_coords_from_action(action)
                        print(f"  {i+1}. ({row},{col}): {masked_policy[action]:.4f}")
                        
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")

def test_random_vs_random():
    """Test random vs random to establish baseline."""
    print("\n" + "="*50)
    print("Testing Random vs Random baseline...")
    
    from play import RandomPlayer, play_game
    
    random1 = RandomPlayer()
    random2 = RandomPlayer()
    
    wins1 = 0
    wins2 = 0
    draws = 0
    
    games = 20
    
    for i in range(games):
        if i % 2 == 0:
            winner = play_game(random1, random2, render=False)
            player1_color = -1  # Black
        else:
            winner = play_game(random2, random1, render=False)
            player1_color = 1   # White
        
        if winner == 0:
            draws += 1
        elif winner == player1_color:
            wins1 += 1
        else:
            wins2 += 1
    
    print(f"Random1 wins: {wins1} ({wins1/games*100:.1f}%)")
    print(f"Random2 wins: {wins2} ({wins2/games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/games*100:.1f}%)")
    print("Expected: roughly 50-50 distribution")

def verify_game_symmetry():
    """Check if the game has proper symmetry in scoring."""
    print("\n" + "="*50)
    print("Verifying game symmetry...")
    
    from play import RandomPlayer, play_game
    
    random_player = RandomPlayer()
    
    # Play many games and check distribution
    outcomes = {'black_wins': 0, 'white_wins': 0, 'draws': 0}
    games = 50
    
    for i in range(games):
        winner = play_game(random_player, random_player, render=False)
        if winner == -1:
            outcomes['black_wins'] += 1
        elif winner == 1:
            outcomes['white_wins'] += 1
        else:
            outcomes['draws'] += 1
    
    print(f"Results over {games} random games:")
    print(f"Black wins: {outcomes['black_wins']} ({outcomes['black_wins']/games*100:.1f}%)")
    print(f"White wins: {outcomes['white_wins']} ({outcomes['white_wins']/games*100:.1f}%)")
    print(f"Draws: {outcomes['draws']} ({outcomes['draws']/games*100:.1f}%)")
    
    # Check if there's a significant bias
    if abs(outcomes['black_wins'] - outcomes['white_wins']) > games * 0.2:
        print("⚠️  WARNING: Significant color bias detected!")
    else:
        print("✓ Game appears balanced between colors")

if __name__ == "__main__":
    compare_checkpoint_predictions()
    test_random_vs_random()
    verify_game_symmetry()