#!/usr/bin/env python3

import torch
import numpy as np
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from utils.device import get_device
from fixed_mcts import FixedMCTS
from play import RandomPlayer, play_game

class FixedAlphaZeroPlayer:
    """AlphaZero player using the fixed MCTS implementation."""
    def __init__(self, model_path, num_simulations=800, c_puct=1.0, use_mps=True):
        self.device = get_device(use_mps=use_mps)
        print(f"Using device: {self.device} for Fixed AlphaZero player")
        
        self.model = AlphaZeroNetwork(game_size=6, device=self.device)
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.mcts = FixedMCTS(self.model, c_puct=c_puct, num_simulations=num_simulations)
    
    def get_action(self, env):
        """Get action from the Fixed AlphaZero MCTS."""
        state = env.board.get_observation()
        
        # Use MCTS to get action probabilities
        action_probs = self.mcts.search(state, env, temperature=0.1)
        
        # Choose the best action
        action = np.argmax(action_probs)
        return action

def test_fixed_mcts():
    """Test the fixed MCTS implementation."""
    print("Testing Fixed MCTS implementation...")
    
    # Test against random player
    fixed_player = FixedAlphaZeroPlayer("models/checkpoint_30.pt", num_simulations=200)
    random_player = RandomPlayer()
    
    wins = 0
    games = 10
    
    for i in range(games):
        print(f"Game {i+1}/{games}")
        
        # Alternate who plays first
        if i % 2 == 0:
            winner = play_game(fixed_player, random_player, render=False)
            fixed_color = -1  # Black
        else:
            winner = play_game(random_player, fixed_player, render=False)
            fixed_color = 1   # White
        
        if winner == fixed_color:
            wins += 1
            print(f"  Fixed AlphaZero won!")
        elif winner == 0:
            print(f"  Draw!")
        else:
            print(f"  Fixed AlphaZero lost!")
    
    win_rate = wins / games * 100
    print(f"\nFixed MCTS Results: {wins}/{games} wins ({win_rate:.1f}%)")
    
    return win_rate

def compare_original_vs_fixed():
    """Compare original MCTS vs fixed MCTS."""
    print("\n" + "="*50)
    print("Comparing Original vs Fixed MCTS...")
    
    from play import AlphaZeroPlayer
    
    original_player = AlphaZeroPlayer("models/checkpoint_30.pt", num_simulations=100)
    fixed_player = FixedAlphaZeroPlayer("models/checkpoint_30.pt", num_simulations=100)
    
    wins_original = 0
    wins_fixed = 0
    draws = 0
    games = 10
    
    for i in range(games):
        print(f"Game {i+1}/{games}")
        
        # Alternate who plays first
        if i % 2 == 0:
            winner = play_game(original_player, fixed_player, render=False)
            original_color = -1  # Black
        else:
            winner = play_game(fixed_player, original_player, render=False)
            original_color = 1   # White
        
        if winner == 0:
            draws += 1
            print("  Draw!")
        elif winner == original_color:
            wins_original += 1
            print("  Original MCTS won!")
        else:
            wins_fixed += 1
            print("  Fixed MCTS won!")
    
    print(f"\nHead-to-head results:")
    print(f"Original MCTS: {wins_original}/{games} wins ({wins_original/games*100:.1f}%)")
    print(f"Fixed MCTS: {wins_fixed}/{games} wins ({wins_fixed/games*100:.1f}%)")
    print(f"Draws: {draws}/{games} ({draws/games*100:.1f}%)")

if __name__ == "__main__":
    fixed_win_rate = test_fixed_mcts()
    
    if fixed_win_rate > 50:
        print("🎉 Fixed MCTS shows improvement!")
    else:
        print("😞 Fixed MCTS still needs work...")
    
    compare_original_vs_fixed()