#!/usr/bin/env python3

import torch
import numpy as np
from env.othello import OthelloEnv
from play import AlphaZeroPlayer, RandomPlayer, play_game

def debug_single_game():
    """Debug a single game to see what's happening."""
    print("Starting debug game...")
    
    # Create players
    alphazero_player = AlphaZeroPlayer("models/checkpoint_34.pt", num_simulations=50)
    random_player = RandomPlayer()
    
    # AlphaZero plays as black, random as white
    print("AlphaZero (Black) vs Random (White)")
    winner = play_game(alphazero_player, random_player, render=True)
    
    print(f"Winner: {winner}")
    if winner == -1:
        print("Black (AlphaZero) wins!")
    elif winner == 1:
        print("White (Random) wins!")
    else:
        print("Draw!")

def test_mcts_action_selection():
    """Test if MCTS is selecting reasonable actions."""
    print("\nTesting MCTS action selection...")
    
    env = OthelloEnv(size=6)
    env.reset()
    
    alphazero_player = AlphaZeroPlayer("models/checkpoint_34.pt", num_simulations=50)
    
    print("Initial board:")
    env.render()
    print(f"Valid moves: {env.board.get_valid_moves()}")
    
    # Get action from AlphaZero
    action = alphazero_player.get_action(env)
    print(f"AlphaZero selected action: {action}")
    
    if action == env.size * env.size:
        print("AlphaZero chose to pass")
    else:
        row, col = env.get_coords_from_action(action)
        print(f"AlphaZero chose move: ({row}, {col})")
        
        # Check if it's a valid move
        valid_moves = env.board.get_valid_moves()
        if (row, col) in valid_moves:
            print("✓ Move is valid")
        else:
            print("✗ Move is INVALID!")
            print(f"Valid moves were: {valid_moves}")

if __name__ == "__main__":
    test_mcts_action_selection()
    print("\n" + "="*50 + "\n")
    debug_single_game()