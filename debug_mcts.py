#!/usr/bin/env python3

import torch
import numpy as np
from env.othello import OthelloEnv
from play import AlphaZeroPlayer, RandomPlayer, play_game

def manual_game_test():
    """Manually play a game and see decision making."""
    print("Manual game test...")
    
    env = OthelloEnv(size=6)
    env.reset()
    
    alphazero_player = AlphaZeroPlayer("models/checkpoint_30.pt", num_simulations=10)  # Reduced for speed
    
    print("Initial board:")
    env.render()
    print(f"Valid moves: {env.board.get_valid_moves()}")
    
    # Let AlphaZero make first move
    action = alphazero_player.get_action(env)
    print(f"AlphaZero chose action: {action}")
    
    if action == env.size * env.size:
        print("AlphaZero chose to pass")
    else:
        row, col = env.get_coords_from_action(action)
        print(f"AlphaZero chose move: ({row}, {col})")
    
    # Take the action
    obs, reward, done, info = env.step(action)
    
    print("After AlphaZero's move:")
    env.render()
    print(f"Reward: {reward}, Done: {done}")
    print(f"Current player: {env.board.current_player}")
    print(f"Valid moves: {env.board.get_valid_moves()}")
    
    # If game not done, let random player move
    if not done:
        # Random player move
        valid_moves = env.board.get_valid_moves()
        if valid_moves:
            move = valid_moves[np.random.randint(len(valid_moves))]
            action = env.get_action_from_coords(move[0], move[1])
            print(f"Random player chose: {move}")
            
            obs, reward, done, info = env.step(action)
            print("After Random player's move:")
            env.render()
            print(f"Reward: {reward}, Done: {done}")

def simple_evaluation_test():
    """Run a very simple evaluation with detailed output."""
    print("\nSimple evaluation test...")
    
    alphazero_player = AlphaZeroPlayer("models/checkpoint_30.pt", num_simulations=50)
    random_player = RandomPlayer()
    
    wins = 0
    games = 5
    
    for i in range(games):
        print(f"\n--- Game {i+1} ---")
        
        # Alternate who plays first
        if i % 2 == 0:
            winner = play_game(alphazero_player, random_player, render=False)
            alphazero_color = -1  # Black
        else:
            winner = play_game(random_player, alphazero_player, render=False)
            alphazero_color = 1   # White
        
        print(f"Winner: {winner}, AlphaZero was: {'Black' if alphazero_color == -1 else 'White'}")
        
        if winner == alphazero_color:
            wins += 1
            print("AlphaZero won!")
        elif winner == 0:
            print("Draw!")
        else:
            print("AlphaZero lost!")
    
    print(f"\nFinal results: {wins}/{games} wins ({wins/games*100:.1f}%)")

def test_mcts_consistency():
    """Test if MCTS gives consistent results."""
    print("\nTesting MCTS consistency...")
    
    env = OthelloEnv(size=6)
    env.reset()
    
    alphazero_player = AlphaZeroPlayer("models/checkpoint_30.pt", num_simulations=100)
    
    print("Running same position multiple times:")
    for i in range(3):
        env_copy = env.__class__(size=6)
        env_copy.board = env.board.__class__(size=6)
        env_copy.board.board = env.board.board.copy()
        env_copy.board.current_player = env.board.current_player
        env_copy.board.done = env.board.done
        env_copy.board.winner = env.board.winner
        
        action = alphazero_player.get_action(env_copy)
        if action == env.size * env.size:
            print(f"  Run {i+1}: Pass")
        else:
            row, col = env.get_coords_from_action(action)
            print(f"  Run {i+1}: ({row}, {col})")

if __name__ == "__main__":
    manual_game_test()
    simple_evaluation_test() 
    test_mcts_consistency()