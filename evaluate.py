import torch
import numpy as np
from tqdm import tqdm

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from play import AlphaZeroPlayer, RandomPlayer, play_game

def evaluate_model(model_path, num_games=100, opponent='random', mcts_simulations=100):
    """
    Evaluate the AlphaZero model against an opponent.
    
    Args:
        model_path: Path to the model checkpoint
        num_games: Number of games to play
        opponent: Type of opponent ('random')
        mcts_simulations: Number of MCTS simulations
    
    Returns:
        win_rate: Percentage of games won
        draw_rate: Percentage of games drawn
        loss_rate: Percentage of games lost
    """
    # Create players
    alphazero_player = AlphaZeroPlayer(model_path, num_simulations=mcts_simulations)
    
    if opponent == 'random':
        opponent_player = RandomPlayer()
    else:
        raise ValueError(f"Unknown opponent type: {opponent}")
    
    # Statistics
    wins = 0
    draws = 0
    losses = 0
    
    # Play games
    for i in tqdm(range(num_games), desc=f"Evaluating against {opponent}"):
        # AlphaZero plays as both black and white
        if i % 2 == 0:
            black_player = alphazero_player
            white_player = opponent_player
            alphazero_color = -1  # Black
        else:
            black_player = opponent_player
            white_player = alphazero_player
            alphazero_color = 1  # White
        
        # Play a game
        winner = play_game(black_player, white_player, render=False)
        
        # Update statistics
        if winner == 0:
            draws += 1
        elif winner == alphazero_color:
            wins += 1
        else:
            losses += 1
    
    # Calculate rates
    win_rate = wins / num_games * 100
    draw_rate = draws / num_games * 100
    loss_rate = losses / num_games * 100
    
    print(f"Evaluation results against {opponent} ({num_games} games):")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Draw rate: {draw_rate:.1f}%")
    print(f"Loss rate: {loss_rate:.1f}%")
    
    return win_rate, draw_rate, loss_rate

def compare_models(model_path1, model_path2, num_games=50, mcts_simulations=100):
    """
    Compare two AlphaZero models by playing games against each other.
    
    Args:
        model_path1: Path to the first model checkpoint
        model_path2: Path to the second model checkpoint
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations
    
    Returns:
        win_rate1: Percentage of games won by model 1
        draw_rate: Percentage of games drawn
        win_rate2: Percentage of games won by model 2
    """
    # Create players
    model1_player = AlphaZeroPlayer(model_path1, num_simulations=mcts_simulations)
    model2_player = AlphaZeroPlayer(model_path2, num_simulations=mcts_simulations)
    
    # Statistics
    wins1 = 0
    wins2 = 0
    draws = 0
    
    # Play games
    for i in tqdm(range(num_games), desc="Comparing models"):
        # Alternate which model plays black
        if i % 2 == 0:
            black_player = model1_player
            white_player = model2_player
            model1_color = -1  # Black
        else:
            black_player = model2_player
            white_player = model1_player
            model1_color = 1  # White
        
        # Play a game
        winner = play_game(black_player, white_player, render=False)
        
        # Update statistics
        if winner == 0:
            draws += 1
        elif winner == model1_color:
            wins1 += 1
        else:
            wins2 += 1
    
    # Calculate rates
    win_rate1 = wins1 / num_games * 100
    draw_rate = draws / num_games * 100
    win_rate2 = wins2 / num_games * 100
    
    print(f"Comparison results ({num_games} games):")
    print(f"Model 1 win rate: {win_rate1:.1f}%")
    print(f"Draw rate: {draw_rate:.1f}%")
    print(f"Model 2 win rate: {win_rate2:.1f}%")
    
    return win_rate1, draw_rate, win_rate2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero Othello model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--compare_with', type=str, help='Path to another model checkpoint to compare with')
    parser.add_argument('--num_games', type=int, default=50, help='Number of games to play')
    parser.add_argument('--mcts_simulations', type=int, default=100, help='Number of MCTS simulations')
    args = parser.parse_args()
    
    if args.compare_with:
        compare_models(args.model, args.compare_with, args.num_games, args.mcts_simulations)
    else:
        evaluate_model(args.model, args.num_games, 'random', args.mcts_simulations)