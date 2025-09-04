import argparse
import os
import torch

from train import AlphaZeroTrainer
from play import main as play_main
from evaluate import evaluate_model, compare_models
from utils.device import check_mps_availability

def main():
    parser = argparse.ArgumentParser(description='AlphaZero for Othello')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Check device command
    check_parser = subparsers.add_parser('check_device', help='Check available devices (CUDA, MPS, CPU)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the AlphaZero model')
    # Support both --num_iterations (code default) and --iterations (README style) for convenience
    train_parser.add_argument('--num_iterations', '--iterations', dest='num_iterations', type=int, default=100, help='Number of training iterations')
    train_parser.add_argument('--self_play_games', type=int, default=1000, help='Number of self-play games per iteration')
    train_parser.add_argument('--mcts_simulations', type=int, default=100, help='Number of MCTS simulations per move (recommended: 50-200 for training)')
    train_parser.add_argument('--temperature', type=float, default=0.8, help='Temperature parameter for MCTS action selection (higher = more exploration)')
    train_parser.add_argument('--c_puct', type=float, default=1.0, help='c_puct value for MCTS (exploration constant)')
    train_parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    train_parser.add_argument('--l2_regularization', type=float, default=1e-4, help='L2 regularization coefficient (c in AlphaZero paper)')
    train_parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    train_parser.add_argument('--use_mps', action='store_true', help='Use MPS acceleration on macOS if available')
    train_parser.add_argument('--use_cpu', action='store_true', help='Force use CPU even if GPU is available')
    train_parser.add_argument('--use_multiprocessing', action='store_true', help='Use multiprocessing for self-play')
    train_parser.add_argument('--mp_num_workers', type=int, default=None, help='Number of worker processes for multiprocessing')
    train_parser.add_argument('--mp_games_per_worker', type=int, default=1, help='Number of self-play games each worker runs per task')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against the trained model')
    play_parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    play_parser.add_argument('--player_color', type=str, default='black', choices=['black', 'white'], help='Player color (black or white)')
    play_parser.add_argument('--mcts_simulations', type=int, default=200, help='Number of MCTS simulations (recommended: 200 for better gameplay)')
    play_parser.add_argument('--use_mps', action='store_true', help='Use MPS acceleration on macOS if available')
    play_parser.add_argument('--use_cpu', action='store_true', help='Force use CPU even if GPU is available')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model against an opponent')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    eval_parser.add_argument('--opponent', type=str, default='random', choices=['random'], help='Opponent type')
    eval_parser.add_argument('--num_games', type=int, default=50, help='Number of games to play')
    eval_parser.add_argument('--mcts_simulations', type=int, default=100, help='Number of MCTS simulations (default: 100 for evaluation)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('--model1', type=str, required=True, help='Path to the first model checkpoint')
    compare_parser.add_argument('--model2', type=str, required=True, help='Path to the second model checkpoint')
    compare_parser.add_argument('--num_games', type=int, default=50, help='Number of games to play')
    compare_parser.add_argument('--mcts_simulations', type=int, default=100, help='Number of MCTS simulations (default: 100 for model comparison)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Check if this is a macOS system and MPS is available
        if args.use_mps:
            check_mps_availability()
        
        # Create trainer and train the model
        trainer = AlphaZeroTrainer(
            num_iterations=args.num_iterations,
            num_self_play_games=args.self_play_games,
            num_mcts_simulations=args.mcts_simulations,
            temperature=args.temperature,
            c_puct=args.c_puct,
            batch_size=args.batch_size,
            l2_regularization=args.l2_regularization,
            use_mps=args.use_mps,
            use_cuda=not args.use_cpu,
            use_multiprocessing=args.use_multiprocessing,
            mp_num_workers=args.mp_num_workers,
            mp_games_per_worker=args.mp_games_per_worker
        )
        trainer.train(resume_from=args.resume)
    
    elif args.command == 'play':
        # Call play function from play.py
        import sys
        sys.argv = [
            sys.argv[0],
            '--model', args.model,
            '--player_color', args.player_color,
            '--mcts_simulations', str(args.mcts_simulations)
        ]
        play_main()
    
    elif args.command == 'evaluate':
        # Evaluate the model
        evaluate_model(
            model_path=args.model,
            num_games=args.num_games,
            opponent=args.opponent,
            mcts_simulations=args.mcts_simulations
        )
    
    elif args.command == 'compare':
        # Compare two models
        compare_models(
            model_path1=args.model1,
            model_path2=args.model2,
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations
        )
    
    elif args.command == 'check_device':
        # Run the device check
        check_mps_availability()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()