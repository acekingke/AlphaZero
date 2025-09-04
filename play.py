import torch
import numpy as np
import time

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS
from utils.device import get_device

class OthelloPlayer:
    """Base class for Othello players."""
    def get_action(self, env):
        """Get the action to take in the current environment."""
        pass

class HumanPlayer(OthelloPlayer):
    """Human player that takes input from the console."""
    def get_action(self, env):
        """Get action from human input."""
        valid_moves = env.board.get_valid_moves()
        
        if not valid_moves:
            print("No valid moves available. Passing...")
            return env.size * env.size  # Pass action
        
        while True:
            try:
                print("Enter row and column (e.g., 3 4) or 'p' to pass:")
                user_input = input()
                
                if user_input.lower() == 'p':
                    return env.size * env.size  # Pass action
                
                row, col = map(int, user_input.split())
                
                if (row, col) in valid_moves:
                    return env.get_action_from_coords(row, col)
                else:
                    print("Invalid move! Try again.")
            except Exception as e:
                print(f"Error: {e}")
                print("Please enter valid coordinates or 'p' to pass.")

class AlphaZeroPlayer(OthelloPlayer):
    """Player that uses the trained AlphaZero model."""
    def __init__(self, model_path, num_simulations=800, c_puct=2.0, use_mps=True):
        # Get best available device
        self.device = get_device(use_mps=use_mps)
        print(f"Using device: {self.device} for AlphaZero player")
        
        self.model = AlphaZeroNetwork(game_size=6, device=self.device)
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device,weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.mcts = MCTS(self.model, c_puct=c_puct, num_simulations=num_simulations)
    
    def get_action(self, env):
        """Get action from the AlphaZero MCTS."""
        # Use canonical state for consistency with training
        canonical_state = env.board.get_canonical_state()
        # Convert to observation format (consistent with MCTS internal conversion)
        state = self.mcts._canonical_to_observation(canonical_state, env)
        
        # Use MCTS to get action probabilities
        action_probs = self.mcts.search(state, env, temperature=0.1)
        
        # Choose the best action
        action = np.argmax(action_probs)
        return action

class RandomPlayer(OthelloPlayer):
    """Player that selects random valid moves."""
    def get_action(self, env):
        """Get a random valid action."""
        valid_moves = env.board.get_valid_moves()
        
        if not valid_moves:
            return env.size * env.size  # Pass action
        
        # Select a random move from valid moves
        row, col = valid_moves[np.random.randint(len(valid_moves))]
        return env.get_action_from_coords(row, col)

def play_game(black_player, white_player, render=True):
    """Play a game between two players."""
    env = OthelloEnv(size=6)
    env.reset()
    
    if render:
        print("Initial board:")
        env.render()
    
    done = False
    while not done:
        # Get current player
        current_player = black_player if env.board.current_player == -1 else white_player
        
        if render:
            print(f"{'Black' if env.board.current_player == -1 else 'White'} player's turn")
        
        # Get action from current player
        action = current_player.get_action(env)
        
        # Take the action
        _, reward, done, _ = env.step(action)
        
        if render:
            if action == env.size * env.size:
                print("Player passed.")
            else:
                row, col = env.get_coords_from_action(action)
                print(f"Move: ({row}, {col})")
            env.render()
        
        if done:
            winner = env.board.get_winner()
            if winner == 0:
                result = "Draw!"
            elif winner == -1:
                result = "Black wins!"
            else:
                result = "White wins!"
            
            if render:
                print(f"Game over. {result}")
            
            return winner

def main():
    """Main function to play against AlphaZero."""
    import argparse
    parser = argparse.ArgumentParser(description='Play Othello against AlphaZero')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--player_color', type=str, default='black', choices=['black', 'white'], help='Player color (black or white)')
    parser.add_argument('--mcts_simulations', type=int, default=25, help='Number of MCTS simulations')
    args = parser.parse_args()
    
    # Create players
    alphazero_player = AlphaZeroPlayer(args.model, num_simulations=args.mcts_simulations)
    human_player = HumanPlayer()
    
    # Set players based on chosen color
    if args.player_color.lower() == 'black':
        black_player = human_player
        white_player = alphazero_player
        print("You are playing as Black (B)")
    else:
        black_player = alphazero_player
        white_player = human_player
        print("You are playing as White (W)")
    
    # Play the game
    play_game(black_player, white_player)

if __name__ == "__main__":
    main()