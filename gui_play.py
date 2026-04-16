import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import sys
import os
import argparse
import glob
import re

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS

class OthelloGUI(tk.Tk):
    def __init__(self, board_size=6, square_size=60, num_simulations=200, c_puct=1.0, model_path=None):
        super().__init__()
        self.title("Othello - You vs AlphaZero")
        self.board_size = board_size
        self.square_size = square_size
        self.canvas_size = board_size * square_size

        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='green')
        self.canvas.pack()

        self.status_bar = tk.Label(self, text="Your turn (Black)", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)

        self.env = OthelloEnv(self.board_size)
        self.ai_player_id = 1  # AI is White
        self.human_player_id = -1 # Human is Black

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.model = self.load_model(model_path)
        print(f"Using device: {self.device} for AI player")
        self.mcts = MCTS(self.model, c_puct=c_puct, num_simulations=num_simulations)

        self.canvas.bind("<Button-1>", self.handle_click)
        
        self.reset_game()

    def load_model(self, model_path=None):
        model = AlphaZeroNetwork(game_size=6, device=self.device)
        if model_path is None:
            model_path = self._find_latest_checkpoint()
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def _find_latest_checkpoint(self):
        """Prefer the highest-numbered frozen checkpoint_N.pt over best.pt.

        Rationale: best.pt gets overwritten every time Arena accepts a new
        model, so loading it mid-training is a race — the file may change
        between when we stat() it and when we torch.load() it. Numbered
        checkpoints are write-once. Pass --model explicitly to override.
        """
        pattern = './models/checkpoint_*.pt'
        files = glob.glob(pattern)
        if files:
            def extract_num(path):
                match = re.search(r'checkpoint_(\d+)\.pt$', path)
                return int(match.group(1)) if match else -1
            files = [f for f in files if extract_num(f) >= 0]
            if files:
                return max(files, key=extract_num)
        # Fallback to best.pt only if no numbered checkpoints exist.
        best_path = './models/best.pt'
        if os.path.exists(best_path):
            return best_path
        raise FileNotFoundError(
            f"No checkpoints found: neither {pattern} nor ./models/best.pt"
        )

    def reset_game(self):
        self.env.reset()
        # Clear MCTS stats between games so the new game doesn't reuse
        # opening analysis from the finished one.
        self.mcts.reset_tree()
        self.update_board()
        self.status_bar.config(text="Your turn (Black)")

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.board_size):
            for j in range(self.board_size):
                x1 = j * self.square_size
                y1 = i * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="green")
                
                piece = self.env.get_board().board[i][j]
                if piece != 0:
                    color = "white" if piece == 1 else "black"
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill=color, outline=color)

    def update_board(self):
        self.draw_board()
        self.draw_valid_moves()
        self.update_status()
        self.check_game_over()
        # If it's the human's turn and they have no valid moves, auto-pass to AI
        board = self.env.get_board()
        if not board.is_done() and board.current_player == self.human_player_id:
            valid_mask = self.env.get_valid_moves_mask()
            pass_action = self.board_size * self.board_size
            # When no placement moves are valid, the mask will allow only the pass action
            if valid_mask[pass_action] == 1 and valid_mask.sum() == 1:
                # Inform the player and perform the pass
                try:
                    messagebox.showinfo("Pass", "You have no valid moves. Passing to AI.")
                except Exception:
                    # In case the GUI is not fully available (tests), continue silently
                    pass
                self.env.step(pass_action)
                # Refresh board after passing
                self.draw_board()
                self.draw_valid_moves()
                self.update_status()
                self.check_game_over()
                if not self.env.get_board().is_done():
                    # Let the AI play after a short delay
                    self.after(100, self.ai_move)

    def draw_valid_moves(self):
        if self.env.get_board().current_player == self.human_player_id:
            valid_moves = self.env.get_board().get_valid_moves()
            for r, c in valid_moves:
                x1 = c * self.square_size
                y1 = r * self.square_size
                self.canvas.create_oval(x1 + self.square_size//2 - 5, y1 + self.square_size//2 - 5,
                                        x1 + self.square_size//2 + 5, y1 + self.square_size//2 + 5,
                                        fill="gray", outline="gray")

    def handle_click(self, event):
        if self.env.get_board().current_player != self.human_player_id or self.env.get_board().is_done():
            return

        col = event.x // self.square_size
        row = event.y // self.square_size
        
        action = self.env.get_action_from_coords(row, col)
        valid_moves_mask = self.env.get_valid_moves_mask()

        if valid_moves_mask[action] == 1:
            self.env.step(action)
            self.update_board()
            if not self.env.get_board().is_done():
                # env.step auto-resolves forced passes: if the opponent had
                # no legal moves, current_player is flipped back to us. Detect
                # that here and notify the player instead of silently handing
                # the turn back.
                if self.env.board.current_player == self.human_player_id:
                    try:
                        messagebox.showinfo(
                            "AI Passes",
                            "AI (White) has no valid moves. You play again.",
                        )
                    except Exception:
                        pass
                    # Turn is already ours; no AI move to schedule.
                else:
                    self.after(100, self.ai_move)  # Give a small delay for AI move
        else:
            messagebox.showwarning("Invalid Move", "This is not a valid move.")

    def ai_move(self):
        if self.env.get_board().current_player == self.ai_player_id and not self.env.get_board().is_done():
            self.status_bar.config(text="AI (White) is thinking...")
            self.update()

            # Check if AI has any valid moves
            valid_moves_mask = self.env.get_valid_moves_mask()
            pass_action = self.board_size * self.board_size
            
            # If AI has no valid placement moves, it must pass
            if valid_moves_mask[pass_action] == 1 and valid_moves_mask.sum() == 1:
                try:
                    messagebox.showinfo("Pass", "AI (White) has no valid moves. Passing to you.")
                except Exception:
                    # In case the GUI is not fully available (tests), continue silently
                    pass
                action = pass_action
            else:
                # Get canonical state for consistency with training
                canonical_state = self.env.board.get_canonical_state()
                # Convert to observation format (consistent with MCTS)
                state = self.mcts.canonical_to_observation(canonical_state, self.env)
                
                # Use MCTS to get action probabilities
                action_probs = self.mcts.search(state, self.env, temperature=0)
                
                # Choose the best action
                action = np.argmax(action_probs)
                
                # Double-check that the chosen action is valid
                if self.env.get_valid_moves_mask()[action] == 0:
                    # Fallback: find first valid action
                    valid_actions = np.where(valid_moves_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
                    else:
                        action = pass_action

            self.env.step(action)
            self.update_board()
            # NOTE: update_board() already calls check_game_over() internally.
            # A second call here would pop the Game Over dialog twice.

            # Detect human forced-pass: env.step auto-flips back to AI if the
            # human has no legal moves after AI's move. Notify and chain
            # another AI move.
            if not self.env.get_board().is_done():
                if self.env.board.current_player == self.ai_player_id:
                    try:
                        messagebox.showinfo(
                            "You Pass",
                            "You have no valid moves. AI plays again.",
                        )
                    except Exception:
                        pass
                    self.after(100, self.ai_move)

    def update_status(self):
        board = self.env.get_board()
        white_count = np.sum(board.board == 1)
        black_count = np.sum(board.board == -1)
        
        status_text = f"White (AI): {white_count} | Black (You): {black_count} | "
        if board.is_done():
            winner = board.get_winner()
            if winner == 1:
                status_text += "White (AI) wins!"
            elif winner == -1:
                status_text += "Black (You) win!"
            else:
                status_text += "It's a draw!"
        else:
            current_player_name = "White (AI)" if board.current_player == 1 else "Black (You)"
            status_text += f"Turn: {current_player_name}"
        
        self.status_bar.config(text=status_text)

    def check_game_over(self):
        if self.env.get_board().is_done():
            winner = self.env.get_board().get_winner()
            if winner == self.human_player_id:
                message = "Congratulations, you win!"
            elif winner == self.ai_player_id:
                message = "AI wins. Better luck next time!"
            else:
                message = "It's a draw!"
            
            if messagebox.askyesno("Game Over", message + "\nDo you want to play again?"):
                self.reset_game()
            else:
                self.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Othello GUI Game - Play against AlphaZero')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint (default: latest)')
    parser.add_argument('--mcts_simulations', type=int, default=200, help='Number of MCTS simulations per move (default: 200, matches play.py)')
    parser.add_argument('--c_puct', type=float, default=1.0, help='c_puct value for MCTS (default: 1.0, matches v20 training)')
    parser.add_argument('--board_size', type=int, default=6, help='Board size (default: 6)')
    parser.add_argument('--square_size', type=int, default=60, help='Square size in pixels (default: 60)')
    
    args = parser.parse_args()
    
    print(f"Starting Othello GUI with:")
    print(f"  MCTS simulations: {args.mcts_simulations}")
    print(f"  c_puct: {args.c_puct}")
    print(f"  Board size: {args.board_size}x{args.board_size}")
    
    app = OthelloGUI(
        board_size=args.board_size,
        square_size=args.square_size,
        num_simulations=args.mcts_simulations,
        c_puct=args.c_puct,
        model_path=args.model,
    )
    app.mainloop()
