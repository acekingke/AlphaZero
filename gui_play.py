import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import sys
import os

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS

class OthelloGUI(tk.Tk):
    def __init__(self, board_size=8, square_size=60, num_simulations=400, c_puct=1.0):
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
        self.model = self.load_model()
        print(f"Using device: {self.device} for AI player")
        self.mcts = MCTS(self.model, c_puct=c_puct, num_simulations=num_simulations)

        self.canvas.bind("<Button-1>", self.handle_click)
        
        self.reset_game()

    def load_model(self):
        model = AlphaZeroNetwork(game_size=self.board_size, device=self.device)
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir('models') if f.startswith('checkpoint_') and f.endswith('.pt')]
        print(checkpoints)
        if not checkpoints:
            messagebox.showerror("Error", "No model checkpoint found in 'models/' directory.")
            self.destroy()
            return None
        
        latest_checkpoint_file = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)[0]
        #latest_checkpoint_file = "checkpoint_4.pt"
        print(f"Loading model from {latest_checkpoint_file}")
        checkpoint = torch.load(os.path.join('models', latest_checkpoint_file), map_location=self.device, weights_only=True)

        # The model might be saved directly or inside a dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def reset_game(self):
        self.env.reset()
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
                self.after(100, self.ai_move) # Give a small delay for AI move
        else:
            messagebox.showwarning("Invalid Move", "This is not a valid move.")

    def ai_move(self):
        if self.env.get_board().current_player == self.ai_player_id and not self.env.get_board().is_done():
            self.status_bar.config(text="AI (White) is thinking...")
            self.update()

            # Get observation for the model
            state = self.env.board.get_observation()
            
            # Use MCTS to get action probabilities
            action_probs = self.mcts.search(state, self.env, temperature=0)
            
            # Choose the best action
            action = np.argmax(action_probs)

            # If AI has no valid moves, it must pass
            if self.env.get_valid_moves_mask()[action] == 0:
                action = self.env.get_board().get_action_space_size() - 1 # Pass action

            self.env.step(action)
            self.update_board()

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
    app = OthelloGUI()
    app.mainloop()
