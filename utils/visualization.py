import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def plot_board(board, policy=None, save_path=None):
    """
    Visualize the Othello board and optionally the policy.
    
    Args:
        board: The OthelloBoard object
        policy: Optional policy vector to visualize
        save_path: Path to save the figure
    """
    board_state = board.get_state()
    size = board.size
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw board squares
    for i in range(size):
        for j in range(size):
            ax.add_patch(Rectangle((j, size - i - 1), 1, 1, 
                                   edgecolor='black', facecolor='darkgreen', alpha=0.7))
    
    # Draw the pieces
    for i in range(size):
        for j in range(size):
            if board_state[i, j] == 1:  # White
                ax.add_patch(Circle((j + 0.5, size - i - 0.5), 0.4, 
                                    edgecolor='black', facecolor='white'))
            elif board_state[i, j] == -1:  # Black
                ax.add_patch(Circle((j + 0.5, size - i - 0.5), 0.4, 
                                    edgecolor='black', facecolor='black'))
    
    # Draw policy probabilities if provided
    if policy is not None:
        for i in range(size):
            for j in range(size):
                action = i * size + j
                if action < len(policy) and policy[action] > 0.01:  # Only show significant probabilities
                    plt.text(j + 0.5, size - i - 0.5, f'{policy[action]:.2f}', 
                             ha='center', va='center', color='red',
                             fontsize=8, fontweight='bold')
    
    # Draw grid lines
    for i in range(size + 1):
        ax.plot([0, size], [i, i], 'k-')
        ax.plot([i, i], [0, size], 'k-')
    
    # Add labels
    for i in range(size):
        plt.text(i + 0.5, -0.3, chr(97 + i), ha='center', va='center', fontsize=12)
        plt.text(-0.3, i + 0.5, str(size - i), ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-0.5, size + 0.5)
    ax.set_ylim(-0.5, size + 0.5)
    plt.axis('off')
    
    # Current player info
    player_str = 'Black' if board.current_player == -1 else 'White'
    plt.title(f'Current Player: {player_str}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_game_history(game_history, interval=5, save_dir='game_visualization'):
    """
    Visualize a game history.
    
    Args:
        game_history: List of (board, action, policy) tuples
        interval: Save every 'interval' moves
        save_dir: Directory to save the figures
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (board, action, policy) in enumerate(game_history):
        if i % interval == 0 or i == len(game_history) - 1:
            save_path = f"{save_dir}/move_{i:03d}.png"
            plot_board(board, policy, save_path)
            print(f"Saved {save_path}")
            
def create_game_gif(image_dir, output_file='game.gif', fps=2):
    """
    Create a GIF from a series of game board images.
    
    Args:
        image_dir: Directory containing the board images
        output_file: Output GIF file
        fps: Frames per second in the GIF
    """
    import glob
    from PIL import Image
    
    # Get all PNG images and sort them
    images = sorted(glob.glob(f"{image_dir}/*.png"))
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    # Create a GIF
    frames = [Image.open(image) for image in images]
    frames[0].save(
        output_file,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=int(1000/fps),  # milliseconds
        loop=0
    )
    
    print(f"GIF created: {output_file}")