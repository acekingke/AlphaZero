import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import time
import matplotlib.pyplot as plt

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS
from utils.device import get_device
from utils.training_logger import TrainingLogger

class AlphaZeroTrainer:
    """Trainer for AlphaZero."""
    def __init__(self, game_size=8, num_iterations=50, num_self_play_games=100, 
                 num_mcts_simulations=800, temperature=1.0, c_puct=1.0,
                 batch_size=128, num_epochs=20, lr=0.001, checkpoint_path='./models/checkpoint',
                 use_mps=True, use_cuda=True, log_dir='./logs'):
        self.game_size = game_size
        self.num_iterations = num_iterations
        self.num_self_play_games = num_self_play_games
        self.num_mcts_simulations = num_mcts_simulations
        self.temperature = temperature
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        self.log_dir = log_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Get the best available device (CUDA, MPS, or CPU)
        self.device = get_device(use_mps=use_mps, use_cuda=use_cuda)
        print(f"训练将使用: {self.device}")
        
        # Initialize neural network
        self.model = AlphaZeroNetwork(game_size, device=self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Experience buffer
        self.buffer = deque(maxlen=100000)
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        
        # Initialize training logger
        self.logger = TrainingLogger(log_dir=log_dir, model_name='alphazero')
    
    def self_play(self):
        """
        Perform self-play to generate training data.
        
        Returns:
            List of (state, policy, value) tuples from the self-play games
        """
        examples = []
        
        for _ in tqdm(range(self.num_self_play_games), desc="Self-Play Games"):
            env = OthelloEnv(size=self.game_size)
            env.reset()
            
            mcts = MCTS(self.model, c_puct=self.c_puct, num_simulations=self.num_mcts_simulations)
            game_history = []
            
            # Track the game step to decay temperature
            step = 0
            
            while not env.board.is_done():
                # Decay temperature over time to increase exploitation
                if step < 10:
                    current_temp = self.temperature
                elif step < 20:
                    current_temp = 0.5
                else:
                    current_temp = 0.25
                
                # Add Dirichlet noise only at root node for exploration
                state = env.board.get_observation()
                action_probs = mcts.search(state, env, current_temp, add_noise=(step==0))
                
                # Save state and probabilities
                game_history.append((state, action_probs))
                
                # Choose action based on the MCTS probabilities
                action = np.random.choice(len(action_probs), p=action_probs)
                _, reward, done, _ = env.step(action)
                
                step += 1
                
                # Break if game is over
                if done:
                    break
            
            # Get the final result of the game
            winner = env.board.get_winner()
            
            # Update examples with the outcome
            for state, policy in game_history:
                if winner == 0:  # Draw
                    value = 0
                else:
                    # Value is from the perspective of the player who made the move
                    current_player = 1 if np.sum(state[0]) == np.sum(state[1]) else -1
                    value = 1 if winner == current_player else -1
                
                examples.append((state, policy, value))
        
        return examples
    
    def train_network(self, examples):
        """
        Train the neural network on the provided examples.
        
        Args:
            examples: List of (state, policy, value) tuples
        """
        # Add examples to buffer
        self.buffer.extend(examples)
        
        # If buffer doesn't have enough samples, don't train
        if len(self.buffer) < self.batch_size:
            print(f"Buffer has only {len(self.buffer)} samples, need at least {self.batch_size}. Skipping training.")
            return 0.0, 0.0, 0.0  # Return zero losses when skipping training
        
        # Extract data
        mini_batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size * self.num_epochs))
        
        # Split data
        states, policies, values = zip(*mini_batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        policies = torch.FloatTensor(np.array(policies)).to(self.device)
        values = torch.FloatTensor(np.array(values).reshape(-1, 1)).to(self.device)
        
        # Training loop
        self.model.train()
        
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        
        # Split into batches
        for i in range(0, len(mini_batch), self.batch_size):
            batch_states = states[i:i+self.batch_size]
            batch_policies = policies[i:i+self.batch_size]
            batch_values = values[i:i+self.batch_size]
            
            # Forward pass
            policy_logits, value_pred = self.model(batch_states)
            
            # Calculate loss
            policy_loss = -torch.sum(batch_policies * torch.log_softmax(policy_logits, dim=1)) / batch_states.size(0)
            value_loss = nn.MSELoss()(value_pred, batch_values)
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Calculate average loss
        num_batches = (len(mini_batch) + self.batch_size - 1) // self.batch_size
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        # Record metrics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.total_losses.append(avg_total_loss)
        
        return avg_policy_loss, avg_value_loss, avg_total_loss
    
    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'total_losses': self.total_losses
        }, f"{self.checkpoint_path}_{iteration}.pt")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.total_losses = checkpoint.get('total_losses', [])
        
        # 将已有的loss记录加载到logger中
        for i, (p_loss, v_loss, t_loss) in enumerate(zip(
            self.policy_losses, self.value_losses, self.total_losses)):
            self.logger.log_iteration(
                iteration=i+1,
                policy_loss=p_loss,
                value_loss=v_loss,
                total_loss=t_loss,
                examples_count=0,
                elapsed_time=0
            )
            
        return checkpoint.get('iteration', 0)
    
    def plot_metrics(self):
        """Plot training metrics."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.policy_losses)
        plt.title('Policy Loss')
        plt.xlabel('Training Iterations')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.value_losses)
        plt.title('Value Loss')
        plt.xlabel('Training Iterations')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.total_losses)
        plt.title('Total Loss')
        plt.xlabel('Training Iterations')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./training_metrics.png')
        plt.close()
    
    def train(self, resume_from=None):
        """Main training loop for AlphaZero."""
        start_iteration = 0
        
        # Load from checkpoint if specified
        if resume_from:
            start_iteration = self.load_checkpoint(resume_from)
            print(f"Resuming training from iteration {start_iteration+1}")
        
        # Training loop
        for iteration in range(start_iteration, self.num_iterations):
            iteration_start_time = time.time()
            print(f"Starting iteration {iteration+1}/{self.num_iterations}")
            
            # Self-play
            print("Self-play phase...")
            try:
                examples = self.self_play()
                examples_count = len(examples)
                print(f"Generated {examples_count} examples from self-play")
                
                # Train neural network
                print("Training phase...")
                policy_loss, value_loss, total_loss = self.train_network(examples)
            except Exception as e:
                print(f"Error during iteration {iteration+1}: {e}")
                import traceback
                traceback.print_exc()
                print("Skipping to next iteration...")
                continue
            
            # Calculate elapsed time
            elapsed_time = time.time() - iteration_start_time
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Total Loss: {total_loss:.4f}")
            print(f"Iteration completed in {elapsed_time:.2f} seconds")
            
            # Log metrics
            self.logger.log_iteration(
                iteration=iteration+1,
                policy_loss=policy_loss,
                value_loss=value_loss,
                total_loss=total_loss,
                examples_count=examples_count,
                elapsed_time=elapsed_time
            )
            
            # Save checkpoint
            self.save_checkpoint(iteration)
            
            # Plot metrics
            if (iteration + 1) % 5 == 0 or iteration == self.num_iterations - 1:
                self.plot_metrics()
                metrics_image = self.logger.plot_metrics(
                    save_path=os.path.join(self.log_dir, f'metrics_iter_{iteration+1}.png')
                )
                print(f"Training metrics plotted to: {metrics_image}")
        
        print("Training completed!")
        self.plot_metrics()
        self.logger.plot_metrics(save_path='./training_metrics_final.png')