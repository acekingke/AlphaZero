import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

class TrainingLogger:
    """Logger tool for recording and visualizing the training process"""
    
    def __init__(self, log_dir='./logs', model_name='alphazero'):
        """
        Initialize training logger
        
        Parameters:
            log_dir (str): Directory to save logs
            model_name (str): Model name, used to generate log filenames
        """
        self.log_dir = log_dir
        self.model_name = model_name
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成唯一的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{model_name}_{timestamp}.csv')
        
        # 初始化日志数据
        self.log_data = []
        self.iterations = []
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.timestamps = []
        
        # 创建CSV文件并写入表头
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'timestamp', 'policy_loss', 'value_loss', 'total_loss', 
                            'examples', 'elapsed_time'])
    
    def log_iteration(self, iteration, policy_loss, value_loss, total_loss, examples_count, elapsed_time=None):
        """
        Record metrics for a single training iteration
        
        Parameters:
            iteration (int): Current iteration number
            policy_loss (float): Policy loss value
            value_loss (float): Value loss value
            total_loss (float): Total loss value
            examples_count (int): Number of examples used in this iteration
            elapsed_time (float): Training time for this iteration (seconds)
        """
        timestamp = time.time()
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存数据到内存
        self.iterations.append(iteration)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.total_losses.append(total_loss)
        self.timestamps.append(timestamp)
        
        # 记录到CSV文件
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, timestamp, policy_loss, value_loss, 
                            total_loss, examples_count, elapsed_time])
    
    def plot_metrics(self, save_path=None, show_plot=False):
        """
        Plot training metrics
        
        Parameters:
            save_path (str): Path to save the chart, if None use default path
            show_plot (bool): Whether to display the chart
        """
        if len(self.iterations) == 0:
            print("No training data available for plotting")
            return None
            
        # Use training log filename as default chart save path
        if save_path is None:
            save_path = os.path.join(self.log_dir, f'{self.model_name}_metrics.png')
        
        # Create a figure with four subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'AlphaZero Training Metrics - {self.model_name}', fontsize=16)
        
        # Plot loss curves
        axes[0, 0].plot(self.iterations, self.policy_losses, 'b-', label='Policy Loss')
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Iterations')
        axes[0, 0].set_ylabel('Loss Value')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.iterations, self.value_losses, 'r-', label='Value Loss')
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Iterations')
        axes[0, 1].set_ylabel('Loss Value')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.iterations, self.total_losses, 'g-', label='Total Loss')
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].set_xlabel('Iterations')
        axes[1, 0].set_ylabel('Loss Value')
        axes[1, 0].grid(True)
        
        # Plot loss over time
        if len(self.timestamps) > 1:
            date_format = mdates.DateFormatter('%m-%d %H:%M')
            dates = [datetime.fromtimestamp(ts) for ts in self.timestamps]
            
            axes[1, 1].plot(dates, self.total_losses, 'g-', label='Total Loss')
            axes[1, 1].set_title('Loss Over Time')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Total Loss')
            axes[1, 1].grid(True)
            axes[1, 1].xaxis.set_major_formatter(date_format)
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def load_from_csv(self, csv_path=None):
        """
        Load training logs from CSV file
        
        Parameters:
            csv_path (str): CSV file path, if None use the instance's log file
        """
        file_path = csv_path if csv_path else self.log_file
        
        try:
            df = pd.read_csv(file_path)
            self.iterations = df['iteration'].tolist()
            self.policy_losses = df['policy_loss'].tolist()
            self.value_losses = df['value_loss'].tolist()
            self.total_losses = df['total_loss'].tolist()
            
            if 'timestamp' in df.columns:
                self.timestamps = df['timestamp'].tolist()
            
            return True
        except Exception as e:
            print(f"加载训练日志时出错: {e}")
            return False
    
    def get_latest_metrics(self, n=1):
        """
        Get the latest n iterations of training metrics
        
        Parameters:
            n (int): Return metrics for the most recent n iterations
            
        Returns:
            dict: Dictionary containing the latest training metrics
        """
        if not self.iterations:
            return None
            
        n = min(n, len(self.iterations))
        return {
            'iteration': self.iterations[-n:],
            'policy_loss': self.policy_losses[-n:],
            'value_loss': self.value_losses[-n:],
            'total_loss': self.total_losses[-n:]
        }
    
    def get_summary(self):
        """
        Get statistical summary of the training process
        
        Returns:
            dict: Training statistics summary
        """
        if not self.iterations:
            return None
            
        return {
            'iterations': len(self.iterations),
            'policy_loss': {
                'min': min(self.policy_losses),
                'max': max(self.policy_losses),
                'avg': sum(self.policy_losses) / len(self.policy_losses),
                'final': self.policy_losses[-1]
            },
            'value_loss': {
                'min': min(self.value_losses),
                'max': max(self.value_losses),
                'avg': sum(self.value_losses) / len(self.value_losses),
                'final': self.value_losses[-1]
            },
            'total_loss': {
                'min': min(self.total_losses),
                'max': max(self.total_losses),
                'avg': sum(self.total_losses) / len(self.total_losses),
                'final': self.total_losses[-1]
            }
        }