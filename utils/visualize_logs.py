#!/usr/bin/env python3
"""
AlphaZero Training Log Visualization Tool

Usage:
    python visualize_logs.py --log_dir ./logs [--latest] [--save path/to/save.png]

Options:
    --log_dir: Log directory path
    --log_file: Specific log file to visualize
    --latest: Use only the latest log file
    --save: Path to save the chart
    --show: Display the chart window
"""

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
try:
    import pandas as pd
except ImportError:
    print("Please install pandas first: pip install pandas")
    sys.exit(1)

# 将项目根目录添加到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.training_logger import TrainingLogger

def find_latest_log(log_dir):
    """Find the latest log file"""
    log_files = glob.glob(os.path.join(log_dir, 'alphazero_*.csv'))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return None
    
    # 按修改时间排序，取最新的
    return max(log_files, key=os.path.getmtime)

def merge_logs(log_dir):
    """Merge all log files"""
    log_files = glob.glob(os.path.join(log_dir, 'alphazero_*.csv'))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return None
    
    all_data = []
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    if not all_data:
        return None
    
    # 合并数据框并按迭代次数排序
    merged_df = pd.concat(all_data).sort_values(by='iteration')
    return merged_df

def visualize_logs(log_path=None, log_dir=None, use_latest=False, save_path=None, show_plot=False):
    """Visualize training logs"""
    logger = TrainingLogger()
    
    # Determine which log file to use
    if log_path:
        # Use specified log file
        print(f"Using specified log file: {log_path}")
        logger.load_from_csv(log_path)
    elif use_latest and log_dir:
        # Use the latest log file
        latest_log = find_latest_log(log_dir)
        if latest_log:
            print(f"Using latest log file: {latest_log}")
            logger.load_from_csv(latest_log)
        else:
            print("No log files found")
            return False
    elif log_dir:
        # Merge all log files
        print(f"Merging all log files")
        merged_df = merge_logs(log_dir)
        if merged_df is not None:
            # Store merged data in temporary file
            tmp_file = os.path.join(log_dir, 'tmp_merged_logs.csv')
            merged_df.to_csv(tmp_file, index=False)
            logger.load_from_csv(tmp_file)
            os.remove(tmp_file)  # Delete temporary file
        else:
            print("Failed to merge log files")
            return False
    else:
        print("Must provide either log file path or log directory")
        return False
    
    # 生成图表
    metrics_path = logger.plot_metrics(save_path=save_path, show_plot=show_plot)
    
    # Print training summary
    summary = logger.get_summary()
    if summary:
        print("\nTraining Summary:")
        print(f"Total Iterations: {summary['iterations']}")
        print(f"Final Policy Loss: {summary['policy_loss']['final']:.6f}")
        print(f"Final Value Loss: {summary['value_loss']['final']:.6f}")
        print(f"Final Total Loss: {summary['total_loss']['final']:.6f}")
        print(f"\nPolicy Loss - Min: {summary['policy_loss']['min']:.6f}, Max: {summary['policy_loss']['max']:.6f}, Avg: {summary['policy_loss']['avg']:.6f}")
        print(f"Value Loss - Min: {summary['value_loss']['min']:.6f}, Max: {summary['value_loss']['max']:.6f}, Avg: {summary['value_loss']['avg']:.6f}")
        print(f"Total Loss - Min: {summary['total_loss']['min']:.6f}, Max: {summary['total_loss']['max']:.6f}, Avg: {summary['total_loss']['avg']:.6f}")
    
    print(f"\nChart saved to: {metrics_path}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize AlphaZero Training Logs')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory path')
    parser.add_argument('--log_file', type=str, help='Specific log file to visualize')
    parser.add_argument('--latest', action='store_true', help='Use only the latest log file')
    parser.add_argument('--save', type=str, help='Path to save the chart')
    parser.add_argument('--show', action='store_true', help='Display the chart window')
    
    args = parser.parse_args()
    
    visualize_logs(
        log_path=args.log_file,
        log_dir=args.log_dir,
        use_latest=args.latest,
        save_path=args.save,
        show_plot=args.show
    )

if __name__ == "__main__":
    main()