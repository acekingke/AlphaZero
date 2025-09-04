#!/usr/bin/env python3
"""
Test training with the new simplified MCTS implementation.
"""

import torch
from train import AlphaZeroTrainer

def test_training():
    """Test a quick training iteration."""
    print("🎯 Testing Training with New MCTS...")
    
    try:
        # Initialize trainer with reduced parameters for quick test
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,      # Just 1 iteration
            num_self_play_games=5,        # Just 5 self-play games
            num_mcts_simulations=25,  # Reduced simulations
            num_epochs=1,          # Just 1 training epoch
            batch_size=32,
            temperature=1.0,
            dirichlet_alpha=0.5,
            c_puct=1.0,
            checkpoint_path='./models/test_checkpoint',
            use_mps=torch.backends.mps.is_available(),
            use_cuda=False
        )
        
        print(f"✓ Trainer initialized")
        print(f"✓ Device: {trainer.device}")
        print(f"✓ MCTS simulations: {trainer.num_mcts_simulations}")
        print(f"✓ Self-play games: {trainer.num_self_play_games}")
        
        # Test one iteration
        print("\n🚀 Running one training iteration...")
        trainer.train()
        
        print("✓ Training iteration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Training Test with New MCTS")
    print("=" * 40)
    
    success = test_training()
    
    if success:
        print("\n" + "=" * 40)
        print("🎉 TRAINING TEST PASSED! 🎉")
        print("Training works correctly with new MCTS!")
        print("=" * 40)
    else:
        print("\n" + "=" * 40)
        print("❌ TRAINING TEST FAILED")
        print("=" * 40)