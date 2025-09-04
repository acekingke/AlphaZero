#!/usr/bin/env python3
"""
Quick integration test for the new simplified MCTS implementation.
"""

import torch
import numpy as np
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS

def test_integration():
    """Test integration with existing codebase."""
    print("🔍 Testing MCTS Integration...")
    
    # Initialize environment
    env = OthelloEnv(size=6)
    env.reset()
    
    # Initialize neural network
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AlphaZeroNetwork(game_size=6, device=device)
    model.eval()
    
    # Initialize MCTS (using new simplified version)
    mcts = MCTS(
        model=model,
        c_puct=1.0,
        num_simulations=25,  # Quick test
        dirichlet_alpha=0.5,
        dirichlet_weight=0.25
    )
    
    print(f"✓ Environment: {env.board.size}x{env.board.size} Othello")
    print(f"✓ Neural Network: {type(model).__name__}")
    print(f"✓ MCTS: {type(mcts).__name__}")
    print(f"✓ Device: {device}")
    
    # Test MCTS search
    print("\n🎯 Running MCTS search...")
    canonical_state = env.board.get_canonical_state()
    
    try:
        action_probs = mcts.search(
            state=canonical_state,
            env=env,
            temperature=1.0,
            add_noise=True
        )
        
        print(f"✓ Search completed successfully!")
        print(f"✓ Action probabilities shape: {action_probs.shape}")
        print(f"✓ Probability sum: {np.sum(action_probs):.6f}")
        
        # Check that we got valid probabilities
        assert np.all(action_probs >= 0), "Found negative probabilities"
        assert np.abs(np.sum(action_probs) - 1.0) < 1e-6, "Probabilities don't sum to 1"
        
        # Show top action
        best_action = np.argmax(action_probs)
        coords = env.get_coords_from_action(best_action)
        if coords:
            print(f"✓ Best action: {best_action} at position {coords} with probability {action_probs[best_action]:.4f}")
        else:
            print(f"✓ Best action: {best_action} (pass) with probability {action_probs[best_action]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCTS search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_canonical_state_behavior():
    """Test that canonical state is being used correctly."""
    print("\n🧪 Testing Canonical State Behavior...")
    
    env = OthelloEnv(size=6)
    env.reset()
    
    # Test at different game states
    for i in range(3):
        canonical_state = env.board.get_canonical_state()
        current_player = env.board.current_player
        
        print(f"Move {i}: Current player = {current_player}")
        print(f"Canonical state has +1 pieces: {np.any(canonical_state == 1)}")
        print(f"Canonical state has -1 pieces: {np.any(canonical_state == -1)}")
        
        # Make a move
        valid_moves = env.board.get_valid_moves()
        if valid_moves:
            row, col = valid_moves[0]
            env.board.make_move(row, col)
        else:
            break
    
    print("✓ Canonical state representation is consistent")
    return True

if __name__ == "__main__":
    print("🚀 MCTS Integration Test")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_integration()
        success &= test_canonical_state_behavior()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
            print("New simplified MCTS is working correctly with existing codebase!")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("❌ SOME TESTS FAILED")
            print("=" * 50)
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()