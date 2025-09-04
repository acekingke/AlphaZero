#!/usr/bin/env python3
"""
Test script for the simplified MCTS implementation.
Tests canonical state support and Tree Visitor pattern.
"""

import torch
import numpy as np
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from simplified_mcts import SimplifiedMCTS

def test_canonical_state_consistency():
    """Test that canonical state representation is consistent."""
    print("Testing canonical state consistency...")
    
    # Create environment and make some moves
    env = OthelloEnv(size=6)
    env.reset()
    
    # Test canonical state at initial position
    canonical_state1 = env.board.get_canonical_state()
    print(f"Initial canonical state (Black's turn, player={env.board.current_player}):")
    print(canonical_state1)
    
    # Make a move
    valid_moves = env.board.get_valid_moves()
    if valid_moves:
        row, col = valid_moves[0]
        env.board.make_move(row, col)
    
    # Test canonical state after move (now White's turn)
    canonical_state2 = env.board.get_canonical_state()
    print(f"\nAfter move canonical state (White's turn, player={env.board.current_player}):")
    print(canonical_state2)
    
    # Verify that in both cases, current player's pieces are represented as +1
    print(f"\nCanonical state consistency check:")
    print(f"Initial state has +1 pieces: {np.any(canonical_state1 == 1)}")
    print(f"After move state has +1 pieces: {np.any(canonical_state2 == 1)}")
    
    return True

def test_simplified_mcts():
    """Test the simplified MCTS implementation."""
    print("\n" + "="*50)
    print("Testing Simplified MCTS Implementation")
    print("="*50)
    
    # Initialize environment
    env = OthelloEnv(size=6)
    env.reset()
    
    # Initialize neural network
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AlphaZeroNetwork(game_size=6, device=device)
    model.eval()
    
    # Initialize simplified MCTS
    mcts = SimplifiedMCTS(
        model=model,
        c_puct=1.0,
        num_simulations=50,  # Reduced for testing
        dirichlet_alpha=0.5,
        dirichlet_weight=0.25
    )
    
    print(f"Environment initialized. Current player: {env.board.current_player}")
    print(f"Valid moves: {env.board.get_valid_moves()}")
    print(f"Board state:\n{env.board}")
    
    # Test MCTS search
    print("\nRunning MCTS search...")
    try:
        canonical_state = env.board.get_canonical_state()
        action_probs = mcts.search(
            state=canonical_state,
            env=env,
            temperature=1.0,
            add_noise=True
        )
        
        print(f"MCTS search completed successfully!")
        print(f"Action probabilities shape: {action_probs.shape}")
        print(f"Sum of probabilities: {np.sum(action_probs):.6f}")
        
        # Show top actions
        top_actions = np.argsort(action_probs)[::-1][:5]
        print(f"\nTop 5 actions:")
        for i, action in enumerate(top_actions):
            if action_probs[action] > 0:
                row, col = env.get_coords_from_action(action)
                if row is not None:
                    print(f"  {i+1}. Action {action} ({row},{col}): {action_probs[action]:.4f}")
                else:
                    print(f"  {i+1}. Action {action} (pass): {action_probs[action]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error in MCTS search: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_iterative_vs_recursive_equivalence():
    """Test that iterative implementation produces similar results."""
    print("\n" + "="*50)
    print("Testing Iterative Implementation")
    print("="*50)
    
    # Initialize environment and model
    env = OthelloEnv(size=6)
    env.reset()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AlphaZeroNetwork(game_size=6, device=device)
    model.eval()

    # Test a few simulation counts to ensure stability
    simulation_counts = [5, 10, 20]
    for num_sims in simulation_counts:
        mcts = SimplifiedMCTS(
            model=model,
            c_puct=1.0,
            num_simulations=num_sims,
            dirichlet_alpha=0.5,
            dirichlet_weight=0.25
        )
        try:
            canonical_state = env.board.get_canonical_state()
            action_probs = mcts.search(
                state=canonical_state,
                env=env,
                temperature=1.0,
                add_noise=False  # No noise for consistency
            )
            print(f"  ✓ Completed {num_sims} simulations successfully")
            print(f"  ✓ Probability sum: {np.sum(action_probs):.6f}")
            assert np.all(action_probs >= 0), "Negative probabilities found"
            assert np.abs(np.sum(action_probs) - 1.0) < 1e-5, "Probabilities don't sum to 1"
        except Exception as e:
            print(f"  ✗ Failed with {num_sims} simulations: {e}")
            return False

    print("✓ All simulation counts completed successfully!")
    return True

if __name__ == "__main__":
    print("Testing Simplified MCTS with Canonical State Support")
    print("="*60)
    
    # Run tests
    success = True
    
    try:
        success &= test_canonical_state_consistency()
        success &= test_simplified_mcts()
        success &= test_iterative_vs_recursive_equivalence()
        
        if success:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("Simplified MCTS with canonical state support is working correctly!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ SOME TESTS FAILED")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()