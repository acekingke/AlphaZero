import numpy as np

def get_all_symmetries(state, policy, size):
    """
    Generate all symmetrical board positions and their corresponding policies.
    
    Args:
        state: The board state as numpy array (shape: 3 x size x size)
        policy: The policy vector (shape: size*size + 1)
        size: The size of the board
        
    Returns:
        List of tuples (state, policy) containing all symmetrical positions
    """
    # Initialize list to store all symmetries
    symmetries = []
    
    # Ensure state is numpy array with correct shape and type
    state_array = np.array(state, dtype=np.float32)
    policy_array = np.array(policy, dtype=np.float32)
    
    # Get policy without the pass move (last element)
    policy_without_pass = policy_array[:-1].reshape(size, size)
    pass_policy = policy_array[-1]
    
    # For each of 6 symmetries (3 rotations [90,180,270] * 2 flips)
    for i in range(1, 4):  # 3 rotations (skipping 0 degrees)
        for flip in [False, True]:  # 2 flips (None, horizontal)
            # Rotate each channel of state and policy
            sym_state = np.zeros_like(state_array)
            for channel in range(3):
                sym_state[channel] = np.rot90(state_array[channel], i)
            sym_policy = np.rot90(policy_without_pass, i)
            
            # Flip if needed
            if flip:
                for channel in range(3):
                    sym_state[channel] = np.fliplr(sym_state[channel])
                sym_policy = np.fliplr(sym_policy)
            
            # For channel 2 (current player indicator), no need to transform as it's constant
            
            # Flatten policy and add pass move back
            sym_policy_flat = np.append(sym_policy.flatten(), pass_policy)
            
            # Add to symmetries list, ensuring correct types
            symmetries.append((sym_state.astype(np.float32), sym_policy_flat.astype(np.float32)))
            
    return symmetries