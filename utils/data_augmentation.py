import numpy as np

def get_all_symmetries(state, policy, size):
    """
    Generate all 7 non-trivial D4 symmetries of a square board position.

    The dihedral group D4 has 8 elements (4 rotations × {no-flip, flip}).
    The identity (rot=0, no-flip) is the original input — it's returned by
    the caller separately. This function returns the 7 remaining symmetries.

    Args:
        state: The board state as numpy array (shape: 3 x size x size)
        policy: The policy vector (shape: size*size + 1, last = pass action)
        size: The size of the board

    Returns:
        List of 7 tuples (state, policy) containing all non-identity symmetries.
    """
    symmetries = []

    state_array = np.array(state, dtype=np.float32)
    policy_array = np.array(policy, dtype=np.float32)

    # Separate pass action from board policy
    policy_without_pass = policy_array[:-1].reshape(size, size)
    pass_policy = policy_array[-1]

    # Iterate all 8 D4 elements (4 rotations × 2 flips) and skip the identity.
    for i in range(4):  # rotations: 0, 90, 180, 270 degrees
        for flip in [False, True]:
            if i == 0 and not flip:
                continue  # skip identity — the caller already has the original

            # Rotate each channel of state
            sym_state = np.zeros_like(state_array)
            for channel in range(3):
                sym_state[channel] = np.rot90(state_array[channel], i)
            sym_policy = np.rot90(policy_without_pass, i)

            if flip:
                for channel in range(3):
                    sym_state[channel] = np.fliplr(sym_state[channel])
                sym_policy = np.fliplr(sym_policy)

            # Channel 2 (current-player indicator) is constant (all 1s), so
            # rotation/flip doesn't change it — this is correct.

            # Flatten board policy and append pass (pass is invariant under
            # board symmetries since it's "no spatial move").
            sym_policy_flat = np.append(sym_policy.flatten(), pass_policy)

            symmetries.append(
                (sym_state.astype(np.float32), sym_policy_flat.astype(np.float32))
            )

    return symmetries