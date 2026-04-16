import numpy as np

def get_all_symmetries(state, policy, size):
    """
    Generate all non-identity D4 symmetries of a board position.

    Matches alpha-zero-general OthelloGame.getSymmetries():
    - 4 rotations × 2 flips = 8 total, including identity
    - Returns 7 non-identity symmetries (caller provides identity)

    Args:
        state: Board state — either (size, size) canonical board
               or (C, size, size) multi-channel observation
        policy: Policy vector (size*size + 1, last = pass action)
        size: Board size

    Returns:
        List of 7 tuples (state, policy) for non-identity symmetries.
    """
    symmetries = []

    state_array = np.array(state, dtype=np.float32)
    policy_array = np.array(policy, dtype=np.float32)

    policy_without_pass = policy_array[:-1].reshape(size, size)
    pass_policy = policy_array[-1]

    is_multichannel = state_array.ndim == 3

    for i in range(4):  # rotations: 0, 90, 180, 270
        for flip in [False, True]:
            if i == 0 and not flip:
                continue  # skip identity

            if is_multichannel:
                sym_state = np.zeros_like(state_array)
                for ch in range(state_array.shape[0]):
                    rotated = np.rot90(state_array[ch], i)
                    sym_state[ch] = np.fliplr(rotated) if flip else rotated
            else:
                sym_state = np.rot90(state_array, i)
                if flip:
                    sym_state = np.fliplr(sym_state)

            sym_policy = np.rot90(policy_without_pass, i)
            if flip:
                sym_policy = np.fliplr(sym_policy)

            sym_policy_flat = np.append(sym_policy.ravel(), pass_policy)

            symmetries.append(
                (sym_state.astype(np.float32), sym_policy_flat.astype(np.float32))
            )

    return symmetries
