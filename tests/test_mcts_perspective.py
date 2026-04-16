"""
Tests for MCTS perspective handling.

Historical note (2026-04-14): TestTerminalValuePerspective and
TestBackpropWithForcedPass classes were removed after the dict-based MCTS
rewrite. They drove the removed Node/_simulate_iterative internals directly.
The invariants (terminal-value sign, forced-pass backprop) are now exercised
end-to-end through the public search() API in test_state_consistency.py.
"""
import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.othello import OthelloEnv
from mcts.mcts import MCTS
from models.neural_network import AlphaZeroNetwork


class TestRandomTieBreaking(unittest.TestCase):
    """
    Verify that MCTS with temperature=0 randomly breaks ties among actions
    with equal max visit counts. Without this, multiple arena games from
    the same starting position collapse to 2 unique outcomes due to
    deterministic np.argmax behavior.
    """

    def test_tie_breaking_differs_across_seeds(self):
        """
        Given a fresh untrained model with 50 MCTS sims (which produces
        ties in visit counts like [13, 13, 13, 11] for 4 actions), running
        MCTS multiple times with different numpy random seeds should yield
        different argmax actions due to random tie-breaking.
        """
        import torch
        torch.manual_seed(42)  # Fix model weights so NN output is deterministic

        game_size = 6
        env = OthelloEnv(size=game_size)
        env.reset()

        model = AlphaZeroNetwork(game_size, device='cpu')
        model.eval()
        # Use 50 sims so visit counts produce ties (13/13/13/11 for 4 actions)
        mcts = MCTS(model, c_puct=2.0, num_simulations=50)

        canonical = env.board.get_canonical_state()
        state = mcts.canonical_to_observation(canonical, env)

        # First verify there ARE ties for this setup (else the test is vacuous)
        raw_probs = mcts.search(state, env, temperature=1.0, add_noise=False)
        valid_mask = env.get_valid_moves_mask()
        valid_visits = raw_probs[valid_mask == 1] * 50  # approximate visit counts
        max_val = valid_visits.max()
        num_ties = int((np.abs(valid_visits - max_val) < 0.5).sum())
        self.assertGreater(
            num_ties, 1,
            f"Test setup invalid: no ties in visit counts {valid_visits}. "
            f"Need to adjust num_simulations so MCTS produces tied visit counts."
        )

        # Run MCTS with temperature=0 multiple times, varying only np.random seed.
        # With random tie-breaking, the chosen argmax action should differ
        # across seeds.
        selected_actions = set()
        for seed in range(20):
            np.random.seed(seed)
            action_probs = mcts.search(state, env, temperature=0, add_noise=False)
            action = int(np.argmax(action_probs))
            selected_actions.add(action)

        # We expect at least 2 different actions across the 20 seeds.
        # If the code always uses deterministic argmax (picking lowest index),
        # len(selected_actions) would be 1.
        self.assertGreater(
            len(selected_actions), 1,
            f"MCTS with temp=0 should break ties randomly across different seeds. "
            f"All 20 seeds picked the same action: {selected_actions}. "
            f"This indicates the tie-breaking fix is not working."
        )


if __name__ == "__main__":
    unittest.main()
