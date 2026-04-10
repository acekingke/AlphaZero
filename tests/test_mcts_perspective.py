"""
Tests that verify MCTS value calculation uses a single consistent perspective
(the leaf's current player) rather than mixing perspectives. These tests
specifically target the bugs discovered in the 2026-04-10 investigation.
"""
import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.othello import OthelloEnv
from mcts.mcts import MCTS, Node
from models.neural_network import AlphaZeroNetwork


class TestTerminalValuePerspective(unittest.TestCase):
    """
    Verify that terminal state values in MCTS are computed from the leaf's
    current player perspective (to match NN canonical-state convention).
    """

    def setUp(self):
        self.game_size = 6
        # Use a real model but with deterministic weights so we can test
        # the MCTS logic without randomness in the NN.
        self.model = AlphaZeroNetwork(self.game_size, device="cpu")
        self.model.eval()
        self.mcts = MCTS(self.model, c_puct=2.0, num_simulations=1)

    def test_terminal_win_for_leaf_current_player_is_positive(self):
        """
        When the leaf's current player is the winner, terminal value must be +1.0
        (from that player's perspective).
        """
        # Set up a board where Black (-1) is about to make a move that ends
        # the game with Black winning. After Black's move, current_player
        # becomes White (+1). Terminal value at the leaf should be -1 from
        # White's perspective (because Black won = bad for White).
        env = OthelloEnv(size=self.game_size)
        env.reset()

        # Manually construct a near-terminal state: board almost full, Black
        # to move, Black has one move that fills the board with Black winning.
        # Board: 33 Black pieces, 2 White pieces (at (4,5) and (5,4)), one empty at (5,5).
        # Black to move. Black plays (5,5):
        #   - Up direction: (4,5)=W, (3,5)=B → flips (4,5)
        #   - Left direction: (5,4)=W, (5,3)=B → flips (5,4)
        # After move: board is full (36 Black, 0 White), Black wins.
        # After step(): env_copy.current_player = +1 (White, switched in make_move).
        # This sets up the bug scenario:
        #   - Bug: terminal value uses search_start_player (-1) → +1
        #   - Fix: terminal value uses leaf.current_player (+1) → -1
        # After backprop sign-flip into root, the test detects the difference.
        env.board.board = np.array([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1,  1],
            [-1, -1, -1, -1,  1,  0],
        ], dtype=np.int8)
        env.board.current_player = -1  # Black to move
        env.board.done = False

        # Sanity check: Black has a move that fills the board
        valid_moves = env.board.get_valid_moves()
        self.assertTrue(
            len(valid_moves) > 0,
            "Test setup failed: Black should have valid moves",
        )

        # Run a single MCTS simulation with search starting from Black's turn
        # Normally set by MCTS.search(); we bypass that to drive _simulate_iterative directly.
        self.mcts.search_start_player = -1
        root = Node(0)  # prior probability = 0 for root
        # Manually expand root with a uniform policy over valid moves
        policy = env.get_valid_moves_mask().astype(np.float32)
        policy = policy / policy.sum()
        root.expand(env.board.get_canonical_state(), policy)

        # Run one simulation — this will descend into a child, which should
        # be a terminal state
        self.mcts._simulate_iterative(root, env)

        # Check the root's value after one simulation.
        # Root represents Black's turn. Black has a winning move here.
        # Therefore the Q value of Black's chosen action should be POSITIVE
        # (from Black's perspective at the root).
        # If the bug exists, this will be NEGATIVE.
        root_avg_value = root.value_sum / max(root.visit_count, 1)
        # After fix: leaf is terminal with White to move, Black has won.
        # Leaf value = -1 (from White's perspective). After backprop sign-flip
        # crossing into root (Black's perspective), root.value_sum = +1, and
        # with visit_count = 1, root_avg_value should be exactly +1.0.
        self.assertAlmostEqual(
            root_avg_value, 1.0, places=5,
            msg=(
                f"Root value should be +1.0 for a winning position from Black's "
                f"perspective, got {root_avg_value}. With the perspective bug, "
                f"this returns -1.0 (sign inverted)."
            ),
        )


if __name__ == "__main__":
    unittest.main()
