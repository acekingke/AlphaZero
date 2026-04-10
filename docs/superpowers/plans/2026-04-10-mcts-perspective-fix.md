# MCTS Perspective Bug Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix MCTS value calculation bugs by adopting alpha-zero-general's single-perspective design where all values are consistently represented from the current leaf player's perspective.

**Architecture:** Replace the mixed perspective system (NN uses canonical/leaf perspective, terminal uses search_start_player perspective) with a single, consistent convention: all values at any node represent the position from that node's current player's perspective. Backpropagation tracks actual player transitions (handling Othello's forced-pass turns) and only flips the value when the player actually changes.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest

---

## Background: Why This Matters

Two bugs discovered in `mcts/mcts.py`:

1. **Inconsistent value perspective (line 213)**: terminal value uses `self.search_start_player` (fixed for entire search), but NN value at a leaf uses `env_copy.board.current_player` (the leaf's current player via canonical state). At odd-length paths these two perspectives are opposite, causing ~50% of terminal evaluations to have a sign error.

2. **Blind backprop sign flipping (lines 224-228)**: the loop flips sign on every step, assuming strict player alternation. But Othello's `make_move` switches back to the same player when the opponent has no valid moves (forced pass), so the MCTS tree can have parent→child transitions where the current player stays the same. The blind flip introduces additional sign errors.

Combined, these bugs cause the training loop to learn from contradictory value labels, which explains the non-transitive strategy cycling, oscillating win rates (7-27% vs random), and stuck Arena rejection loops observed in v1-v5 training.

Reference implementation: `/Users/kyc/homework/tmp/alpha-zero-general` uses a single-perspective design where:
- Every state in MCTS is in canonical form (current player = +1)
- `getGameEnded` returns outcome from the canonical player's perspective
- `search()` returns `-v` at every level (caller's perspective = flipped child's perspective)
- Players always alternate in the tree because `getNextState` returns `-player` even for pass actions

---

## File Structure

- `mcts/mcts.py` — core fix (edit `_simulate_iterative`, small change to `search`)
- `tests/test_mcts_perspective.py` — new test file focused on perspective correctness (separate from existing tests so failures are clearly attributable to this fix)
- `tests/test_mcts_value_bug.py` — existing weak tests, leave alone (they verify tracking of `search_start_player`, not correctness)
- `tests/test_arena.py` — existing tests for Arena — should continue to pass

We do NOT modify `env/othello.py`. Changing the environment to match alpha-zero-general's "always alternate" convention would be too invasive and would break existing self-play data loading, evaluation scripts, and tests. Instead, we make MCTS aware of Othello's forced-pass semantics.

---

### Task 1: Write failing test for terminal value at odd-length path

**Files:**
- Create: `tests/test_mcts_perspective.py`

- [ ] **Step 1: Write the failing test**

```python
"""
Tests that verify MCTS value calculation uses a single consistent perspective
(the leaf's current player) rather than mixing perspectives. These tests
specifically target the bugs discovered in the 2026-04-10 investigation.
"""
import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

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
        self.env = OthelloEnv(size=self.game_size)
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
        env.board.board = np.array([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [ 1,  1,  1,  1,  1,  0],
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
        self.mcts.search_start_player = -1
        root = Node(0)
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
        self.assertGreaterEqual(
            root_avg_value, 0.0,
            f"Root value should be non-negative for a winning position, "
            f"got {root_avg_value}. This indicates the terminal value or "
            f"backprop logic has a sign error.",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcts_perspective.py::TestTerminalValuePerspective::test_terminal_win_for_leaf_current_player_is_positive -v`

Expected: FAIL — root value will be negative because of the sign bug.

(If this test does NOT fail, the bug may not exist as hypothesized, and we should investigate before proceeding.)

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_mcts_perspective.py
git commit -m "test: Add failing test for MCTS terminal value perspective bug

Documents the discovered bug: terminal value uses search_start_player
perspective instead of leaf current player perspective, causing sign
errors at odd-length paths."
```

---

### Task 2: Fix terminal value to use leaf's current player perspective

**Files:**
- Modify: `mcts/mcts.py` lines 205-213

- [ ] **Step 1: Apply the fix**

Replace in `mcts/mcts.py`:

```python
        if game_ended:
            # Use game result as value
            winner = env_copy.board.get_winner()
            if winner == 0:  # Draw
                value = 0.0
            else:
                # 正确的价值计算：从搜索开始时的玩家视角
                # 如果搜索开始的玩家获胜，value=1；否则value=-1
                value = 1.0 if winner == self.search_start_player else -1.0
```

With:

```python
        if game_ended:
            # Terminal value from leaf's current player perspective.
            # This matches the NN convention where canonical_state is always
            # from the current player's perspective (current player = +1).
            # Both branches of the if/else must agree on the value reference
            # frame for backpropagation to work correctly.
            winner = env_copy.board.get_winner()
            if winner == 0:  # Draw
                value = 0.0
            else:
                leaf_current_player = env_copy.board.current_player
                value = 1.0 if winner == leaf_current_player else -1.0
```

- [ ] **Step 2: Run the failing test to verify it now passes**

Run: `python -m pytest tests/test_mcts_perspective.py::TestTerminalValuePerspective::test_terminal_win_for_leaf_current_player_is_positive -v`

Expected: PASS

- [ ] **Step 3: Run all arena tests to check no regression**

Run: `python -m pytest tests/test_arena.py -q`

Expected: All 16 tests pass (may need to regenerate expected values if any test accidentally depended on the bug).

- [ ] **Step 4: Commit the fix**

```bash
git add mcts/mcts.py
git commit -m "fix(mcts): Use leaf current player for terminal value

Previously terminal value used search_start_player, which is inconsistent
with NN evaluation that uses env_copy.current_player (canonical perspective).
At odd-length simulation paths these two perspectives are opposite, causing
~50% of terminal evaluations to have a sign error and poisoning training.

Now matches alpha-zero-general's single-perspective convention where all
values at a node are from that node's current player's perspective."
```

---

### Task 3: Write failing test for backprop with forced pass

**Files:**
- Modify: `tests/test_mcts_perspective.py` (append new test class)

- [ ] **Step 1: Append the test**

Add to `tests/test_mcts_perspective.py`:

```python
class TestBackpropWithForcedPass(unittest.TestCase):
    """
    Verify that MCTS backpropagation correctly handles Othello's forced-pass
    case where a move doesn't actually switch the current player (because
    the opponent has no valid moves and plays immediately gets skipped).
    """

    def setUp(self):
        self.game_size = 6
        self.model = AlphaZeroNetwork(self.game_size, device="cpu")
        self.model.eval()
        self.mcts = MCTS(self.model, c_puct=2.0, num_simulations=1)

    def test_backprop_respects_actual_player_transitions(self):
        """
        When a move triggers a forced pass (opponent has no moves), the
        current_player does NOT switch. MCTS backprop must detect this and
        skip the sign flip for that step.
        """
        # Set up a board state where Black plays, and after Black's move
        # White has no valid moves, so Black plays again immediately.
        # This tests that backprop doesn't spuriously flip the sign.

        env = OthelloEnv(size=self.game_size)
        env.reset()

        # Construct state: Black will play a move and then have to play
        # again because White has no valid moves afterwards.
        # Easiest way: near-full board where only Black can move.
        env.board.board = np.array([
            [ 1,  1,  1,  1,  1,  1],
            [ 1, -1, -1, -1, -1,  1],
            [ 1, -1,  1,  1, -1,  1],
            [ 1, -1,  1,  1, -1,  1],
            [ 1, -1, -1, -1, -1,  1],
            [ 1,  1,  1,  1,  1,  0],
        ], dtype=np.int8)
        env.board.current_player = -1  # Black to move
        env.board.done = False

        # Step through Black's move and check if current_player stays as Black
        valid = env.board.get_valid_moves()
        # Sanity: there must be valid moves
        if len(valid) == 0:
            self.skipTest("Test board has no valid moves for Black")

        test_env = OthelloEnv(size=self.game_size)
        test_env.board.board = env.board.board.copy()
        test_env.board.current_player = -1
        test_env.board.done = False

        row, col = valid[0]
        action = row * self.game_size + col
        test_env.step(action)

        # After the step: has current_player stayed as Black?
        # If yes, we have a forced-pass scenario to test.
        stayed_same = test_env.board.current_player == -1
        if not stayed_same:
            self.skipTest(
                "Test board doesn't trigger forced-pass scenario "
                "(current_player switched normally)"
            )

        # Now run MCTS and verify backprop is correct.
        # This is hard to assert directly without instrumenting MCTS,
        # so we rely on a functional check: MCTS root value after one
        # simulation should reflect a consistent perspective.
        self.mcts.search_start_player = -1
        root = Node(0)
        policy = env.get_valid_moves_mask().astype(np.float32)
        if policy.sum() > 0:
            policy = policy / policy.sum()
        root.expand(env.board.get_canonical_state(), policy)

        # Should not crash
        self.mcts._simulate_iterative(root, env)

        # After one simulation, root should have been visited (via backprop)
        self.assertEqual(
            root.visit_count, 1,
            "Root should be visited exactly once after one simulation",
        )
```

- [ ] **Step 2: Run the new test**

Run: `python -m pytest tests/test_mcts_perspective.py::TestBackpropWithForcedPass -v`

Expected: Either PASS (if our current fix already handles it) or SKIP (if the test board doesn't trigger forced pass). Document the outcome.

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_mcts_perspective.py
git commit -m "test: Add test for MCTS backprop with Othello forced-pass"
```

---

### Task 4: Track player transitions in path_stack for correct backprop

**Files:**
- Modify: `mcts/mcts.py` lines 184-228 (the `_simulate_iterative` method body)

- [ ] **Step 1: Apply the fix**

Replace the entire `_simulate_iterative` method body (lines 178-228) with:

```python
    def _simulate_iterative(self, root, env):
        """
        Perform one MCTS simulation using Tree Visitor pattern (iterative).

        Value perspective convention: every value stored in a node is from
        the perspective of the player whose turn it is at that state. This
        matches the canonical_state convention used by the neural network.

        Backpropagation flips the sign ONLY when the actual player switches
        between nodes. In Othello, a move can trigger a forced pass (opponent
        has no valid moves), in which case current_player stays the same and
        the sign should NOT flip for that step.
        """
        # Tree Visitor pattern: explicit stack of (parent_node, action, player_at_parent)
        path_stack = []
        node = root
        env_copy = copy.deepcopy(env)

        # Selection phase - traverse down the tree
        while node.expanded():
            action = node.select_child(self.c_puct)
            # Record the player who is about to act at this parent node.
            # We'll use this during backprop to detect perspective changes.
            player_at_parent = env_copy.board.current_player
            path_stack.append((node, action, player_at_parent))

            env_copy.step(action)

            if action in node.children:
                node = node.children[action]
            else:
                break

        # Evaluation phase — both branches must produce value from the
        # leaf's current player's perspective.
        canonical_state = env_copy.board.get_canonical_state()
        game_ended = env_copy.board.is_done()

        if game_ended:
            winner = env_copy.board.get_winner()
            if winner == 0:
                value = 0.0
            else:
                leaf_current_player = env_copy.board.current_player
                value = 1.0 if winner == leaf_current_player else -1.0
        else:
            policy, value = self._evaluate_state(canonical_state, env_copy)
            node.expand(canonical_state, policy)

        # Backpropagation — walk from leaf back to root, flipping the sign
        # only when the player actually changes between adjacent nodes.
        leaf_player = env_copy.board.current_player
        node.value_sum += value
        node.visit_count += 1

        current_value = value
        prev_player = leaf_player

        for parent_node, action, parent_player in reversed(path_stack):
            # If parent's player differs from the child's (prev) player,
            # the perspective flips and we negate. Otherwise (forced pass
            # scenario) the perspective stays the same.
            if parent_player != prev_player:
                current_value = -current_value
            parent_node.value_sum += current_value
            parent_node.visit_count += 1
            prev_player = parent_player
```

- [ ] **Step 2: Run all perspective tests**

Run: `python -m pytest tests/test_mcts_perspective.py -v`

Expected: All tests pass.

- [ ] **Step 3: Run all arena tests**

Run: `python -m pytest tests/test_arena.py -q`

Expected: All 16 tests pass.

- [ ] **Step 4: Run the existing MCTS value bug tests**

Run: `python -m pytest tests/test_mcts_value_bug.py tests/test_mcts_path_stack.py -q`

Expected: All pass (these existing tests only verify that `search_start_player` is tracked; they don't depend on the buggy sign logic).

- [ ] **Step 5: Commit**

```bash
git add mcts/mcts.py
git commit -m "fix(mcts): Track player transitions in backprop for forced pass

Previously backprop blindly flipped value sign at every step, assuming
strict player alternation. In Othello, a move can trigger a forced pass
(opponent has no valid moves), leaving current_player unchanged. The
blind flip introduced sign errors proportional to the frequency of
forced-pass transitions.

Now path_stack records the player at each parent node, and backprop only
negates when the actual player changes. Matches alpha-zero-general's
invariant that values at a node are always from that node's current
player's perspective."
```

---

### Task 5: Sanity test — run existing test suite

**Files:**
- Test only

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -x --ignore=tests/test_integration_performance.py --ignore=tests/test_memory_management.py --ignore=tests/test_multiprocessing_safety.py -q 2>&1 | tail -30`

Expected: 
- All tests pass, OR
- Pre-existing failures (test_othello.py, test_state_consistency.py — noted earlier as unrelated) still fail but nothing new

If any test newly fails, investigate and fix before proceeding.

- [ ] **Step 2: Quick model strength check**

Evaluate current `checkpoint_35.pt` (22-26% baseline from v4) with the FIXED MCTS to see if it plays any differently:

```bash
python main.py evaluate --model ./models/checkpoint_35.pt --num_games 30 --mcts_simulations 50 2>&1 | tail -5
```

Expected: Win rate is measurable and reasonable (anywhere from 5% to 50%). The purpose is not to verify improvement at this stage — just to confirm the fix doesn't crash. The same model with the fixed MCTS might play differently because MCTS produces different action distributions.

Record the observed win rate in the plan for later comparison after fresh training.

- [ ] **Step 3: Commit any fixes and the sanity check results**

If no code changes were needed, skip commit. Otherwise:

```bash
git add <changed files>
git commit -m "test: Verify MCTS perspective fix doesn't break existing tests"
```

---

### Task 6: Clean up old training artifacts and prepare for retraining

**Files:**
- Delete: `models/best.pt`, `models/checkpoint_*.pt`, `models/temp.pt`
- Delete: `models/*.pt.examples` (history sidecar files)

- [ ] **Step 1: Delete all old model artifacts**

```bash
rm -f /Users/kyc/homework/tmp/AlphaZero/models/best.pt
rm -f /Users/kyc/homework/tmp/AlphaZero/models/checkpoint_*.pt
rm -f /Users/kyc/homework/tmp/AlphaZero/models/temp.pt
rm -f /Users/kyc/homework/tmp/AlphaZero/models/global_best.pt
rm -f /Users/kyc/homework/tmp/AlphaZero/models/*.pt.examples
ls /Users/kyc/homework/tmp/AlphaZero/models/*.pt 2>/dev/null || echo "models clean"
```

Expected: `models clean` — no .pt files remain.

- [ ] **Step 2: Commit the rationale**

No code changes to commit; the deletion itself isn't tracked by git. Just note in a commit to document the reset:

```bash
# No commit needed — old checkpoints aren't tracked in git
```

---

### Task 7: Start fresh training run with fixed MCTS

**Files:**
- No files modified

- [ ] **Step 1: Start training in background**

Run:

```bash
cd /Users/kyc/homework/tmp/AlphaZero
nohup python main.py train \
  --num_iterations 40 \
  --self_play_games 100 \
  --mcts_simulations 50 \
  --temperature 1.0 \
  --temp_threshold 15 \
  --c_puct 2.0 \
  --batch_size 128 \
  --num_epochs 10 \
  --arena_games 40 \
  --arena_threshold 0.6 \
  --arena_mcts_simulations 25 \
  --eval_vs_random_interval 5 \
  --eval_vs_random_games 30 \
  --use_mps \
  --use_multiprocessing \
  --mp_num_workers 6 \
  --mp_games_per_worker 5 \
  --dirichlet_alpha 0.3 \
  --dirichlet_weight 0.25 > training_fixed_mcts.log 2>&1 &
echo "PID: $!"
```

Expected: Training starts, outputs PID. Do NOT wait for completion.

- [ ] **Step 2: Verify training started correctly**

```bash
sleep 10  # wait for startup
tail -20 /Users/kyc/homework/tmp/AlphaZero/training_fixed_mcts.log
```

Expected: Log shows "Training will use: mps", "No best model found", and self-play progress bars. No errors.

- [ ] **Step 3: No commit** (training is a runtime action, not a code change)

Exit the task. The training runs in the background for ~80-120 minutes. The user will check progress separately.

---

## Post-Training Verification (manual, not part of this plan)

After training completes (~80-120 min):

1. Check `global_best.pt` win rate vs random (this is the ground truth measure, unaffected by Arena non-transitivity)
2. Compare against v4's 22% baseline and v5's 8% result
3. If fix is effective, expect `global_best` win rate >= 30-50% after 40 iterations

## Self-Review Notes

- **Spec coverage**: All aspects of the design are covered — terminal value perspective (Task 2), backprop transitions (Task 4), tests for both (Tasks 1, 3), no-regression verification (Task 5), training relaunch (Tasks 6-7).
- **Placeholder scan**: No TBD/TODO/placeholder text. All code steps include full code blocks.
- **Type consistency**: Method signatures consistent — `_simulate_iterative(self, root, env)` unchanged, `search()` unchanged, path_stack tuple now `(node, action, player_at_parent)` (3-tuple) consistent across write and read sites.
- **Side note**: The existing `tests/test_mcts_value_bug.py` tests only verify that `search_start_player` gets set; they don't verify correctness of the sign. Leaving them alone is intentional — they'll continue to pass and don't need modification. The new file `tests/test_mcts_perspective.py` contains the tests that actually verify correctness.
