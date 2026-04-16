# Research: Arena-Based Model Selection

## R1: Arena Match Implementation Pattern

**Decision**: Reuse existing `MCTS` class and `OthelloEnv` for arena matches, similar to `play.py:play_game()` but with two AI players instead of human+AI.

**Rationale**: The project already has all the building blocks. `play.py` demonstrates the match flow pattern with `AlphaZeroPlayer.get_action()`. Arena matches follow the same pattern but with two `AlphaZeroPlayer`-like instances.

**Alternatives considered**:
- Importing `play_game()` from `play.py` directly — rejected because it has render/print logic and human player support that would need to be stripped
- Creating a separate `arena.py` module — rejected because the Arena is tightly coupled with the training loop and only ~60 lines of code

## R2: Model Weight Loading for Arena

**Decision**: Load both models by creating two `AlphaZeroNetwork` instances — one from `best.pt` and one from the just-trained weights (already in memory). Use CPU for arena to avoid GPU memory pressure during evaluation.

**Rationale**: Arena evaluation doesn't need GPU speed (25 sims is fast). Using CPU avoids loading two models on the GPU simultaneously and keeps the GPU free.

**Alternatives considered**:
- Sharing a single model and swapping weights — rejected because it's error-prone and makes the code harder to follow
- Using GPU for arena — rejected because 25 MCTS sims on CPU is fast enough and avoids GPU memory contention

## R3: Win Rate Calculation

**Decision**: `win_rate = new_wins / (new_wins + old_wins)`, draws excluded. Threshold >= 0.6.

**Rationale**: Follows alpha-zero-general reference implementation exactly. Excluding draws focuses the metric on decisive outcomes.

**Alternatives considered**:
- Including draws as 0.5 wins — rejected because it dilutes the signal and makes the threshold harder to reason about
- Using Elo rating — rejected as overkill for a simple comparison

## R4: First Iteration Bootstrapping

**Decision**: If `best.pt` does not exist, the first trained model is auto-accepted without arena evaluation.

**Rationale**: There is no meaningful opponent for the first model. Any model is better than no model.

**Alternatives considered**:
- Playing against random — rejected because even a weak model should beat random, so this doesn't add useful filtering
