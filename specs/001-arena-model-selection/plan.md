# Implementation Plan: Arena-Based Model Selection

**Branch**: `001-arena-model-selection` | **Date**: 2026-04-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-arena-model-selection/spec.md`

## Summary

Add an Arena evaluation step to the AlphaZero training loop. After each training iteration, the newly trained model plays 40 games (20 as each color) against the current best model using 25 MCTS simulations. The new model is accepted only if it wins >= 60% of non-draw games. This prevents training regression by maintaining a quality gate that only promotes improving models.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PyTorch, NumPy, tqdm (all existing)
**Storage**: File-based model checkpoints (`.pt` files)
**Testing**: pytest (existing test suite in `tests/`)
**Target Platform**: macOS (MPS), Linux (CUDA), CPU fallback
**Project Type**: CLI training tool
**Performance Goals**: Arena evaluation < 3 minutes for 40 games with 25 MCTS simulations
**Constraints**: Single-machine training, no distributed compute
**Scale/Scope**: 6x6 Othello board, single trainer user

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution is not configured (template only). No gates to enforce. Proceeding.

## Project Structure

### Documentation (this feature)

```text
specs/001-arena-model-selection/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
# Files to modify:
train.py                 # Add Arena class, integrate into training loop
main.py                  # Add arena CLI parameters

# Files to add:
tests/test_arena.py      # Arena unit tests

# Existing files used (no changes):
mcts/mcts.py             # MCTS for arena match play
env/othello.py           # Game environment
models/neural_network.py # AlphaZeroNetwork
play.py                  # AlphaZeroPlayer (reference for match play pattern)

# Model files:
models/best.pt           # Best model checkpoint (new)
models/temp.pt           # Temporary candidate checkpoint (new, transient)
models/checkpoint_N.pt   # Numbered history checkpoints (existing pattern)
```

**Structure Decision**: Minimal change approach — Arena class is added directly to `train.py` since it is tightly coupled with the training loop and uses the same model/MCTS infrastructure. No new modules needed.

## Implementation Tasks

### Task 1: Arena Class (P1 — Core)

Add an `Arena` class to `train.py` that plays matches between two models.

**Inputs**: Two model state dicts, game size, number of games, MCTS simulations
**Outputs**: (new_wins, old_wins, draws)

Logic:
- For each game: create fresh env, create MCTS for each model
- First half: new model plays Black, old model plays White
- Second half: swap colors
- Play game to completion, record winner
- Return aggregate results

### Task 2: Integrate Arena into Training Loop (P1 — Core)

Modify `AlphaZeroTrainer.train()` method:

1. At start: check if `best.pt` exists → load as best model; if not, first iteration auto-accepts
2. After `train_network()`: save candidate as `temp.pt`
3. Run Arena: candidate vs best, 40 games, 25 MCTS sims
4. If `new_wins / (new_wins + old_wins) >= 0.6`:
   - ACCEPT: copy `temp.pt` → `best.pt`, also save as `checkpoint_N.pt`
   - Reload best model weights into trainer
5. If threshold not met:
   - REJECT: delete `temp.pt`, reload `best.pt` weights into trainer
6. Log results: wins, losses, draws, win rate, decision

### Task 3: CLI Parameters (P2)

Add to `main.py` train subparser:
- `--arena_games` (default 40)
- `--arena_threshold` (default 0.6)
- `--arena_mcts_simulations` (default 25)

Pass through to `AlphaZeroTrainer.__init__()`.

### Task 4: Tests (P2)

Create `tests/test_arena.py`:
- Test Arena plays correct number of games with color swap
- Test model accepted when win rate >= 60%
- Test model rejected when win rate < 60%
- Test all-draws results in rejection
- Test first iteration auto-accepts
- Test temp.pt lifecycle (created before arena, deleted/promoted after)

### Task 5: Cleanup Old Checkpoints (P3)

Delete all existing `models/checkpoint_*.pt` files before retraining. This is a manual step documented in the plan, not automated.

## Complexity Tracking

No constitution violations to justify.
