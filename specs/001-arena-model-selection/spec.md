# Feature Specification: Arena-Based Model Selection

**Feature Branch**: `001-arena-model-selection`  
**Created**: 2026-04-10  
**Status**: Draft  
**Input**: User description: "训练方式 使用Arena 对战，胜率>=60% 才接受新模型"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Automatic Model Quality Gate (Priority: P1)

As a trainer, when I run the training loop, the system automatically evaluates each newly trained model against the current best model through competitive matches. Only models that demonstrate a significant improvement (winning 60% or more of matches) replace the current best. This prevents training regression where later iterations produce weaker models.

**Why this priority**: This is the core feature. Without it, every training iteration overwrites the previous model regardless of quality, which has historically caused win rates to drop from 45% to 5%.

**Independent Test**: Can be fully tested by running a training session and verifying that weak models are rejected while strong models are accepted, and that a `best.pt` file is always maintained.

**Acceptance Scenarios**:

1. **Given** a trained best model exists, **When** a new model wins 65% of arena matches, **Then** the new model replaces the best model and is saved as `best.pt`
2. **Given** a trained best model exists, **When** a new model wins only 40% of arena matches, **Then** the new model is rejected and the best model remains unchanged
3. **Given** no best model exists (first iteration), **When** training begins, **Then** the first trained model is automatically accepted as the best model

---

### User Story 2 - Arena Match Fairness (Priority: P2)

As a trainer, the arena evaluation must be fair. Each model plays an equal number of games as both the first and second player to eliminate positional advantage bias.

**Why this priority**: Unfair evaluation would lead to incorrect model selection decisions, undermining the entire quality gate.

**Independent Test**: Can be tested by running an arena evaluation and verifying that each model plays exactly half the games as first player and half as second player.

**Acceptance Scenarios**:

1. **Given** 40 arena games are configured, **When** arena evaluation runs, **Then** the new model plays 20 games as Black (first player) and 20 games as White (second player)
2. **Given** an arena match completes, **When** results are tallied, **Then** draws are excluded from the win rate calculation

---

### User Story 3 - Training Progress Visibility (Priority: P3)

As a trainer, I can see the arena evaluation results after each iteration, including the number of wins, losses, draws, and whether the model was accepted or rejected.

**Why this priority**: Visibility into model selection decisions helps the trainer understand training dynamics and tune parameters.

**Independent Test**: Can be tested by running a training iteration and observing the console output for arena results.

**Acceptance Scenarios**:

1. **Given** an arena evaluation completes, **When** results are displayed, **Then** the output shows wins, losses, draws, win rate percentage, and the accept/reject decision
2. **Given** training runs for multiple iterations, **When** a model is rejected, **Then** the log clearly indicates the model was rejected and the previous best was retained

---

### Edge Cases

- What happens when the arena produces only draws (0 wins on both sides)? The new model is rejected (conservative approach).
- What happens when training starts fresh with no existing best model? The first trained model is automatically accepted.
- What happens if all old checkpoints are deleted before starting? Training starts from scratch with a randomly initialized model.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST evaluate each newly trained model against the current best model through competitive matches before accepting it
- **FR-002**: System MUST use a configurable win rate threshold (default 60%) to determine model acceptance
- **FR-003**: System MUST play arena matches with player positions swapped (half as Black, half as White) to ensure fairness
- **FR-004**: System MUST exclude draws from the win rate calculation (win rate = new_wins / (new_wins + old_wins))
- **FR-005**: System MUST maintain a persistent `best.pt` checkpoint that always contains the strongest accepted model
- **FR-006**: System MUST reject the new model and revert to the best model when the win rate threshold is not met
- **FR-007**: System MUST accept the first trained model automatically when no best model exists
- **FR-008**: System MUST support a configurable number of arena games (default 40)
- **FR-009**: System MUST display arena results (wins, losses, draws, win rate, accept/reject decision) after each evaluation
- **FR-010**: System MUST save accepted models as both `best.pt` and `checkpoint_N.pt` (for history)

### Key Entities

- **Best Model**: The strongest model found so far, persisted as `best.pt`. Serves as the opponent in all arena evaluations.
- **Candidate Model**: The newly trained model from the current iteration. Must beat the best model to be accepted.
- **Arena Match**: A competitive game between two models using their respective search strategies. Each match has a winner, loser, or draw result.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Over a 10-iteration training run, the best model's win rate against a random player never decreases between accepted checkpoints
- **SC-002**: The model accepted after 10+ iterations achieves at least 50% win rate against a random opponent (up from the current 2%)
- **SC-003**: At least 30% of training iterations result in model rejection, demonstrating that the quality gate is actively filtering
- **SC-004**: Arena evaluation completes within 5 minutes per iteration (for 40 games with 50 MCTS simulations)

## Assumptions

- The existing MCTS and self-play infrastructure will be reused for arena matches
- Arena matches use the same board size and rules as training
- The trainer is willing to accept longer training times due to arena evaluation overhead (approximately 5 minutes per iteration)
- Old numbered checkpoints (checkpoint_0.pt through checkpoint_27.pt) can be deleted before retraining
- Default arena configuration (40 games, 60% threshold) follows the alpha-zero-general reference implementation
