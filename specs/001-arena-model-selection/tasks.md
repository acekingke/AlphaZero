# Tasks: Arena-Based Model Selection

**Input**: Design documents from `/specs/001-arena-model-selection/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: Included — arena evaluation correctness is critical and must be verified.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Clean slate and parameter wiring

- [ ] T001 Delete all existing model checkpoints in models/checkpoint_*.pt
- [ ] T002 Add arena parameters to AlphaZeroTrainer.__init__() in train.py: arena_games (default 40), arena_threshold (default 0.6), arena_mcts_simulations (default 25)
- [ ] T003 Add CLI arguments --arena_games, --arena_threshold, --arena_mcts_simulations to train subparser in main.py and pass to AlphaZeroTrainer

**Checkpoint**: CLI wired, parameters flow from command line to trainer

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Arena class that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Implement Arena class in train.py with method play_games(model1_state_dict, model2_state_dict) that returns (model1_wins, model2_wins, draws). Must: create two AlphaZeroNetwork instances on CPU, create MCTS with arena_mcts_simulations, play arena_games total (half with model1 as Black, half as White), use temperature=0 for deterministic play
- [ ] T005 Add Arena.play_single_game(env, mcts1, mcts2, model1_is_black) helper that plays one game to completion and returns winner

**Checkpoint**: Arena can play matches between two models and return results

---

## Phase 3: User Story 1 - Automatic Model Quality Gate (Priority: P1) 🎯 MVP

**Goal**: Training loop evaluates new models against best model; accepts only if win rate >= 60%

**Independent Test**: Run training for 3 iterations; verify best.pt is maintained and weak models are rejected

### Tests for User Story 1

- [ ] T006 [P] [US1] Test arena plays correct total games (20 as each color) in tests/test_arena.py
- [ ] T007 [P] [US1] Test model accepted when new_wins/(new_wins+old_wins) >= 0.6 in tests/test_arena.py
- [ ] T008 [P] [US1] Test model rejected when win rate < 0.6 in tests/test_arena.py
- [ ] T009 [P] [US1] Test all-draws results in rejection (conservative) in tests/test_arena.py
- [ ] T010 [P] [US1] Test first iteration auto-accepts when no best.pt exists in tests/test_arena.py

### Implementation for User Story 1

- [ ] T011 [US1] Modify AlphaZeroTrainer.train() in train.py: at start, check if models/best.pt exists; if yes, load as best model state dict; if no, set best_model_state = None
- [ ] T012 [US1] Modify AlphaZeroTrainer.train() in train.py: after train_network(), save candidate as models/temp.pt
- [ ] T013 [US1] Modify AlphaZeroTrainer.train() in train.py: if best_model_state is None (first iteration), auto-accept — copy temp.pt to best.pt and checkpoint_N.pt, set best_model_state
- [ ] T014 [US1] Modify AlphaZeroTrainer.train() in train.py: if best_model_state exists, run Arena.play_games(candidate_state, best_state); compute win_rate = new_wins/(new_wins+old_wins) with 0.0 if denominator is 0
- [ ] T015 [US1] Modify AlphaZeroTrainer.train() in train.py: if win_rate >= threshold, ACCEPT — promote temp.pt to best.pt, save checkpoint_N.pt, update best_model_state; else REJECT — delete temp.pt, reload best.pt weights into self.model
- [ ] T016 [US1] Add accept/reject logging: print wins, losses, draws, win rate, and ACCEPTING/REJECTING decision

**Checkpoint**: Training loop with full arena quality gate. Run 3 iterations and verify best.pt is maintained correctly.

---

## Phase 4: User Story 2 - Arena Match Fairness (Priority: P2)

**Goal**: Arena matches swap colors so each model plays equal games as Black and White

**Independent Test**: Run arena and verify game count split is exactly 50/50 by color

### Implementation for User Story 2

- [ ] T017 [US2] Verify Arena.play_games() in train.py splits games evenly: first half model1=Black, second half model1=White (already designed in T004; this task validates and adds assertion logging)
- [ ] T018 [US2] Verify win rate calculation excludes draws: win_rate = new_wins/(new_wins+old_wins), not total games (already designed in T014; this task adds explicit test)

**Checkpoint**: Arena fairness verified — color swap and draw exclusion confirmed

---

## Phase 5: User Story 3 - Training Progress Visibility (Priority: P3)

**Goal**: Console output shows arena results clearly after each iteration

**Independent Test**: Run training and observe console output for arena results

### Implementation for User Story 3

- [ ] T019 [US3] Enhance arena logging in train.py: after each arena evaluation, print formatted summary: "Arena: New model wins: X, Best model wins: Y, Draws: Z, Win rate: XX.X% → ACCEPTED/REJECTED"
- [ ] T020 [US3] Add iteration summary in train.py: at end of each iteration, print whether this iteration's model was accepted or rejected, and total accepted/rejected count so far

**Checkpoint**: Clear, readable training output showing arena decisions

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup

- [ ] T021 Run full test suite (pytest tests/) to verify no regressions
- [ ] T022 Run training for 5+ iterations and verify best.pt win rate against random improves or stays stable
- [ ] T023 Update quickstart.md in specs/001-arena-model-selection/quickstart.md with actual verified commands

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on T002 (parameters exist)
- **User Story 1 (Phase 3)**: Depends on Phase 2 (Arena class exists)
- **User Story 2 (Phase 4)**: Depends on Phase 3 (quality gate exists to verify fairness)
- **User Story 3 (Phase 5)**: Depends on Phase 3 (logging needs arena results)
- **Polish (Phase 6)**: Depends on all story phases

### Within Each User Story

- Tests (T006-T010) can all run in parallel [P]
- Implementation tasks (T011-T016) are sequential (each builds on prior)

### Parallel Opportunities

- T006, T007, T008, T009, T010 — all arena tests can be written in parallel
- T002 and T003 — parameter setup in train.py and main.py can be done in parallel
- T019 and T020 — logging enhancements are independent

---

## Parallel Example: User Story 1 Tests

```bash
# Launch all US1 tests together:
Task: "Test arena plays correct total games in tests/test_arena.py"
Task: "Test model accepted when win rate >= 0.6 in tests/test_arena.py"
Task: "Test model rejected when win rate < 0.6 in tests/test_arena.py"
Task: "Test all-draws results in rejection in tests/test_arena.py"
Task: "Test first iteration auto-accepts in tests/test_arena.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational Arena class (T004-T005)
3. Complete Phase 3: US1 tests + implementation (T006-T016)
4. **STOP and VALIDATE**: Run `pytest tests/test_arena.py` and train for 3 iterations
5. Verify best.pt exists and weak models are rejected

### Full Delivery

1. MVP (above)
2. Add US2: Verify fairness (T017-T018)
3. Add US3: Enhance logging (T019-T020)
4. Polish: Full validation (T021-T023)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Arena uses CPU for evaluation (avoid GPU memory contention)
- Temperature=0 for deterministic arena play
- Commit after each phase completion
