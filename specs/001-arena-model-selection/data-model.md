# Data Model: Arena-Based Model Selection

## Entities

### ArenaResult

Represents the outcome of a complete arena evaluation between two models.

| Field       | Type  | Description                                    |
|-------------|-------|------------------------------------------------|
| new_wins    | int   | Games won by the candidate (new) model         |
| old_wins    | int   | Games won by the best (old) model              |
| draws       | int   | Games ending in a draw                         |
| win_rate    | float | new_wins / (new_wins + old_wins), 0.0 if no decisive games |
| accepted    | bool  | True if win_rate >= threshold                  |

### Checkpoint Lifecycle

```
[training iteration completes]
        │
        ▼
   Save as temp.pt
        │
        ▼
   Arena evaluation (40 games)
        │
   ┌────┴────┐
   │         │
win_rate   win_rate
 >= 0.6    < 0.6
   │         │
   ▼         ▼
ACCEPT     REJECT
   │         │
   ├─ temp.pt → best.pt    ├─ delete temp.pt
   ├─ save checkpoint_N.pt  └─ reload best.pt weights
   └─ log ACCEPTING
```

### File Artifacts

| File                  | Lifecycle              | Purpose                              |
|-----------------------|------------------------|--------------------------------------|
| `models/best.pt`     | Persistent             | Current strongest model              |
| `models/temp.pt`     | Transient (per iteration) | Candidate model during arena eval |
| `models/checkpoint_N.pt` | Persistent (accepted only) | Historical record of accepted models |
