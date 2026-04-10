# Quickstart: Arena-Based Model Selection

## Training with Arena

```bash
# Clean start (delete old checkpoints first)
rm models/checkpoint_*.pt

# Train with arena evaluation (default: 40 games, 60% threshold)
python main.py train \
  --num_iterations 10 \
  --self_play_games 100 \
  --mcts_simulations 50 \
  --arena_games 40 \
  --arena_threshold 0.6 \
  --arena_mcts_simulations 25 \
  --use_mps

# The best model is always at models/best.pt
```

## Evaluating the Best Model

```bash
python main.py evaluate --model ./models/best.pt --num_games 50 --mcts_simulations 50
```

## Playing Against the Best Model

```bash
python gui_play.py --model ./models/best.pt
```

## Key Files

- `models/best.pt` — Always the strongest accepted model
- `models/checkpoint_N.pt` — Historical record of accepted models (N = iteration)
- `models/temp.pt` — Temporary; only exists during arena evaluation
