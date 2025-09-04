# AlphaZero Bug Fix Branch

This branch (`bugfix-fresh-training`) contains all critical bug fixes for the AlphaZero implementation.

## 🐛 **Bugs Fixed**

### 1. **Training Value Calculation Bug** (Critical)
- **File**: `train.py` 
- **Issue**: Used `winner * player_at_move` instead of proper win/loss labeling
- **Fix**: Winners get `1.0`, losers get `-1.0`, draws get `0.0`
- **Impact**: Was causing inverted learning signals during training

### 2. **Environment Reward Calculation Bug** (Critical)
- **File**: `env/othello.py` - `OthelloEnv.step()`
- **Issue**: Used `current_player` after move (which switches player) for reward calculation
- **Fix**: Store `player_who_moved` before move execution
- **Impact**: Rewards were backwards, severely affecting evaluation

### 3. **Pass Reward Calculation Bug** (Critical)
- **File**: `env/othello.py` - `OthelloEnv.step()` pass logic
- **Issue**: `pass_turn()` switches current_player, corrupting `player_who_moved` tracking
- **Fix**: Preserve actual pass player before calling `pass_turn()`
- **Impact**: Wrong rewards for pass actions, further corrupting training signal

### 4. **State Representation Inconsistency** (Major)
- **Files**: `play.py`, `gui_play.py`
- **Issue**: Evaluation used `get_observation()` while training used canonical states
- **Fix**: Unified to use canonical states + `_canonical_to_observation()` conversion
- **Impact**: Training and evaluation used different state formats

## 📊 **Performance Before Fixes**
- Checkpoint 10 vs random: **5% win rate** → **3.3% win rate** (after partial fixes)
- Checkpoint 20 vs random: **0% win rate**
- Later checkpoints performed worse than earlier ones due to accumulating corrupted training data

## 🎯 **Expected Performance After Fixes**
With all bugs fixed, fresh training should show:
- Significantly higher win rates against random players
- Consistent improvement across training iterations
- Later checkpoints outperforming earlier ones

## 🚀 **Recommended Fresh Training Command**
```bash
python main.py train --num_iterations 50 --self_play_games 200 --mcts_simulations 256 --use_multiprocessing --mp_num_workers 6 --mp_games_per_worker 5 --use_mps
```

## 📝 **Commit History**
- `c59e5f8`: Fix critical pass reward calculation bug
- `bfd0eb1`: Fix critical hidden bugs in AlphaZero implementation  
- `d9170e0`: Fix critical AlphaZero training bugs

## ⚠️ **Important Notes**
- All previous checkpoints were trained with corrupted reward signals
- Fresh training from scratch is recommended to see true performance
- This branch contains the definitive bug-fixed version of the code

---
*Created: 2025年9月4日*
*Branch Purpose: Clean slate for bug-fixed AlphaZero training*