"""MCTS (with tree reuse) vs Random — verifies tree-reuse impact.

Pure-policy baseline already measured 6.7% (mode collapse). With the new
tree-reuse code path, a 32-move game accumulates ~N*32 simulations in the
tree rather than N per move in isolation. If this recovers win rate, we
confirm the root-cause hypothesis.

Usage:
    python scripts/exp_mcts_reuse_vs_random.py --checkpoint models/best.pt --sims 50 --games 30
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS
from play import RandomPlayer


def load_model(path, device, game_size=6):
    model = AlphaZeroNetwork(game_size=game_size, device=device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def play_games(model, num_games, sims, c_puct, device, game_size=6, seed=42):
    np.random.seed(seed)
    rnd = RandomPlayer()
    mcts = MCTS(model, c_puct=c_puct, num_simulations=sims)
    half = num_games // 2
    w = l = d = 0
    first_moves = {}

    for g in range(num_games):
        env = OthelloEnv(size=game_size)
        env.reset()
        mcts.reset_tree()
        model_pid = -1 if g < half else 1
        move_count = 0

        while not env.board.is_done():
            if env.board.current_player == model_pid:
                canonical = env.board.get_canonical_state()
                action_probs = mcts.search(canonical, env, temperature=0, add_noise=False)
                a = int(np.argmax(action_probs))
                mask = env.get_valid_moves_mask()
                if mask[a] == 0:
                    valid = np.where(mask == 1)[0]
                    a = int(valid[0]) if len(valid) else env.board.get_action_space_size() - 1
                if move_count == 0 and model_pid == -1:
                    first_moves[a] = first_moves.get(a, 0) + 1
            else:
                a = rnd.get_action(env)
            env.step(a)
            mcts.advance_root(a)
            move_count += 1

        winner = env.board.get_winner()
        if winner == 0:
            d += 1
        elif winner == model_pid:
            w += 1
        else:
            l += 1

    return w, l, d, first_moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--c_puct", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    w, l, d, hist = play_games(
        model, args.games, args.sims, args.c_puct, device, seed=args.seed
    )
    label = os.path.basename(args.checkpoint)
    wr = w / args.games * 100
    print(f"[{label}] MCTS(sims={args.sims}, reuse=ON) vs Random: {wr:.1f}% ({w}W / {l}L / {d}D)")
    if hist:
        top = sorted(hist.items(), key=lambda x: -x[1])[:5]
        print(f"  First-move distribution (as Black): {top}")


if __name__ == "__main__":
    main()
