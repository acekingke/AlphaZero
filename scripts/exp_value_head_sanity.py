"""Sanity-check the value head on random game trajectories.

Goal: diagnose whether value head is badly calibrated (inverted sign, always
negative, etc.). For each state in a random game, record the network's value
prediction and compare to the ground-truth final outcome.

If value(state) has a consistent sign mismatch with the eventual outcome
from current_player's perspective, we found the bug.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
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


@torch.no_grad()
def predict_value(model, canonical, device, game_size=6):
    board = torch.FloatTensor(canonical.astype(np.float32)).view(1, game_size, game_size).to(device)
    _, v = model(board)
    return float(v.cpu().numpy()[0][0])


def analyze(model, num_games=20, device="cpu", game_size=6, seed=0):
    np.random.seed(seed)
    rnd = RandomPlayer()
    agree = 0   # sign(value) matches sign(final outcome from current player's view)
    disagree = 0
    zero_outcome = 0
    value_sum = 0.0
    value_count = 0

    for g in range(num_games):
        env = OthelloEnv(size=game_size)
        env.reset()
        history = []  # list of (canonical, current_player)

        # Random game trajectory
        while not env.board.is_done():
            canonical = env.board.get_canonical_state().copy()
            history.append((canonical, env.board.current_player))
            a = rnd.get_action(env)
            env.step(a)

        winner = env.board.get_winner()

        # For each recorded state, check value prediction vs ground truth
        for canonical, player in history:
            v_pred = predict_value(model, canonical, device, game_size)
            value_sum += v_pred
            value_count += 1
            if winner == 0:
                zero_outcome += 1
                continue
            ground_truth = 1.0 if winner == player else -1.0
            if v_pred * ground_truth > 0:
                agree += 1
            elif v_pred * ground_truth < 0:
                disagree += 1

    print(f"Value head sanity over {num_games} random games, {value_count} state-predictions:")
    print(f"  Mean value output: {value_sum / max(1, value_count):+.4f}  (ideal: near 0 for diverse states)")
    print(f"  Sign agrees with outcome: {agree}  ({agree / max(1, agree+disagree) * 100:.1f}%)")
    print(f"  Sign disagrees: {disagree}  ({disagree / max(1, agree+disagree) * 100:.1f}%)")
    print(f"  Draws (skipped): {zero_outcome}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model = load_model(args.checkpoint, "cpu")
    print(f"Checkpoint: {args.checkpoint}")
    analyze(model, num_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
