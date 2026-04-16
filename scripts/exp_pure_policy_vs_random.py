"""Pure-policy (no MCTS) vs Random — diagnostic experiment.

Strips MCTS entirely: loads a checkpoint, calls the policy head once per
move, argmax over the masked policy, plays against RandomPlayer.

Goal: separate "network learned nothing" from "25-sim MCTS can't fix
the prior". If raw policy already beats random, the 6% vs_random is
a search-budget problem. If raw policy also tanks, the network itself
is the root cause.
"""
import os
import sys
import argparse
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
def policy_action(model, env, device):
    canonical = env.board.get_canonical_state()
    board = torch.FloatTensor(canonical.astype(np.float32)).view(1, env.board.size, env.board.size).to(device)
    log_pi, _ = model(board)
    policy = torch.exp(log_pi).cpu().numpy()[0]
    mask = env.get_valid_moves_mask()
    masked = policy * mask
    if masked.sum() <= 0:
        valid = np.where(mask == 1)[0]
        return int(valid[0]) if len(valid) else env.board.get_action_space_size() - 1
    return int(np.argmax(masked))


def play_games(model, num_games, device, game_size=6, seed=42):
    np.random.seed(seed)
    rnd = RandomPlayer()
    half = num_games // 2
    w = l = d = 0
    first_move_histogram = {}

    for g in range(num_games):
        env = OthelloEnv(size=game_size)
        env.reset()
        model_pid = -1 if g < half else 1
        move_count = 0

        while not env.board.is_done():
            if env.board.current_player == model_pid:
                a = policy_action(model, env, device)
                if move_count == 0 and model_pid == -1:
                    first_move_histogram[a] = first_move_histogram.get(a, 0) + 1
            else:
                a = rnd.get_action(env)
            env.step(a)
            move_count += 1

        winner = env.board.get_winner()
        if winner == 0:
            d += 1
        elif winner == model_pid:
            w += 1
        else:
            l += 1

    return w, l, d, first_move_histogram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--device", default="cpu", help="cpu to avoid contending with training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    w, l, d, hist = play_games(model, args.games, device, seed=args.seed)

    label = os.path.basename(args.checkpoint)
    wr = w / args.games * 100
    print(f"[{label}] Pure-policy vs Random: {wr:.1f}% ({w}W / {l}L / {d}D) over {args.games} games")
    if hist:
        top = sorted(hist.items(), key=lambda x: -x[1])[:5]
        print(f"  First-move distribution (as Black): {top}")


if __name__ == "__main__":
    main()
