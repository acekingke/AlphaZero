import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS
from utils.device import get_device
from utils.training_logger import TrainingLogger
from utils.data_augmentation import get_all_symmetries


# Helper functions for multiprocessing - must be at module level to be picklable
def _worker_init(
    state_dict,
    game_size,
    num_mcts_simulations,
    c_puct,
    temperature,
    dirichlet_alpha,
    dirichlet_weight,
    temp_threshold,
):
    """Worker process initializer that runs once per worker process."""
    import torch
    from models.neural_network import AlphaZeroNetwork
    from mcts.mcts import MCTS as LocalMCTS

    # Reduce intra-process thread usage
    torch.set_num_threads(1)

    global WORKER_MODEL
    global WORKER_MCTS_CLASS
    global WORKER_GAME_SIZE
    global WORKER_MCTS_SIMULATIONS
    global WORKER_C_PUCT
    global WORKER_TEMPERATURE
    global WORKER_DIRICHLET_ALPHA
    global WORKER_DIRICHLET_WEIGHT
    global WORKER_TEMP_THRESHOLD

    WORKER_MODEL = AlphaZeroNetwork(game_size, device="cpu")
    WORKER_MODEL.load_state_dict(state_dict)
    WORKER_MODEL.to("cpu")
    WORKER_MODEL.eval()

    WORKER_MCTS_CLASS = LocalMCTS
    WORKER_GAME_SIZE = game_size
    WORKER_MCTS_SIMULATIONS = num_mcts_simulations
    WORKER_C_PUCT = c_puct
    WORKER_TEMPERATURE = temperature
    WORKER_DIRICHLET_ALPHA = dirichlet_alpha
    WORKER_DIRICHLET_WEIGHT = dirichlet_weight
    WORKER_TEMP_THRESHOLD = temp_threshold


def _worker_play(seed_and_count):
    """Play `count` games using global worker model. seed_and_count is (seed, count)."""
    import random
    import numpy as np
    import time
    from env.othello import OthelloEnv

    seed, count = seed_and_count
    results = []
    game_stats = []

    worker_id = mp.current_process().name

    for i in range(count):
        s = seed + i
        random.seed(s)
        np.random.seed(s + 1)

        env = OthelloEnv(size=WORKER_GAME_SIZE)
        env.reset()

        # Randomly switch first player to balance training data
        if random.random() < 0.5:
            # Switch to white first by making black pass and white start
            env.board.current_player = 1  # White goes first

        mcts = WORKER_MCTS_CLASS(
            WORKER_MODEL,
            c_puct=WORKER_C_PUCT,
            num_simulations=WORKER_MCTS_SIMULATIONS,
            dirichlet_alpha=WORKER_DIRICHLET_ALPHA,
            dirichlet_weight=WORKER_DIRICHLET_WEIGHT,
        )
        game_history = []
        step = 0
        game_start_time = time.time()

        while not env.board.is_done():
            # Segmented temperature: explore early, exploit late.
            # Mirrors alpha-zero-general's tempThreshold mechanism.
            current_temp = WORKER_TEMPERATURE if step < WORKER_TEMP_THRESHOLD else 0.0

            # Use canonical state for consistency with MCTS
            canonical_state = env.board.get_canonical_state()
            # Convert to observation format for neural network (consistent with MCTS)
            state = mcts.canonical_to_observation(canonical_state, env)
            action_probs = mcts.search(state, env, current_temp, add_noise=True)
            # Save state, the pi (policy) returned by MCTS, and the player who made the move
            game_history.append((state, action_probs, env.board.current_player))
            action = np.random.choice(len(action_probs), p=action_probs)
            _, reward, done, _ = env.step(action)
            step += 1
            if done:
                break

        game_duration = time.time() - game_start_time
        winner = env.board.get_winner()
        outcome = "Draw" if winner == 0 else "Win" if winner == 1 else "Loss"

        # Collect game statistics
        game_stats.append(
            {
                "moves": step,
                "duration": game_duration,
                "outcome": outcome,
            }
        )

        for state, policy, player_at_move in game_history:
            # state 已经是 get_observation() 格式，无需转换
            # 正确计算value标签：获胜者=1，失败者=-1，平局=0
            if winner == 0:  # 平局
                value_for_training = 0.0
            else:
                # 如果当前玩家是获胜者，value=1；否则value=-1
                value_for_training = 1.0 if winner == player_at_move else -1.0

            results.append((state, policy, value_for_training))
            # 数据增强
            for sym_state, sym_policy in get_all_symmetries(
                state, policy, WORKER_GAME_SIZE
            ):
                results.append((sym_state, sym_policy, value_for_training))

    # Return results, task metadata, and game statistics
    return results, seed, count, game_stats, worker_id


class Arena:
    """Arena for evaluating two models against each other."""

    def __init__(self, game_size=6, num_games=40, mcts_simulations=25, c_puct=2.0):
        self.game_size = game_size
        self.num_games = num_games
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct

    def play_single_game(self, env, mcts1, mcts2, model1_is_black):
        """Play a single game between two MCTS agents.

        Args:
            env: OthelloEnv instance (already reset)
            mcts1: MCTS instance for model 1
            mcts2: MCTS instance for model 2
            model1_is_black: if True, model1 plays as Black (-1)

        Returns:
            1 if model1 wins, 2 if model2 wins, 0 if draw
        """
        # Black is -1, White is 1
        if model1_is_black:
            player_map = {-1: mcts1, 1: mcts2}
        else:
            player_map = {-1: mcts2, 1: mcts1}

        while not env.board.is_done():
            current_player = env.board.current_player
            mcts = player_map[current_player]

            canonical_state = env.board.get_canonical_state()
            state = mcts.canonical_to_observation(canonical_state, env)
            # temperature=0 triggers random tie-breaking inside MCTS
            # (see mcts/mcts.py _get_action_probabilities).
            action_probs = mcts.search(state, env, temperature=0, add_noise=False)
            action = int(np.argmax(action_probs))  # action_probs is already one-hot after tie-break

            _, _, done, _ = env.step(action)
            if done:
                break

        winner = env.board.get_winner()
        if winner == 0:
            return 0  # draw

        # Map board winner to model number
        if model1_is_black:
            return 1 if winner == -1 else 2
        else:
            return 1 if winner == 1 else 2

    def play_games(self, model1_state_dict, model2_state_dict):
        """Play arena games between two models.

        Args:
            model1_state_dict: state dict for model 1 (candidate)
            model2_state_dict: state dict for model 2 (best)

        Returns:
            (model1_wins, model2_wins, draws)
        """
        model1 = AlphaZeroNetwork(self.game_size, device="cpu")
        model1.load_state_dict(model1_state_dict)
        model1.to("cpu")
        model1.eval()

        model2 = AlphaZeroNetwork(self.game_size, device="cpu")
        model2.load_state_dict(model2_state_dict)
        model2.to("cpu")
        model2.eval()

        mcts1 = MCTS(model1, c_puct=self.c_puct, num_simulations=self.mcts_simulations)
        mcts2 = MCTS(model2, c_puct=self.c_puct, num_simulations=self.mcts_simulations)

        model1_wins = 0
        model2_wins = 0
        draws = 0

        half = self.num_games // 2

        for game_idx in range(self.num_games):
            env = OthelloEnv(size=self.game_size)
            env.reset()

            # First half: model1 plays as Black; second half: model1 plays as White
            model1_is_black = game_idx < half

            result = self.play_single_game(env, mcts1, mcts2, model1_is_black)
            if result == 1:
                model1_wins += 1
            elif result == 2:
                model2_wins += 1
            else:
                draws += 1

        return model1_wins, model2_wins, draws


class AlphaZeroTrainer:
    """Trainer for AlphaZero."""

    def __init__(
        self,
        game_size=6,
        num_iterations=100,
        num_self_play_games=1000,  # Increased from 200 to 1000 for better training
        num_mcts_simulations=25,
        temperature=0.8,
        c_puct=2.0,
        batch_size=64,
        num_epochs=2,
        lr=0.001,
        checkpoint_path="./models/checkpoint",
        use_mps=True,
        use_cuda=True,
        log_dir="./logs",
        use_multiprocessing=False,
        mp_num_workers=None,
        mp_games_per_worker=1,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        arena_games=40,
        arena_threshold=0.6,
        arena_mcts_simulations=25,
        temp_threshold=15,
        eval_vs_random_interval=5,
        eval_vs_random_games=30,
        lr_decay_factor=0.98,
        lr_min=1e-5,
    ):
        self.game_size = game_size
        self.num_iterations = num_iterations
        self.num_self_play_games = num_self_play_games
        self.num_mcts_simulations = num_mcts_simulations
        self.temperature = temperature
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        self.log_dir = log_dir

        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Validate critical parameters that would cause crashes or
        # incorrect behavior if misconfigured.
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
        if eval_vs_random_interval < 1:
            raise ValueError(
                f"eval_vs_random_interval must be >= 1, got {eval_vs_random_interval}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if arena_games < 2 or arena_games % 2 != 0:
            raise ValueError(
                f"arena_games must be even and >= 2, got {arena_games}"
            )

        # Get the best available device (CUDA, MPS, or CPU)
        self.device = get_device(use_mps=use_mps, use_cuda=use_cuda)
        print(f"Training will use: {self.device}")

        # Initialize neural network
        self.model = AlphaZeroNetwork(game_size, device=self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Sliding window training data (per-iteration history).
        # Keeps the most recent N iterations of self-play data, discarding older
        # ones to avoid training on stale labels from weak earlier models.
        # Reference: AlphaGo Zero (Nature 2017) Methods - "most recent 500K games"
        self.train_examples_history = []  # list of lists, one per iteration
        self.num_iters_for_history = 20

        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []

        # Initialize training logger
        self.logger = TrainingLogger(log_dir=log_dir, model_name="alphazero")

        # Multiprocessing options for self-play
        self.use_multiprocessing = use_multiprocessing
        self.mp_num_workers = mp_num_workers
        self.mp_games_per_worker = mp_games_per_worker

        # Dirichlet noise parameters for MCTS root mixing
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        # Segmented temperature: temp=temperature for first temp_threshold moves, then temp=0
        self.temp_threshold = temp_threshold

        # External baseline evaluation (vs random) to guard against
        # non-transitive strategy cycles that Arena alone cannot detect.
        # vs_random_best.pt stores the model with the highest vs-random win
        # rate ever seen. NOTE: this is independent of Arena; it may contain
        # a model that was rejected by Arena (because Arena compares to the
        # previous best, not to any fixed external baseline).
        self.eval_vs_random_interval = eval_vs_random_interval
        self.eval_vs_random_games = eval_vs_random_games
        self.global_best_win_rate = 0.0

        # LR decay: multiply LR by this factor per AlphaZero iteration.
        # Applied BETWEEN iterations (not within batches), so Adam's
        # momentum state stays consistent with the LR within each call.
        self.lr_decay_factor = lr_decay_factor
        self.lr_min = lr_min
        self._iteration_count = 0

        # Arena parameters
        self.arena_games = arena_games
        self.arena_threshold = arena_threshold
        self.arena_mcts_simulations = arena_mcts_simulations

    def self_play(self):
        """
        Perform self-play to generate training data.

        Returns:
            List of (state, policy, value) tuples from the self-play games
        """
        examples = []

        # Default: serial execution (existing behavior)
        for game_idx in tqdm(range(self.num_self_play_games), desc="Self-Play Games"):
            env = OthelloEnv(size=self.game_size)
            env.reset()

            # Randomly switch first player to balance training data
            if np.random.random() < 0.5:
                # Switch to white first
                env.board.current_player = 1  # White goes first

            mcts = MCTS(
                self.model,
                c_puct=self.c_puct,
                num_simulations=self.num_mcts_simulations,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_weight=self.dirichlet_weight,
            )
            game_history = []
            step = 0

            # Segmented temperature: temp=self.temperature for first temp_threshold moves, then temp=0
            while not env.board.is_done():
                current_temp = self.temperature if step < self.temp_threshold else 0.0

                # Use canonical state for consistency with MCTS
                canonical_state = env.board.get_canonical_state()
                # Convert to observation format for neural network (consistent with MCTS)
                state = mcts.canonical_to_observation(canonical_state, env)
                action_probs = mcts.search(state, env, current_temp, add_noise=True)

                # Save state, the pi (policy) returned by MCTS, and the player who made the move
                # This ensures value labels are computed from the correct player's perspective
                game_history.append((state, action_probs, env.board.current_player))

                # Choose action based on the MCTS probabilities
                action = np.random.choice(len(action_probs), p=action_probs)
                _, reward, done, _ = env.step(action)

                step += 1

                # Break if game is over
                if done:
                    break

            # Get the final result of the game
            winner = env.board.get_winner()

            # state 已经是 get_observation() 格式，无需转换
            # 正确计算value标签：获胜者=1，失败者=-1，平局=0
            for state, policy, player_at_move in game_history:
                if winner == 0:  # 平局
                    value_for_training = 0.0
                else:
                    # 如果当前玩家是获胜者，value=1；否则value=-1
                    value_for_training = 1.0 if winner == player_at_move else -1.0

                examples.append((state, policy, value_for_training))
                # 数据增强
                for sym_state, sym_policy in get_all_symmetries(
                    state, policy, self.game_size
                ):
                    examples.append((sym_state, sym_policy, value_for_training))

        return examples

    def self_play_multiprocess(self, num_workers=None, games_per_worker=1):
        """
        Multiprocessed self-play. Splits `self.num_self_play_games` into tasks and runs them
        in a multiprocessing Pool. Each worker initializes its own model copy in `worker_init`.

        Args:
            num_workers: number of worker processes to use (defaults to CPU count - 1)
            games_per_worker: how many games each worker runs per task (batching; default 1)

        Returns:
            List of (state, policy, value) tuples aggregated from all games.
        """
        # Ensure this function is safe to call in main process
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        # Prepare model state dict on CPU to send to workers
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Each worker uses the full simulation budget to ensure consistent search quality
        # regardless of parallelization
        per_worker_simulations = self.num_mcts_simulations

        # Build tasks: seeds and counts per task
        total_games = self.num_self_play_games
        tasks = []
        seed_base = int(time.time())
        for i in range(0, total_games, games_per_worker):
            count = min(games_per_worker, total_games - i)
            tasks.append((seed_base + i, count))

        # Use spawn to be safe across platforms
        ctx = mp.get_context("spawn")
        examples = []

        # Display progress info
        print(f"Starting {num_workers} worker processes for {total_games} games")

        # Track completed games and statistics
        completed_games = 0
        all_game_stats = []
        worker_progress = {}

        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(
                state_dict,
                self.game_size,
                per_worker_simulations,
                self.c_puct,
                self.temperature,
                self.dirichlet_alpha,
                self.dirichlet_weight,
                self.temp_threshold,
            ),
        ) as pool:
            # Create main progress bar
            with tqdm(total=total_games, desc="Self-play progress") as main_pbar:
                # Create progress dictionary for workers
                worker_pbars = {}

                for result_tuple in pool.imap_unordered(_worker_play, tasks):
                    # Unpack returned tuple: (examples, seed, count, game_stats, worker_id)
                    examples_batch, task_seed, task_count, game_stats, worker_id = (
                        result_tuple
                    )

                    # Update progress tracking
                    examples.extend(examples_batch)
                    main_pbar.update(task_count)
                    completed_games += task_count
                    all_game_stats.extend(game_stats)

                    # Track per-worker progress
                    if worker_id not in worker_progress:
                        worker_progress[worker_id] = 0
                    worker_progress[worker_id] += task_count

                    # Calculate statistics
                    avg_moves = (
                        sum(g["moves"] for g in game_stats) / len(game_stats)
                        if game_stats
                        else 0
                    )
                    avg_duration = (
                        sum(g["duration"] for g in game_stats) / len(game_stats)
                        if game_stats
                        else 0
                    )
                    outcomes = {"Win": 0, "Loss": 0, "Draw": 0}
                    for g in game_stats:
                        outcomes[g["outcome"]] += 1

                    # Update main progress bar postfix with statistics
                    main_pbar.set_postfix(
                        {
                            "games": f"{completed_games}/{total_games}",
                            "avg_moves": f"{avg_moves:.1f}",
                            "W/L/D": f"{outcomes['Win']}/{outcomes['Loss']}/{outcomes['Draw']}",
                        }
                    )

                    # Show detailed progress information every ~10% completion or for each worker update
                    if completed_games % max(1, total_games // 10) <= task_count:
                        # Calculate overall statistics
                        total_moves = sum(g["moves"] for g in all_game_stats)
                        overall_avg_moves = (
                            total_moves / len(all_game_stats) if all_game_stats else 0
                        )
                        total_outcomes = {"Win": 0, "Loss": 0, "Draw": 0}
                        for g in all_game_stats:
                            total_outcomes[g["outcome"]] += 1

                        print(
                            f"\nCompleted: {completed_games}/{total_games} games ({completed_games / total_games * 100:.1f}%)"
                        )
                        print(f"Average moves per game: {overall_avg_moves:.1f}")
                        print(
                            f"Outcomes - Win: {total_outcomes['Win']}, Loss: {total_outcomes['Loss']}, Draw: {total_outcomes['Draw']}"
                        )
                        print(
                            f"Worker progress: {', '.join([f'{w}: {c}' for w, c in worker_progress.items()])}"
                        )

        return examples

    def train_network(self, examples):
        """
        Train the neural network on the provided examples.

        Performs `num_epochs` full passes over the entire sliding-window data
        pool, with each epoch shuffling the data and training in mini-batches.
        This is the standard AlphaZero training loop and matches
        alpha-zero-general's behavior.

        Args:
            examples: List of (state, policy, value) tuples for the current iteration
        """
        # Sliding window: append this iteration's examples and drop oldest if needed
        self.train_examples_history.append(examples)
        if len(self.train_examples_history) > self.num_iters_for_history:
            dropped = self.train_examples_history.pop(0)
            print(
                f"Sliding window: dropped oldest iteration ({len(dropped)} samples), "
                f"keeping {len(self.train_examples_history)} iterations"
            )

        # Flatten all retained iterations into a single training pool
        all_examples = [e for iter_examples in self.train_examples_history for e in iter_examples]

        # If pool doesn't have enough samples, don't train
        if len(all_examples) < self.batch_size:
            print(
                f"Pool has only {len(all_examples)} samples, need at least {self.batch_size}. Skipping training."
            )
            return 0.0, 0.0, 0.0

        # Pre-convert to tensors once (outside the epoch loop) to avoid
        # repeated numpy→torch conversion overhead.
        states_arr, policies_arr, values_arr = zip(*all_examples)
        all_states = torch.FloatTensor(np.array(states_arr)).to(self.device)
        all_policies = torch.FloatTensor(np.array(policies_arr)).to(self.device)
        all_values = torch.FloatTensor(np.array(values_arr).reshape(-1, 1)).to(self.device)

        n = len(all_examples)
        batches_per_epoch = (n + self.batch_size - 1) // self.batch_size
        total_batches = batches_per_epoch * self.num_epochs

        print(
            f"Training pool: {n} samples from {len(self.train_examples_history)} "
            f"iterations | {self.num_epochs} epochs × {batches_per_epoch} batches = {total_batches} total"
        )

        # Compute current effective learning rate using per-iteration step decay.
        # We use a decay PER AlphaZero iteration (not per batch), so Adam's
        # momentum stays consistent with the LR within a single train_network
        # call. Per-batch schedules like CosineAnnealing conflict with Adam
        # because Adam's m/v state accumulates based on LR history.
        current_lr = self.lr * (self.lr_decay_factor ** self._iteration_count)
        current_lr = max(current_lr, self.lr_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        print(
            f"Learning rate this iteration: {current_lr:.6f} "
            f"(base={self.lr}, decay={self.lr_decay_factor}^{self._iteration_count})"
        )
        self._iteration_count += 1

        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_combined_loss = 0.0

        for epoch in range(self.num_epochs):
            # Shuffle indices for this epoch
            perm = torch.randperm(n, device=self.device)

            for batch_start in range(0, n, self.batch_size):
                batch_idx = perm[batch_start : batch_start + self.batch_size]
                batch_states = all_states[batch_idx]
                batch_policies = all_policies[batch_idx]
                batch_values = all_values[batch_idx]

                # Forward pass
                policy_logits, value_pred = self.model(batch_states)

                # Calculate loss (following AlphaZero paper exactly)
                policy_loss = -torch.sum(
                    batch_policies * torch.log_softmax(policy_logits, dim=1)
                ) / batch_states.size(0)
                value_loss = nn.MSELoss()(value_pred, batch_values)

                # L2 regularization is handled by Adam's weight_decay parameter
                combined_loss = value_loss + policy_loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_combined_loss += combined_loss.item()

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        avg_total_loss = total_combined_loss / total_batches

        # Record metrics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.total_losses.append(avg_total_loss)

        # Switch back to evaluation mode after training
        self.model.eval()

        return avg_policy_loss, avg_value_loss, avg_total_loss

    def evaluate_vs_random(self, num_games=30):
        """Evaluate self.model vs RandomPlayer.

        Plays num_games games with colors split evenly (half as Black, half as White)
        to eliminate first-mover bias. Uses temperature=0 (deterministic) and the
        arena MCTS simulation budget for speed.

        Returns:
            (win_rate, wins, losses, draws)
        """
        # Import locally to avoid circular imports at module load
        from play import RandomPlayer

        self.model.eval()
        mcts = MCTS(
            self.model,
            c_puct=self.c_puct,
            num_simulations=self.arena_mcts_simulations,
        )
        random_player = RandomPlayer()

        wins = 0
        losses = 0
        draws = 0
        half = num_games // 2

        for game_idx in range(num_games):
            env = OthelloEnv(size=self.game_size)
            env.reset()

            # First half: model plays Black (-1); second half: White (+1)
            model_player_id = -1 if game_idx < half else 1

            while not env.board.is_done():
                if env.board.current_player == model_player_id:
                    canonical_state = env.board.get_canonical_state()
                    state = mcts.canonical_to_observation(canonical_state, env)
                    action_probs = mcts.search(
                        state, env, temperature=0, add_noise=False
                    )
                    action = int(np.argmax(action_probs))
                    # Fallback: if the argmax action is somehow invalid, pick any valid
                    if env.get_valid_moves_mask()[action] == 0:
                        valid = np.where(env.get_valid_moves_mask() == 1)[0]
                        action = int(valid[0]) if len(valid) > 0 else env.board.get_action_space_size() - 1
                else:
                    action = random_player.get_action(env)

                env.step(action)

            winner = env.board.get_winner()
            if winner == 0:
                draws += 1
            elif winner == model_player_id:
                wins += 1
            else:
                losses += 1

        win_rate = wins / num_games
        return win_rate, wins, losses, draws

    def save_checkpoint(self, iteration, accepted_count=0, rejected_count=0):
        """Save model checkpoint for an accepted iteration."""
        checkpoint_file = f"{self.checkpoint_path}_{iteration}.pt"
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "policy_losses": self.policy_losses,
                "value_losses": self.value_losses,
                "total_losses": self.total_losses,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "global_best_win_rate": self.global_best_win_rate,
                "lr_iteration_count": self._iteration_count,
            },
            checkpoint_file,
        )
        print(f"Saved checkpoint: {checkpoint_file}")

        # Persist sliding-window training history alongside the checkpoint.
        # Stored separately because the data is large (~100 MB) and we don't
        # want to bloat the checkpoint pt file. Mirrors alpha-zero-general.
        examples_file = checkpoint_file + ".examples"
        try:
            with open(examples_file, "wb") as f:
                pickle.dump(self.train_examples_history, f)
            print(
                f"Saved training history: {examples_file} "
                f"({len(self.train_examples_history)} iterations)"
            )
        except Exception as e:
            print(f"Warning: failed to save training history: {e}")

    def load_checkpoint(self, path):
        """Load model checkpoint. Returns iteration number."""
        # map_location ensures checkpoints saved on GPU can be loaded on CPU
        # and vice versa.
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.value_losses = checkpoint.get("value_losses", [])
        self.total_losses = checkpoint.get("total_losses", [])
        self._resumed_accepted_count = checkpoint.get("accepted_count", 0)
        self._resumed_rejected_count = checkpoint.get("rejected_count", 0)
        self.global_best_win_rate = checkpoint.get("global_best_win_rate", 0.0)
        self._iteration_count = checkpoint.get("lr_iteration_count", 0)

        # Load sliding-window training history from sidecar .examples file
        examples_file = path + ".examples"
        if os.path.exists(examples_file):
            try:
                with open(examples_file, "rb") as f:
                    self.train_examples_history = pickle.load(f)
                print(
                    f"Loaded training history: {examples_file} "
                    f"({len(self.train_examples_history)} iterations)"
                )
            except Exception as e:
                print(f"Warning: failed to load training history: {e}")
                self.train_examples_history = []
        else:
            print(
                f"Warning: training history file not found ({examples_file}), "
                "starting with empty history"
            )
            self.train_examples_history = []

        # Load existing loss records into logger
        for i, (p_loss, v_loss, t_loss) in enumerate(
            zip(self.policy_losses, self.value_losses, self.total_losses)
        ):
            self.logger.log_iteration(
                iteration=i + 1,
                policy_loss=p_loss,
                value_loss=v_loss,
                total_loss=t_loss,
                examples_count=0,
                elapsed_time=0,
            )

        return checkpoint.get("iteration", 0)

    def plot_metrics(self):
        """Plot training metrics."""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.policy_losses)
        plt.title("Policy Loss")
        plt.xlabel("Training Iterations")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.value_losses)
        plt.title("Value Loss")
        plt.xlabel("Training Iterations")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.total_losses)
        plt.title("Total Loss")
        plt.xlabel("Training Iterations")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("./training_metrics.png")
        plt.close()

    def train(self, resume_from=None):
        """Main training loop for AlphaZero."""
        start_iteration = 0
        max_iterations = self.num_iterations

        # Load from checkpoint if specified
        if resume_from:
            start_iteration = self.load_checkpoint(resume_from)
            print(
                f"Resuming from iteration {start_iteration}, continuing for {max_iterations} more iterations"
            )
            print(
                f"Next iteration: {start_iteration + 1}/{start_iteration + max_iterations}"
            )

        # Adjust total iterations to ensure we train for the specified number regardless of resuming
        end_iteration = start_iteration + max_iterations

        # Load best model if it exists
        models_dir = os.path.dirname(self.checkpoint_path)
        best_path = os.path.join(models_dir, "best.pt")
        if os.path.exists(best_path):
            best_model_state = torch.load(best_path, map_location="cpu")
            print(f"Loaded best model from {best_path}")
        else:
            best_model_state = None
            print("No best model found, first iteration will auto-accept")

        # Arena for model evaluation
        arena = Arena(
            game_size=self.game_size,
            num_games=self.arena_games,
            mcts_simulations=self.arena_mcts_simulations,
            c_puct=self.c_puct,
        )

        # Restore accept/reject counts from checkpoint if resuming
        accepted_count = getattr(self, "_resumed_accepted_count", 0)
        rejected_count = getattr(self, "_resumed_rejected_count", 0)

        # Training loop
        for iteration in range(start_iteration, end_iteration):
            iteration_start_time = time.time()
            print(f"Starting iteration {iteration}/{end_iteration - 1}")

            # Self-play
            print("Self-play phase...")
            try:
                if self.use_multiprocessing:
                    examples = self.self_play_multiprocess(
                        num_workers=self.mp_num_workers,
                        games_per_worker=self.mp_games_per_worker,
                    )
                else:
                    examples = self.self_play()
                examples_count = len(examples)
                print(f"Generated {examples_count} examples from self-play")

                # Train neural network
                print("Training phase...")
                policy_loss, value_loss, total_loss = self.train_network(examples)
            except Exception as e:
                print(f"Error during iteration {iteration}: {e}")
                import traceback

                traceback.print_exc()
                print("Skipping to next iteration...")
                continue

            # Calculate elapsed time
            elapsed_time = time.time() - iteration_start_time
            print(
                f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Total Loss: {total_loss:.4f}"
            )
            print(f"Iteration completed in {elapsed_time:.2f} seconds")

            # Log metrics
            self.logger.log_iteration(
                iteration=iteration,
                policy_loss=policy_loss,
                value_loss=value_loss,
                total_loss=total_loss,
                examples_count=examples_count,
                elapsed_time=elapsed_time,
            )

            # Save candidate as temp.pt for crash recovery
            temp_path = os.path.join(models_dir, "temp.pt")
            candidate_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(candidate_state, temp_path)

            # Evaluate CANDIDATE vs Random BEFORE arena.
            # This measures absolute strength of the just-trained model,
            # independent of whether Arena accepts it. Arena only tells us
            # "candidate vs previous best", which doesn't guarantee absolute
            # improvement. vs-random is the only independent yardstick.
            if (iteration + 1) % self.eval_vs_random_interval == 0:
                print(f"Evaluating CANDIDATE vs random ({self.eval_vs_random_games} games)...")
                wr_vs_random, w, l, d = self.evaluate_vs_random(
                    num_games=self.eval_vs_random_games
                )
                print(
                    f"Candidate vs random: {wr_vs_random * 100:.1f}% ({w}W / {l}L / {d}D) | "
                    f"global best so far: {self.global_best_win_rate * 100:.1f}%"
                )

                if wr_vs_random > self.global_best_win_rate:
                    old_rate = self.global_best_win_rate
                    self.global_best_win_rate = wr_vs_random
                    # Saved file name reflects its meaning: "the model with
                    # the highest vs-random win rate ever observed". This is
                    # INDEPENDENT of Arena — Arena may have rejected this
                    # exact candidate. Use this file when you want "the
                    # absolute best model by objective metric".
                    vs_random_best_path = os.path.join(models_dir, "vs_random_best.pt")
                    torch.save(candidate_state, vs_random_best_path)
                    print(
                        f"⭐ NEW vs_random BEST: {old_rate * 100:.1f}% -> {wr_vs_random * 100:.1f}% "
                        f"(saved to {vs_random_best_path})"
                    )

            # First iteration auto-accept (no best model to compare against)
            if best_model_state is None:
                print("First iteration — auto-accepting model as best")
                torch.save(candidate_state, best_path)
                accepted_count += 1
                self.save_checkpoint(iteration, accepted_count, rejected_count)
                best_model_state = candidate_state
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(
                    f"Arena: ACCEPTED (first iteration) | Accepted: {accepted_count}, Rejected: {rejected_count}"
                )
            else:
                # Run arena evaluation
                print(f"Arena evaluation: {self.arena_games} games...")
                new_wins, old_wins, draws = arena.play_games(
                    candidate_state, best_model_state
                )
                denominator = new_wins + old_wins
                win_rate = new_wins / denominator if denominator > 0 else 0.0

                print(
                    f"Arena: New model wins: {new_wins}, Best model wins: {old_wins}, "
                    f"Draws: {draws}, Win rate: {win_rate * 100:.1f}%"
                )

                if win_rate >= self.arena_threshold:
                    print(
                        f"Arena: ACCEPTING new model (win rate {win_rate * 100:.1f}% >= {self.arena_threshold * 100:.1f}%)"
                    )
                    torch.save(candidate_state, best_path)
                    accepted_count += 1
                    self.save_checkpoint(iteration, accepted_count, rejected_count)
                    best_model_state = candidate_state
                else:
                    print(
                        f"Arena: REJECTING new model (win rate {win_rate * 100:.1f}% < {self.arena_threshold * 100:.1f}%)"
                    )
                    # Reload best model weights into the live trainer model
                    self.model.load_state_dict(best_model_state)
                    self.model.to(self.device)
                    rejected_count += 1

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                print(
                    f"Iteration {iteration}: Accepted: {accepted_count}, Rejected: {rejected_count}"
                )

            # Plot metrics
            if (iteration + 1) % 5 == 0 or iteration == end_iteration - 1:
                self.plot_metrics()
                curr_iter = start_iteration + iteration
                metrics_image = self.logger.plot_metrics(
                    save_path=os.path.join(
                        self.log_dir, f"metrics_iter_{curr_iter}.png"
                    )
                )
                print(f"Training metrics plotted to: {metrics_image}")

        print("Training completed!")
        self.plot_metrics()
        self.logger.plot_metrics(save_path="./training_metrics_final.png")
