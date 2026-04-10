import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import deque, defaultdict
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
            # Use canonical state for consistency with MCTS
            canonical_state = env.board.get_canonical_state()
            # Convert to observation format for neural network (consistent with MCTS)
            state = mcts.canonical_to_observation(canonical_state, env)
            action_probs = mcts.search(state, env, WORKER_TEMPERATURE, add_noise=True)
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
            action_probs = mcts.search(state, env, temperature=0, add_noise=False)

            action = np.argmax(action_probs)
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
        num_epochs=100,
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

        # Get the best available device (CUDA, MPS, or CPU)
        self.device = get_device(use_mps=use_mps, use_cuda=use_cuda)
        print(f"Training will use: {self.device}")

        # Initialize neural network
        self.model = AlphaZeroNetwork(game_size, device=self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Experience buffer - stores about 50 iterations worth of data
        # Each iteration: ~1000 games * ~32 states/game * 6 augmentations = ~192000 examples
        # 50 iterations = ~9600000 examples
        self.buffer = deque(maxlen=10000000)  # 10M samples

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

            # Use fixed temperature throughout the game
            while not env.board.is_done():
                # Use canonical state for consistency with MCTS
                canonical_state = env.board.get_canonical_state()
                # Convert to observation format for neural network (consistent with MCTS)
                state = mcts.canonical_to_observation(canonical_state, env)
                action_probs = mcts.search(state, env, self.temperature, add_noise=True)

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

        Args:
            examples: List of (state, policy, value) tuples
        """
        # Add examples to buffer
        self.buffer.extend(examples)

        # If buffer doesn't have enough samples, don't train
        if len(self.buffer) < self.batch_size:
            print(
                f"Buffer has only {len(self.buffer)} samples, need at least {self.batch_size}. Skipping training."
            )
            return 0.0, 0.0, 0.0  # Return zero losses when skipping training

        # Extract data
        mini_batch = random.sample(
            self.buffer, min(len(self.buffer), self.batch_size * self.num_epochs)
        )

        # Split data
        states, policies, values = zip(*mini_batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        policies = torch.FloatTensor(np.array(policies)).to(self.device)
        values = torch.FloatTensor(np.array(values).reshape(-1, 1)).to(self.device)

        # Training loop
        self.model.train()

        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0

        # Split into batches
        for i in range(0, len(mini_batch), self.batch_size):
            batch_states = states[i : i + self.batch_size]
            batch_policies = policies[i : i + self.batch_size]
            batch_values = values[i : i + self.batch_size]

            # Forward pass
            policy_logits, value_pred = self.model(batch_states)

            # Calculate loss (following AlphaZero paper exactly)
            policy_loss = -torch.sum(
                batch_policies * torch.log_softmax(policy_logits, dim=1)
            ) / batch_states.size(0)
            value_loss = nn.MSELoss()(value_pred, batch_values)

            # L2 regularization is handled by Adam's weight_decay parameter
            total_loss = value_loss + policy_loss

            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()

        # Calculate average loss
        num_batches = (len(mini_batch) + self.batch_size - 1) // self.batch_size
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        # Record metrics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.total_losses.append(avg_total_loss)

        # Switch back to evaluation mode after training
        self.model.eval()

        return avg_policy_loss, avg_value_loss, avg_total_loss

    def save_checkpoint(self, iteration, accepted_count=0, rejected_count=0):
        """Save model checkpoint for an accepted iteration."""
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
            },
            f"{self.checkpoint_path}_{iteration}.pt",
        )

        print(f"Saved checkpoint: {self.checkpoint_path}_{iteration}.pt")

    def load_checkpoint(self, path):
        """Load model checkpoint. Returns (iteration, accepted_count, rejected_count)."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.value_losses = checkpoint.get("value_losses", [])
        self.total_losses = checkpoint.get("total_losses", [])
        self._resumed_accepted_count = checkpoint.get("accepted_count", 0)
        self._resumed_rejected_count = checkpoint.get("rejected_count", 0)

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
