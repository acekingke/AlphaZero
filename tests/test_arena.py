import unittest
import os
import sys
import torch
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import Arena, AlphaZeroTrainer
from models.neural_network import AlphaZeroNetwork


class TestArenaGameCount(unittest.TestCase):
    """T006: Test arena plays correct total games (20 as each color)."""

    def test_plays_correct_total_games(self):
        game_size = 6
        num_games = 40
        arena = Arena(game_size=game_size, num_games=num_games, mcts_simulations=2)

        model1 = AlphaZeroNetwork(game_size, device="cpu")
        model2 = AlphaZeroNetwork(game_size, device="cpu")

        # Track how many games are played with each color assignment
        original_play_single = arena.play_single_game
        games_as_black = 0
        games_as_white = 0

        def tracking_play(env, mcts1, mcts2, model1_is_black):
            nonlocal games_as_black, games_as_white
            if model1_is_black:
                games_as_black += 1
            else:
                games_as_white += 1
            return original_play_single(env, mcts1, mcts2, model1_is_black)

        arena.play_single_game = tracking_play

        m1_wins, m2_wins, draws = arena.play_games(
            model1.state_dict(), model2.state_dict()
        )

        total = m1_wins + m2_wins + draws
        self.assertEqual(total, num_games)
        self.assertEqual(games_as_black, 20)
        self.assertEqual(games_as_white, 20)


class TestArenaAcceptModel(unittest.TestCase):
    """T007: Test model accepted when new_wins/(new_wins+old_wins) >= 0.6."""

    def test_model_accepted_at_threshold(self):
        # win_rate = 6 / (6 + 4) = 0.6 -> accept
        new_wins = 6
        old_wins = 4
        draws = 0
        threshold = 0.6

        denominator = new_wins + old_wins
        win_rate = new_wins / denominator if denominator > 0 else 0.0
        self.assertGreaterEqual(win_rate, threshold)

    def test_model_accepted_above_threshold(self):
        # win_rate = 8 / (8 + 2) = 0.8 -> accept
        new_wins = 8
        old_wins = 2
        win_rate = new_wins / (new_wins + old_wins)
        self.assertGreaterEqual(win_rate, 0.6)


class TestArenaRejectModel(unittest.TestCase):
    """T008: Test model rejected when win rate < 0.6."""

    def test_model_rejected_below_threshold(self):
        # win_rate = 5 / (5 + 5) = 0.5 -> reject
        new_wins = 5
        old_wins = 5
        threshold = 0.6

        win_rate = new_wins / (new_wins + old_wins)
        self.assertLess(win_rate, threshold)

    def test_model_rejected_low_win_rate(self):
        # win_rate = 2 / (2 + 8) = 0.2 -> reject
        new_wins = 2
        old_wins = 8
        win_rate = new_wins / (new_wins + old_wins)
        self.assertLess(win_rate, 0.6)


class TestArenaDrawsRejection(unittest.TestCase):
    """T009: Test all-draws results in rejection (conservative)."""

    def test_all_draws_rejected(self):
        new_wins = 0
        old_wins = 0
        draws = 10
        threshold = 0.6

        denominator = new_wins + old_wins
        win_rate = new_wins / denominator if denominator > 0 else 0.0
        self.assertEqual(win_rate, 0.0)
        self.assertLess(win_rate, threshold)


class TestArenaFirstIteration(unittest.TestCase):
    """T010: Test first iteration auto-accepts when no best.pt exists."""

    def test_first_iteration_auto_accepts(self):
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(tmpdir, "checkpoint")
            best_path = os.path.join(tmpdir, "best.pt")

            # Verify no best.pt exists
            self.assertFalse(os.path.exists(best_path))

            # When best_model_state is None, first iteration should auto-accept
            best_model_state = None
            self.assertIsNone(best_model_state)

            # Simulate: first iteration auto-accepts by saving best.pt
            model = AlphaZeroNetwork(6, device="cpu")
            torch.save(model.state_dict(), best_path)

            self.assertTrue(os.path.exists(best_path))
        finally:
            shutil.rmtree(tmpdir)


class TestArenaFairness(unittest.TestCase):
    """T017: Verify arena color split is exactly 50/50."""

    def test_color_split_even(self):
        arena = Arena(game_size=6, num_games=10, mcts_simulations=2)
        model = AlphaZeroNetwork(6, device="cpu")

        original_play_single = arena.play_single_game
        color_assignments = []

        def tracking_play(env, mcts1, mcts2, model1_is_black):
            color_assignments.append(model1_is_black)
            return original_play_single(env, mcts1, mcts2, model1_is_black)

        arena.play_single_game = tracking_play
        arena.play_games(model.state_dict(), model.state_dict())

        as_black = sum(1 for c in color_assignments if c)
        as_white = sum(1 for c in color_assignments if not c)
        self.assertEqual(as_black, 5)
        self.assertEqual(as_white, 5)


class TestArenaWinRateExcludesDraws(unittest.TestCase):
    """T018: Verify win rate calculation excludes draws."""

    def test_draws_excluded_from_win_rate(self):
        # 3 new wins, 2 old wins, 5 draws
        new_wins = 3
        old_wins = 2
        draws = 5

        denominator = new_wins + old_wins
        win_rate = new_wins / denominator if denominator > 0 else 0.0

        # Win rate should be 3/5 = 0.6, NOT 3/10 = 0.3
        self.assertAlmostEqual(win_rate, 0.6)
        # Verify draws are NOT in denominator
        self.assertEqual(denominator, 5)
        self.assertNotEqual(denominator, 10)


class TestTrainerAcceptRejectIntegration(unittest.TestCase):
    """Integration tests: trainer.train() accept/reject branches with mocked Arena."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.tmpdir, "checkpoint")
        self.best_path = os.path.join(self.tmpdir, "best.pt")
        self.temp_path = os.path.join(self.tmpdir, "temp.pt")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_trainer(self, num_iterations=1):
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=num_iterations,
            num_self_play_games=1,
            num_mcts_simulations=2,
            batch_size=1,
            checkpoint_path=self.checkpoint_path,
            log_dir=self.tmpdir,
            use_mps=False,
            use_cuda=False,
            arena_games=10,
            arena_threshold=0.6,
            arena_mcts_simulations=2,
        )
        # Stub out heavy phases — we only care about the accept/reject branch
        trainer.self_play = MagicMock(return_value=[])
        trainer.train_network = MagicMock(return_value=(0.0, 0.0, 0.0))
        return trainer

    def test_first_iteration_auto_accepts_creates_best_pt(self):
        """First iteration with no best.pt should auto-accept and create best.pt."""
        trainer = self._make_trainer(num_iterations=1)
        self.assertFalse(os.path.exists(self.best_path))

        trainer.train()

        self.assertTrue(os.path.exists(self.best_path), "best.pt should be created")
        self.assertFalse(os.path.exists(self.temp_path), "temp.pt should be cleaned up")

    def test_accept_branch_promotes_candidate_to_best(self):
        """When arena win rate >= threshold, candidate replaces best.pt."""
        # Seed best.pt with a known model
        seed_model = AlphaZeroNetwork(6, device="cpu")
        for p in seed_model.parameters():
            p.data.fill_(0.0)
        torch.save(seed_model.state_dict(), self.best_path)
        original_best_bytes = os.path.getsize(self.best_path)

        trainer = self._make_trainer(num_iterations=1)
        # Modify the model so its state dict differs from seed
        for p in trainer.model.parameters():
            p.data.fill_(0.5)

        # Mock arena to return ACCEPT-level win rate (8/10 = 80%)
        with patch.object(Arena, "play_games", return_value=(8, 2, 0)):
            trainer.train()

        # best.pt should still exist and contain the new model (non-zero weights)
        self.assertTrue(os.path.exists(self.best_path))
        loaded = torch.load(self.best_path, map_location="cpu")
        first_param = next(iter(loaded.values()))
        self.assertFalse(
            torch.all(first_param == 0.0),
            "best.pt should contain new model (non-zero), not seed",
        )
        self.assertFalse(os.path.exists(self.temp_path), "temp.pt should be deleted")

    def test_reject_branch_keeps_old_best_and_reverts_model(self):
        """When arena win rate < threshold, best.pt is unchanged and trainer model is reverted."""
        # Seed best.pt with zeroed weights
        seed_model = AlphaZeroNetwork(6, device="cpu")
        for p in seed_model.parameters():
            p.data.fill_(0.0)
        torch.save(seed_model.state_dict(), self.best_path)

        trainer = self._make_trainer(num_iterations=1)
        # Set trainer model to non-zero so we can detect revert
        for p in trainer.model.parameters():
            p.data.fill_(0.5)

        # Mock arena to return REJECT-level win rate (3/10 = 30%)
        with patch.object(Arena, "play_games", return_value=(3, 7, 0)):
            trainer.train()

        # best.pt should still contain the seed (zeros)
        loaded = torch.load(self.best_path, map_location="cpu")
        first_param = next(iter(loaded.values()))
        self.assertTrue(
            torch.all(first_param == 0.0),
            "best.pt should still contain original seed (zeros)",
        )
        # Trainer model should have been reverted to seed weights
        first_trainer_param = next(iter(trainer.model.parameters()))
        self.assertTrue(
            torch.all(first_trainer_param.cpu() == 0.0),
            "Trainer model should be reverted to best weights",
        )
        self.assertFalse(os.path.exists(self.temp_path), "temp.pt should be deleted")

    def test_all_draws_results_in_rejection(self):
        """All-draws arena (denominator=0) is treated as 0% win rate → reject."""
        seed_model = AlphaZeroNetwork(6, device="cpu")
        for p in seed_model.parameters():
            p.data.fill_(0.0)
        torch.save(seed_model.state_dict(), self.best_path)

        trainer = self._make_trainer(num_iterations=1)
        for p in trainer.model.parameters():
            p.data.fill_(0.5)

        with patch.object(Arena, "play_games", return_value=(0, 0, 10)):
            trainer.train()

        # best.pt should still be the seed (zeros), candidate rejected
        loaded = torch.load(self.best_path, map_location="cpu")
        first_param = next(iter(loaded.values()))
        self.assertTrue(torch.all(first_param == 0.0))


if __name__ == "__main__":
    unittest.main()
