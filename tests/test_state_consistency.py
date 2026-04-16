"""
状态一致性测试 - 测试状态转换和表示的一致性
Updated for 1-channel input (matching alpha-zero-general).
"""
import pytest
import numpy as np
import torch
import sys
import os
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.mcts import MCTS
from env.othello import OthelloEnv, OthelloBoard
from models.neural_network import AlphaZeroNetwork
from play import AlphaZeroPlayer


class TestStateConsistency:
    """测试状态一致性"""

    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.model = AlphaZeroNetwork(self.game_size, device='cpu')
        self.mcts = MCTS(self.model, num_simulations=10)

    def test_state_representation_consistency(self):
        """测试Bug #9: 状态转换函数一致性"""
        self.env.reset()

        board_state = self.env.board.get_state()
        canonical_state = self.env.board.get_canonical_state()

        assert board_state.shape == (self.game_size, self.game_size)
        assert canonical_state.shape == (self.game_size, self.game_size)

        current_player = self.env.board.current_player
        expected_canonical = board_state * current_player
        np.testing.assert_array_equal(canonical_state, expected_canonical)

    def test_mctscanonical_to_observation_consistency(self):
        """测试MCTS canonical_to_observation returns canonical board as float32."""
        self.env.reset()
        canonical_state = self.env.board.get_canonical_state()

        mcts_observation = self.mcts.canonical_to_observation(canonical_state, self.env)

        # With 1-channel input, canonical_to_observation just returns canonical_state as float32
        assert mcts_observation.shape == (self.game_size, self.game_size)
        assert mcts_observation.dtype == np.float32
        np.testing.assert_array_equal(mcts_observation, canonical_state.astype(np.float32))

    def test_training_vs_evaluation_consistency(self):
        """测试训练和评估中状态表示的一致性"""
        self.env.reset()

        canonical_state = self.env.board.get_canonical_state()
        training_state = self.mcts.canonical_to_observation(canonical_state, self.env)

        player = AlphaZeroPlayer(self.model, num_simulations=10)
        evaluation_canonical = self.env.board.get_canonical_state()
        evaluation_state = player.mcts.canonical_to_observation(evaluation_canonical, self.env)

        np.testing.assert_array_equal(training_state, evaluation_state)

    def test_action_space_consistency(self):
        """测试Bug #10: 动作空间大小不一致"""
        env_action_size = self.env.board.get_action_space_size()
        assert env_action_size == self.game_size * self.game_size + 1

        # 1-channel input: (B, board_x, board_y)
        test_input = torch.randn(1, self.game_size, self.game_size)
        self.model.eval()
        with torch.no_grad():
            log_pi, _ = self.model(test_input)
        network_action_size = log_pi.shape[1]
        assert network_action_size == env_action_size

        self.env.reset()
        canonical = self.env.board.get_canonical_state()
        action_probs = self.mcts.search(canonical, self.env, temperature=1.0)
        mcts_action_size = len(action_probs)
        assert mcts_action_size == env_action_size

        mask = self.env.get_valid_moves_mask()
        mask_size = len(mask)
        assert mask_size == env_action_size


class TestStateTransformations:
    """测试状态转换"""

    def setup_method(self):
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)

    def test_player_perspective_consistency(self):
        """测试玩家视角的一致性"""
        self.env.reset()

        initial_player = self.env.board.current_player
        initial_canonical = self.env.board.get_canonical_state()

        valid_moves = self.env.board.get_valid_moves()
        if valid_moves:
            action = self.env.get_action_from_coords(*valid_moves[0])
            self.env.step(action)

            new_player = self.env.board.current_player
            new_canonical = self.env.board.get_canonical_state()

            assert new_player == -initial_player
            expected_new_canonical = self.env.board.get_state() * new_player
            np.testing.assert_array_equal(new_canonical, expected_new_canonical)

    def test_action_coordinate_consistency(self):
        """测试动作坐标转换的一致性"""
        for row in range(self.game_size):
            for col in range(self.game_size):
                action = self.env.get_action_from_coords(row, col)
                converted_row, converted_col = self.env.get_coords_from_action(action)
                assert converted_row == row
                assert converted_col == col

        pass_action = self.game_size * self.game_size
        pass_coords = self.env.get_coords_from_action(pass_action)
        assert pass_coords is None

    def test_state_immutability_during_mcts(self):
        """测试MCTS搜索期间原始状态的不变性"""
        self.env.reset()

        initial_state = self.env.board.get_state().copy()
        initial_player = self.env.board.current_player

        model = AlphaZeroNetwork(self.game_size, device='cpu')
        mcts = MCTS(model, num_simulations=10)

        canonical = self.env.board.get_canonical_state()
        _ = mcts.search(canonical, self.env, temperature=1.0)

        final_state = self.env.board.get_state()
        final_player = self.env.board.current_player

        np.testing.assert_array_equal(initial_state, final_state)
        assert initial_player == final_player


class TestCrossComponentConsistency:
    """测试跨组件一致性"""

    def test_training_evaluation_pipeline_consistency(self):
        """测试训练-评估管道的一致性"""
        from train import AlphaZeroTrainer

        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=1,
            num_mcts_simulations=10,
            use_multiprocessing=False
        )

        examples = trainer.self_play()
        assert len(examples) > 0

        state, policy, value = examples[0]
        # 1-channel: state is (6, 6) canonical board
        assert state.shape == (6, 6)
        assert len(policy) == 6 * 6 + 1
        assert -1 <= value <= 1

        # Model accepts (B, board_x, board_y) input
        model_input = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            model_output = trainer.model(model_input)

        assert model_output[0].shape == (1, 37)  # log_softmax policy
        assert model_output[1].shape == (1, 1)   # tanh value

    def test_model_mcts_compatibility(self):
        """测试模型与MCTS的兼容性"""
        model = AlphaZeroNetwork(6, device='cpu')
        mcts = MCTS(model, num_simulations=5)
        env = OthelloEnv(size=6)

        for _ in range(5):
            env.reset()
            canonical = env.board.get_canonical_state()
            action_probs = mcts.search(canonical, env, temperature=1.0)

            assert len(action_probs) == 37
            assert abs(np.sum(action_probs) - 1.0) < 1e-6
            assert all(prob >= 0 for prob in action_probs)

    def test_environment_model_state_format(self):
        """测试环境和模型的状态格式兼容性"""
        env = OthelloEnv(size=6)
        model = AlphaZeroNetwork(6, device='cpu')

        env.reset()
        canonical = env.board.get_canonical_state()

        # 1-channel: feed canonical board directly (B, 6, 6)
        model_input = torch.FloatTensor(canonical.astype(np.float64)).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            log_pi, value = model(model_input)

        assert log_pi.shape == (1, 37)
        assert value.shape == (1, 1)
        assert torch.all(torch.isfinite(log_pi))
        assert torch.all(torch.isfinite(value))
        assert -1 <= value.item() <= 1


class TestDataAugmentationConsistency:
    """测试数据增强的一致性"""

    def test_symmetry_preservation(self):
        """测试对称性保持"""
        from utils.data_augmentation import get_all_symmetries

        # 1-channel canonical board
        state = np.zeros((6, 6), dtype=np.float32)
        state[2, 2] = 1.0   # current player piece
        state[3, 3] = -1.0  # opponent piece

        policy = np.zeros(37, dtype=np.float32)
        policy[2 * 6 + 2] = 0.5
        policy[3 * 6 + 3] = 0.5

        symmetries = get_all_symmetries(state, policy, 6)

        # 7 non-identity symmetries (caller provides identity)
        assert len(symmetries) == 7

        for sym_state, sym_policy in symmetries:
            assert sym_state.shape == state.shape
            assert len(sym_policy) == len(policy)
            assert abs(np.sum(sym_policy) - np.sum(policy)) < 1e-6
            # Same number of +1 and -1 pieces
            assert np.sum(sym_state == 1.0) == np.sum(state == 1.0)
            assert np.sum(sym_state == -1.0) == np.sum(state == -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
