"""
状态一致性测试 - 测试状态转换和表示的一致性
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
        
        # 获取不同的状态表示
        board_state = self.env.board.get_state()
        canonical_state = self.env.board.get_canonical_state()
        observation = self.env.board.get_observation()
        
        # 检查维度
        assert board_state.shape == (self.game_size, self.game_size)
        assert canonical_state.shape == (self.game_size, self.game_size)
        assert observation.shape == (3, self.game_size, self.game_size)
        
        # 检查canonical state的正确性
        current_player = self.env.board.current_player
        expected_canonical = board_state * current_player
        np.testing.assert_array_equal(canonical_state, expected_canonical)
        
        # 检查observation的正确性
        expected_obs = np.zeros((3, self.game_size, self.game_size), dtype=np.float32)
        expected_obs[0] = (board_state == current_player).astype(np.float32)
        expected_obs[1] = (board_state == -current_player).astype(np.float32)
        expected_obs[2] = np.full((self.game_size, self.game_size), 
                                1.0 if current_player == -1 else 0.0, dtype=np.float32)
        
        np.testing.assert_array_equal(observation, expected_obs)
    
    def test_mcts_canonical_to_observation_consistency(self):
        """测试MCTS中canonical state到observation的转换一致性"""
        self.env.reset()
        canonical_state = self.env.board.get_canonical_state()
        
        # 使用MCTS的转换函数
        mcts_observation = self.mcts._canonical_to_observation(canonical_state, self.env)
        
        # 使用环境的观察函数
        env_observation = self.env.board.get_observation()
        
        # 由于canonical state视角，当前玩家总是+1
        expected_obs = np.zeros((3, self.game_size, self.game_size), dtype=np.float32)
        expected_obs[0] = (canonical_state == 1).astype(np.float32)
        expected_obs[1] = (canonical_state == -1).astype(np.float32)
        expected_obs[2] = np.ones((self.game_size, self.game_size), dtype=np.float32)
        
        # MCTS的转换应该产生canonical观察
        np.testing.assert_array_equal(mcts_observation, expected_obs)
    
    def test_training_vs_evaluation_consistency(self):
        """测试训练和评估中状态表示的一致性"""
        self.env.reset()
        
        # 模拟训练中的状态处理
        canonical_state = self.env.board.get_canonical_state()
        training_state = self.mcts._canonical_to_observation(canonical_state, self.env)
        
        # 模拟评估中的状态处理（应该与训练一致）
        player = AlphaZeroPlayer(self.model, num_simulations=10)
        
        # 获取AlphaZeroPlayer处理的状态
        # （这需要查看play.py中的实现）
        evaluation_canonical = self.env.board.get_canonical_state()
        evaluation_state = player.mcts._canonical_to_observation(evaluation_canonical, self.env)
        
        # 训练和评估的状态应该一致
        np.testing.assert_array_equal(training_state, evaluation_state)
    
    def test_action_space_consistency(self):
        """测试Bug #10: 动作空间大小不一致"""
        # 检查不同组件对动作空间大小的理解
        env_action_size = self.env.board.get_action_space_size()
        
        # 环境的动作空间
        assert env_action_size == self.game_size * self.game_size + 1
        
        # 神经网络的输出大小
        test_input = torch.randn(1, 3, self.game_size, self.game_size)
        with torch.no_grad():
            policy_logits, _ = self.model(test_input)
        network_action_size = policy_logits.shape[1]
        
        assert network_action_size == env_action_size
        
        # MCTS返回的策略大小
        state = self.env.reset()
        action_probs = self.mcts.search(state, self.env, temperature=1.0)
        mcts_action_size = len(action_probs)
        
        assert mcts_action_size == env_action_size
        
        # 有效动作掩码大小
        mask = self.env.get_valid_moves_mask()
        mask_size = len(mask)
        
        assert mask_size == env_action_size


class TestStateTransformations:
    """测试状态转换"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
    
    def test_player_perspective_consistency(self):
        """测试玩家视角的一致性"""
        self.env.reset()
        
        # 记录初始状态
        initial_player = self.env.board.current_player
        initial_canonical = self.env.board.get_canonical_state()
        initial_observation = self.env.board.get_observation()
        
        # 执行一个动作
        valid_moves = self.env.board.get_valid_moves()
        if valid_moves:
            action = self.env.get_action_from_coords(*valid_moves[0])
            self.env.step(action)
            
            # 检查玩家切换后的状态
            new_player = self.env.board.current_player
            new_canonical = self.env.board.get_canonical_state()
            new_observation = self.env.board.get_observation()
            
            # 玩家应该已经切换
            assert new_player == -initial_player
            
            # canonical state应该从新玩家的视角
            expected_new_canonical = self.env.board.get_state() * new_player
            np.testing.assert_array_equal(new_canonical, expected_new_canonical)
            
            # observation也应该从新玩家的视角
            assert new_observation.shape == (3, self.game_size, self.game_size)
    
    def test_action_coordinate_consistency(self):
        """测试动作坐标转换的一致性"""
        # 测试所有可能的坐标转换
        for row in range(self.game_size):
            for col in range(self.game_size):
                # 坐标到动作
                action = self.env.get_action_from_coords(row, col)
                
                # 动作到坐标
                converted_row, converted_col = self.env.get_coords_from_action(action)
                
                # 应该一致
                assert converted_row == row
                assert converted_col == col
        
        # 测试pass动作
        pass_action = self.game_size * self.game_size
        pass_coords = self.env.get_coords_from_action(pass_action)
        assert pass_coords is None
    
    def test_state_immutability_during_mcts(self):
        """测试MCTS搜索期间原始状态的不变性"""
        self.env.reset()
        
        # 记录初始状态
        initial_state = self.env.board.get_state().copy()
        initial_player = self.env.board.current_player
        
        # 执行MCTS搜索
        model = AlphaZeroNetwork(self.game_size, device='cpu')
        mcts = MCTS(model, num_simulations=10)
        
        state = self.env.board.get_observation()
        _ = mcts.search(state, self.env, temperature=1.0)
        
        # 原始环境状态应该保持不变
        final_state = self.env.board.get_state()
        final_player = self.env.board.current_player
        
        np.testing.assert_array_equal(initial_state, final_state)
        assert initial_player == final_player


class TestCrossComponentConsistency:
    """测试跨组件一致性"""
    
    def test_training_evaluation_pipeline_consistency(self):
        """测试训练-评估管道的一致性"""
        from train import AlphaZeroTrainer
        
        # 创建trainer
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=1,
            num_mcts_simulations=10,
            use_multiprocessing=False
        )
        
        # 生成一个自我对弈样本
        examples = trainer.self_play()
        assert len(examples) > 0
        
        # 检查样本格式
        state, policy, value = examples[0]
        assert state.shape == (3, 6, 6)
        assert len(policy) == 6 * 6 + 1
        assert -1 <= value <= 1
        
        # 检查状态格式与模型输入兼容
        model_input = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            model_output = trainer.model(model_input)
        
        assert model_output[0].shape == (1, 37)  # policy
        assert model_output[1].shape == (1, 1)   # value
    
    def test_model_mcts_compatibility(self):
        """测试模型与MCTS的兼容性"""
        model = AlphaZeroNetwork(6, device='cpu')
        mcts = MCTS(model, num_simulations=5)
        env = OthelloEnv(size=6)
        
        # 重复多次以确保稳定性
        for _ in range(5):
            state = env.reset()
            action_probs = mcts.search(state, env, temperature=1.0)
            
            # 检查输出格式
            assert len(action_probs) == 37
            assert abs(np.sum(action_probs) - 1.0) < 1e-6
            assert all(prob >= 0 for prob in action_probs)
    
    def test_environment_model_state_format(self):
        """测试环境和模型的状态格式兼容性"""
        env = OthelloEnv(size=6)
        model = AlphaZeroNetwork(6, device='cpu')
        
        # 环境状态
        observation = env.reset()
        
        # 模型应该能处理环境状态
        model_input = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = model(model_input)
        
        # 检查输出维度
        assert policy_logits.shape == (1, 37)
        assert value.shape == (1, 1)
        
        # 检查输出范围
        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))
        assert -1 <= value.item() <= 1


class TestDataAugmentationConsistency:
    """测试数据增强的一致性"""
    
    def test_symmetry_preservation(self):
        """测试对称性保持"""
        from utils.data_augmentation import get_all_symmetries
        
        # 创建一个简单的状态
        state = np.zeros((3, 6, 6), dtype=np.float32)
        state[0, 2, 2] = 1.0  # 当前玩家的一个棋子
        state[1, 3, 3] = 1.0  # 对手的一个棋子
        state[2, :, :] = 1.0  # 当前玩家指示器
        
        # 创建对应的策略
        policy = np.zeros(37, dtype=np.float32)
        policy[2 * 6 + 2] = 0.5  # (2,2)位置
        policy[3 * 6 + 3] = 0.5  # (3,3)位置
        
        # 获取所有对称性
        symmetries = get_all_symmetries(state, policy, 6)
        
        # 应该有8个对称性（原始 + 7个变换）
        assert len(symmetries) == 8
        
        # 每个对称状态应该保持相同的特性
        for sym_state, sym_policy in symmetries:
            # 形状保持
            assert sym_state.shape == state.shape
            assert len(sym_policy) == len(policy)
            
            # 策略和应该保持
            assert abs(np.sum(sym_policy) - np.sum(policy)) < 1e-6
            
            # 状态中的棋子数量应该保持
            assert np.sum(sym_state[0]) == np.sum(state[0])  # 当前玩家棋子
            assert np.sum(sym_state[1]) == np.sum(state[1])  # 对手棋子


if __name__ == "__main__":
    pytest.main([__file__, "-v"])