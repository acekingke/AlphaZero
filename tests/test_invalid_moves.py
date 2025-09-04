#!/usr/bin/env python3
"""
测试自我对弈过程中的无效动作处理
验证MCTS和环境是否正确处理无效动作
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
from train import _worker_play


class TestInvalidMoveHandling:
    """测试无效动作处理"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.model = AlphaZeroNetwork(self.game_size, device='cpu')
        self.mcts = MCTS(self.model, num_simulations=10, c_puct=1.0)
    
    def test_mcts_policy_masking(self):
        """测试MCTS是否正确应用了有效动作掩码"""
        self.env.reset()
        
        # 获取有效动作掩码
        valid_mask = self.env.get_valid_moves_mask()
        print(f"Valid moves mask: {valid_mask}")
        print(f"Valid move indices: {np.where(valid_mask == 1)[0]}")
        
        # 执行MCTS搜索
        canonical_state = self.env.board.get_canonical_state()
        state = self.mcts._canonical_to_observation(canonical_state, self.env)
        action_probs = self.mcts.search(state, self.env, temperature=1.0)
        
        print(f"Action probabilities: {action_probs}")
        print(f"Non-zero probabilities at: {np.where(action_probs > 0)[0]}")
        
        # 验证：只有有效动作才有非零概率
        for i, prob in enumerate(action_probs):
            if prob > 0:
                assert valid_mask[i] == 1, f"动作 {i} 有概率 {prob} 但不是有效动作"
        
        # 验证：所有有效动作都有一定概率（除非是确定性选择）
        for i, mask_val in enumerate(valid_mask):
            if mask_val == 1:
                assert action_probs[i] >= 0, f"有效动作 {i} 的概率不应该为负"
    
    def test_env_invalid_move_punishment(self):
        """测试环境对无效动作的惩罚"""
        self.env.reset()
        
        # 尝试一个明显的无效动作（已占用的位置）
        invalid_action = self.env.get_action_from_coords(2, 2)  # 初始状态下已有白棋
        
        obs, reward, done, info = self.env.step(invalid_action)
        
        # 验证无效动作的处理
        assert reward == -10, f"无效动作应该给予-10奖励，实际得到{reward}"
        assert done == True, "无效动作应该结束游戏"
        assert "valid_moves" in info, "info应该包含valid_moves信息"
    
    def test_mcts_never_selects_invalid_actions(self):
        """测试MCTS永远不会选择无效动作"""
        # 测试多个游戏状态
        for _ in range(10):
            self.env.reset()
            
            # 随机进行几步游戏
            for step in range(np.random.randint(1, 10)):
                if self.env.board.is_done():
                    break
                
                # 获取有效动作掩码
                valid_mask = self.env.get_valid_moves_mask()
                
                if np.sum(valid_mask) == 0:
                    break  # 没有有效动作
                
                # 使用MCTS选择动作
                canonical_state = self.env.board.get_canonical_state()
                state = self.mcts._canonical_to_observation(canonical_state, self.env)
                action_probs = self.mcts.search(state, self.env, temperature=1.0)
                
                # 确保只有有效动作有概率
                for i, prob in enumerate(action_probs):
                    if prob > 0:
                        assert valid_mask[i] == 1, f"步骤{step}: 动作{i}有概率但无效"
                
                # 选择动作
                action = np.random.choice(len(action_probs), p=action_probs)
                
                # 验证选择的动作是有效的
                assert valid_mask[action] == 1, f"步骤{step}: 选择的动作{action}无效"
                
                # 执行动作
                obs, reward, done, info = self.env.step(action)
                
                # 有效动作不应该导致-10惩罚
                assert reward != -10, f"步骤{step}: 有效动作{action}却得到了惩罚奖励{reward}"
                
                if done:
                    break
    
    def test_corner_case_no_valid_moves(self):
        """测试没有有效动作时的情况"""
        # 创建一个没有有效动作的特殊棋盘状态
        self.env.reset()
        board = self.env.board
        
        # 设置一个玩家没有有效动作的状态
        board.board = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ], dtype=np.int8)
        board.current_player = -1  # 黑方，但没有有效动作
        
        # 获取有效动作掩码
        valid_mask = self.env.get_valid_moves_mask()
        print(f"No valid moves scenario - mask: {valid_mask}")
        print(f"Sum of valid moves: {np.sum(valid_mask)}")
        
        # 如果没有有效的放置动作，应该只能pass
        if np.sum(valid_mask[:-1]) == 0:  # 除了pass动作
            # 应该允许pass动作
            pass_action = self.game_size * self.game_size
            if valid_mask[pass_action] == 1:
                obs, reward, done, info = self.env.step(pass_action)
                assert reward != -10, "正确的pass不应该被惩罚"
    
    def test_policy_normalization_edge_cases(self):
        """测试策略归一化的边缘情况"""
        self.env.reset()
        
        # 获取有效动作掩码
        valid_mask = self.env.get_valid_moves_mask()
        print(f"Valid mask in edge case test: {valid_mask}")
        
        # Mock一个神经网络输出极端概率的情况
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        # 创建MCTS实例
        mcts = MCTS(mock_model, num_simulations=1, c_puct=1.0)
        
        # Mock评估函数返回极端值，但只对有效动作
        def mock_evaluate_extreme(canonical_state, env):
            action_space_size = env.board.get_action_space_size()
            valid_mask = env.get_valid_moves_mask()
            
            # 返回极端不均匀的策略，但只在有效动作上
            policy = np.zeros(action_space_size)
            valid_indices = np.where(valid_mask == 1)[0]
            if len(valid_indices) > 0:
                policy[valid_indices[0]] = 1000.0  # 给第一个有效动作极大值
            
            return policy, 0.5
        
        mcts._evaluate_state = mock_evaluate_extreme
        
        # 执行搜索
        canonical_state = self.env.board.get_canonical_state()
        state = mcts._canonical_to_observation(canonical_state, self.env)
        
        try:
            action_probs = mcts.search(state, self.env, temperature=1.0)
            
            # 验证概率和为1
            prob_sum = np.sum(action_probs)
            assert abs(prob_sum - 1.0) < 1e-6, f"概率和应该为1，实际为{prob_sum}"
            
            # 验证所有概率非负
            assert np.all(action_probs >= 0), "所有概率都应该非负"
            
            # 验证只有有效动作有概率
            for i, prob in enumerate(action_probs):
                if prob > 0:
                    assert valid_mask[i] == 1, f"无效动作{i}不应该有概率{prob}"
                    
        except Exception as e:
            pytest.fail(f"极端策略归一化失败: {e}")


class TestSelfPlayInvalidMoveProtection:
    """测试自我对弈中的无效动作保护"""
    
    def test_worker_play_no_invalid_moves(self):
        """测试worker自我对弈不会产生无效动作"""
        
        # 由于multiprocessing的复杂性，我们模拟worker_play的核心逻辑
        game_size = 6
        env = OthelloEnv(size=game_size)
        model = AlphaZeroNetwork(game_size, device='cpu')
        mcts = MCTS(model, num_simulations=5, c_puct=1.0)
        
        invalid_move_count = 0
        total_moves = 0
        
        # 模拟多个游戏
        for game in range(5):
            env.reset()
            step = 0
            
            while not env.board.is_done() and step < 100:  # 防止无限循环
                # 使用canonical state保持一致性
                canonical_state = env.board.get_canonical_state()
                state = mcts._canonical_to_observation(canonical_state, env)
                
                try:
                    action_probs = mcts.search(state, env, temperature=1.0, add_noise=True)
                    action = np.random.choice(len(action_probs), p=action_probs)
                    
                    total_moves += 1
                    
                    # 执行动作
                    obs, reward, done, info = env.step(action)
                    
                    # 检查是否是无效动作
                    if reward == -10:
                        invalid_move_count += 1
                        print(f"游戏{game}步骤{step}: 无效动作{action}，奖励{reward}")
                        print(f"有效动作掩码: {env.get_valid_moves_mask()}")
                        break  # 游戏因无效动作结束
                        
                    step += 1
                    if done:
                        break
                        
                except Exception as e:
                    print(f"游戏{game}步骤{step}出错: {e}")
                    break
        
        print(f"总动作数: {total_moves}, 无效动作数: {invalid_move_count}")
        print(f"无效动作率: {invalid_move_count/total_moves*100:.2f}%" if total_moves > 0 else "N/A")
        
        # 理想情况下应该没有无效动作
        assert invalid_move_count == 0, f"自我对弈中不应该有无效动作，但发现了{invalid_move_count}个"


if __name__ == "__main__":
    print("测试自我对弈中的无效动作处理...")
    print("=" * 60)
    
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])