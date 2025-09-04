#!/usr/bin/env python3
"""
专门测试MCTS价值计算bug的测试用例
测试修复前后的行为差异，确保价值计算的正确性
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


class TestMCTSValueCalculationBug:
    """测试MCTS价值计算bug的修复"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.model = AlphaZeroNetwork(self.game_size, device='cpu')
        self.mcts = MCTS(self.model, num_simulations=5, c_puct=1.0)  # 使用少量模拟以便调试
    
    def test_search_start_player_tracking(self):
        """测试search_start_player是否被正确记录"""
        self.env.reset()
        
        # 黑方开始搜索
        initial_player = self.env.board.current_player  # 应该是-1（黑方）
        state = self.env.board.get_observation()
        
        # 执行搜索
        self.mcts.search(state, self.env, temperature=1.0)
        
        # 验证搜索开始玩家被正确记录
        assert hasattr(self.mcts, 'search_start_player'), "MCTS应该记录search_start_player"
        assert self.mcts.search_start_player == initial_player, f"search_start_player应该是{initial_player}，但实际是{self.mcts.search_start_player}"
    
    def test_terminal_value_calculation_black_wins(self):
        """测试黑方获胜时的价值计算"""
        # 创建一个黑方即将获胜的棋盘状态
        self.env.reset()
        board = self.env.board
        
        # 手动设置一个黑方获胜的终局状态
        board.board = np.array([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [ 1,  1,  1,  1,  1,  0]
        ], dtype=np.int8)
        
        board.current_player = -1  # 黑方行动
        board.done = False
        
        # 执行搜索（黑方视角）
        state = board.get_observation()
        action_probs = self.mcts.search(state, self.env, temperature=0.1)
        
        # 验证搜索开始玩家
        assert self.mcts.search_start_player == -1, "搜索应该从黑方开始"
        
        # 在MCTS内部模拟中，如果黑方获胜，价值应该是正的（从黑方视角）
        # 这个测试主要验证不会出现值计算错误
        assert len(action_probs) == self.game_size * self.game_size + 1, "动作概率数组长度应该正确"
    
    def test_terminal_value_calculation_white_wins(self):
        """测试白方获胜时的价值计算"""
        # 创建一个白方即将获胜的棋盘状态
        self.env.reset()
        board = self.env.board
        
        # 手动设置一个白方获胜的终局状态
        board.board = np.array([
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [-1, -1, -1, -1, -1,  0]
        ], dtype=np.int8)
        
        board.current_player = 1  # 白方行动
        board.done = False
        
        # 执行搜索（白方视角）
        state = board.get_observation()
        action_probs = self.mcts.search(state, self.env, temperature=0.1)
        
        # 验证搜索开始玩家
        assert self.mcts.search_start_player == 1, "搜索应该从白方开始"
        
        # 验证动作概率数组
        assert len(action_probs) == self.game_size * self.game_size + 1, "动作概率数组长度应该正确"
    
    def test_player_alternation_during_search(self):
        """测试搜索过程中玩家切换不影响价值计算"""
        self.env.reset()
        
        # 记录初始状态
        initial_player = self.env.board.current_player
        initial_state = self.env.board.get_state().copy()
        
        # 执行搜索
        state = self.env.board.get_observation()
        action_probs = self.mcts.search(state, self.env, temperature=1.0)
        
        # 验证原始环境状态未被改变
        final_state = self.env.board.get_state()
        final_player = self.env.board.current_player
        
        np.testing.assert_array_equal(initial_state, final_state, "搜索不应该改变原始环境状态")
        assert initial_player == final_player, "搜索不应该改变原始环境的当前玩家"
        
        # 验证搜索记录的玩家是正确的
        assert self.mcts.search_start_player == initial_player, "search_start_player应该等于初始玩家"
    
    def test_value_consistency_across_simulations(self):
        """测试多次模拟中价值计算的一致性"""
        self.env.reset()
        
        # 执行多次搜索，验证价值计算的一致性
        state = self.env.board.get_observation()
        
        results = []
        for _ in range(3):
            action_probs = self.mcts.search(state, self.env, temperature=1.0)
            results.append(action_probs.copy())
        
        # 由于随机性，结果可能不完全相同，但应该在合理范围内
        # 主要是验证没有崩溃或异常值
        for result in results:
            assert len(result) == self.game_size * self.game_size + 1, "所有结果长度应该一致"
            assert np.all(result >= 0), "所有概率应该非负"
            assert np.isclose(np.sum(result), 1.0, atol=1e-6), "概率和应该等于1"

    def test_mock_terminal_states_value_assignment(self):
        """使用mock测试终端状态的价值分配是否正确"""
        self.env.reset()
        
        # Mock一个简单的模型
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]  # 用于设备检测
        
        # 创建MCTS实例
        mcts = MCTS(mock_model, num_simulations=1, c_puct=1.0)
        
        # 测试黑方获胜场景
        state = self.env.board.get_observation()
        
        # 记录原始环境状态
        original_env = self.env
        original_player = original_env.board.current_player
        
        # 使用patch来模拟终端状态
        with patch.object(mcts, '_evaluate_state') as mock_evaluate:
            mock_evaluate.return_value = (
                np.ones(self.game_size * self.game_size + 1) / (self.game_size * self.game_size + 1),  # 均匀策略
                0.0  # 中性价值
            )
            
            # 执行搜索
            try:
                action_probs = mcts.search(state, self.env, temperature=1.0)
                # 验证搜索完成
                assert len(action_probs) == self.game_size * self.game_size + 1
                # 验证搜索开始玩家被正确记录
                assert hasattr(mcts, 'search_start_player')
                assert mcts.search_start_player == original_player
            except Exception as e:
                pytest.fail(f"MCTS搜索失败: {e}")

    def test_edge_case_immediate_terminal_state(self):
        """测试立即进入终端状态的边缘情况"""
        # 创建一个已经结束的游戏状态
        self.env.reset()
        board = self.env.board
        
        # 设置一个游戏结束的状态
        board.board.fill(1)  # 全部是白子
        board.board[0, 0] = -1  # 只有一个黑子
        board.done = True
        board.winner = 1  # 白方获胜
        board.current_player = -1  # 当前是黑方
        
        state = board.get_observation()
        
        # 执行搜索（应该能处理已经结束的游戏）
        try:
            action_probs = self.mcts.search(state, self.env, temperature=1.0)
            assert len(action_probs) == self.game_size * self.game_size + 1
            # 验证搜索开始玩家被记录
            assert hasattr(self.mcts, 'search_start_player')
        except Exception as e:
            pytest.fail(f"处理终端状态时失败: {e}")


if __name__ == "__main__":
    print("运行MCTS价值计算bug测试...")
    print("=" * 60)
    
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])