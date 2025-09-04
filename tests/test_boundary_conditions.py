"""
边界条件测试 - 测试MCTS和Othello的边界条件处理
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.mcts import MCTS, Node
from env.othello import OthelloEnv, OthelloBoard
from models.neural_network import AlphaZeroNetwork


class TestMCTSBoundaryConditions:
    """测试MCTS的边界条件"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.model = AlphaZeroNetwork(self.game_size, device='cpu')
        self.mcts = MCTS(self.model, num_simulations=10)
    
    def test_action_probability_array_bounds(self):
        """测试Bug #1: MCTS动作概率数组越界风险"""
        # 创建一个根节点，故意添加无效的action
        root = Node(0)
        
        # 模拟一个有无效action的children字典
        action_space_size = self.game_size * self.game_size + 1
        root.children = {
            0: Node(0.1),
            1: Node(0.2),
            action_space_size: Node(0.3),  # 等于边界值
            action_space_size + 1: Node(0.4),  # 超出边界
            -1: Node(0.5),  # 负数
            999: Node(0.6),  # 巨大值
        }
        
        # 给所有children设置visit_count
        for child in root.children.values():
            child.visit_count = 1
        
        # 调用_get_action_probabilities应该能安全处理无效action
        action_probs = self.mcts._get_action_probabilities(root, 1.0, self.env)
        
        # 检查结果
        assert len(action_probs) == action_space_size
        assert all(prob >= 0 for prob in action_probs)
        assert not np.any(np.isnan(action_probs))
        assert not np.any(np.isinf(action_probs))
    
    def test_action_probability_empty_children(self):
        """测试空children的情况"""
        root = Node(0)
        action_probs = self.mcts._get_action_probabilities(root, 1.0, self.env)
        
        # 应该返回全零数组
        assert len(action_probs) == self.game_size * self.game_size + 1
        assert np.sum(action_probs) == 0
    
    def test_action_probability_negative_actions(self):
        """测试负数action的处理"""
        root = Node(0)
        root.children = {
            -1: Node(0.5),
            -10: Node(0.3),
            0: Node(0.2),
        }
        
        for child in root.children.values():
            child.visit_count = 1
        
        action_probs = self.mcts._get_action_probabilities(root, 1.0, self.env)
        
        # 只有有效的action应该有概率
        assert action_probs[0] > 0  # action 0 应该有概率
        assert np.sum(action_probs) > 0


class TestOthelloBoundaryConditions:
    """测试Othello游戏的边界条件"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.board = OthelloBoard(size=self.game_size)
    
    def test_board_boundary_access(self):
        """测试棋盘边界访问"""
        # 测试越界访问不会崩溃
        assert not self.board._is_valid_move(-1, 0)
        assert not self.board._is_valid_move(0, -1)
        assert not self.board._is_valid_move(self.game_size, 0)
        assert not self.board._is_valid_move(0, self.game_size)
        assert not self.board._is_valid_move(self.game_size, self.game_size)
    
    def test_action_conversion_boundary(self):
        """测试动作转换的边界条件"""
        # 测试action转换
        max_action = self.game_size * self.game_size
        
        # 有效边界
        row, col = self.env.get_coords_from_action(0)
        assert row == 0 and col == 0
        
        row, col = self.env.get_coords_from_action(max_action - 1)
        assert row == self.game_size - 1 and col == self.game_size - 1
        
        # pass action
        coords = self.env.get_coords_from_action(max_action)
        assert coords is None
        
    def test_valid_moves_mask_boundary(self):
        """测试有效移动掩码的边界条件"""
        mask = self.env.get_valid_moves_mask()
        
        # 检查掩码长度正确
        expected_length = self.game_size * self.game_size + 1
        assert len(mask) == expected_length
        
        # 检查数据类型
        assert mask.dtype == np.int8
        
        # 检查值范围
        assert all(val in [0, 1] for val in mask)


class TestEdgeCaseGameStates:
    """测试游戏状态的边缘情况"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
    
    def test_full_board_state(self):
        """测试棋盘满的情况"""
        # 手动创建一个接近满的棋盘
        self.env.board.board.fill(1)  # 填满白子
        self.env.board.board[0, 0] = -1  # 一个黑子
        self.env.board.board[0, 1] = 0   # 一个空位
        
        # 检查游戏状态
        valid_moves = self.env.board.get_valid_moves()
        mask = self.env.get_valid_moves_mask()
        
        # 应该能正常处理
        assert isinstance(valid_moves, list)
        assert len(mask) == self.game_size * self.game_size + 1
    
    def test_no_valid_moves_state(self):
        """测试没有有效移动的情况"""
        # 创建一个没有有效移动的状态
        self.env.board.board.fill(0)
        self.env.board.board[2, 2] = 1
        self.env.board.board[3, 3] = -1
        # 这种状态下当前玩家可能没有有效移动
        
        mask = self.env.get_valid_moves_mask()
        
        # 如果没有有效移动，pass应该是可用的
        if np.sum(mask[:-1]) == 0:  # 除了pass之外没有移动
            assert mask[-1] == 1  # pass应该可用
    
    def test_extreme_small_board(self):
        """测试极小棋盘"""
        small_env = OthelloEnv(size=4)  # 4x4棋盘
        
        # 应该能正常初始化
        observation = small_env.reset()
        assert observation.shape == (3, 4, 4)
        
        # 应该有有效移动
        mask = small_env.get_valid_moves_mask()
        assert len(mask) == 4 * 4 + 1
        assert np.sum(mask) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])