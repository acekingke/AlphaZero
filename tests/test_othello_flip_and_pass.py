import unittest
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.othello import OthelloBoard, OthelloEnv

class TestOthelloFlipDirection(unittest.TestCase):
    """
    测试 OthelloBoard._flip_direction 函数的基本功能与边界情况
    """
    
    def test_flip_direction_simple(self):
        """测试基本的翻转功能"""
        board = OthelloBoard(size=6)
        
        # 初始状态
        initial_state = board.board.copy()
        
        # 黑方(-1)放置棋子在(2, 3)，应翻转(3, 3)的白子
        board.make_move(2, 3)
        
        # 确认棋子被正确放置
        self.assertEqual(board.board[2][3], -1)
        # 确认对手棋子被正确翻转
        self.assertEqual(board.board[3][3], -1)
        
        # 创建一个新棋盘，手动测试 _flip_direction 函数
        test_board = OthelloBoard(size=6)
        # 模拟黑方刚刚在(2, 3)放置了棋子
        test_board.board[2][3] = -1
        test_board.current_player = -1
        
        # 手动调用 _flip_direction 函数，方向为(1, 0)（向下）
        test_board._flip_direction(2, 3, 1, 0)
        
        # 验证(3, 3)的白子被翻转为黑子
        self.assertEqual(test_board.board[3][3], -1)
    
    def test_flip_direction_multiple_pieces(self):
        """测试翻转多个棋子"""
        board = OthelloBoard(size=6)
        
        # 设置一个可以翻转多个棋子的局面
        board.board = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        board.current_player = -1  # 黑方
        
        # 手动调用 _flip_direction 函数，翻转(3, 3)到(5, 3)的白子
        board._flip_direction(6, 3, -1, 0)  # 方向为向上
        
        # 验证多个棋子被翻转
        self.assertEqual(board.board[3][3], -1)
        self.assertEqual(board.board[4][3], -1)
        self.assertEqual(board.board[5][3], -1)
    
    def test_flip_direction_no_flip(self):
        """测试不满足翻转条件的情况"""
        board = OthelloBoard(size=6)
        
        # 设置一个不能形成翻转的局面
        board.board = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        board.current_player = -1  # 黑方
        
        # 保存初始状态
        initial_state = board.board.copy()
        
        # 手动调用 _flip_direction 函数
        board._flip_direction(2, 3, 1, 0)  # 方向为向下
        
        # 验证没有棋子被翻转（棋盘应保持不变）
        np.testing.assert_array_equal(board.board, initial_state)
    
    def test_flip_direction_boundary_edge(self):
        """测试边缘位置的翻转"""
        board = OthelloBoard(size=6)
        
        # 设置一个边缘翻转的局面
        board.board = np.array([
            [-1, 1, 1, 1, 1, -1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        board.current_player = -1  # 黑方
        
        # 在(0,0)落子，手动调用 _flip_direction 函数，尝试从(0, 0)向右翻转
        board.board[0][0] = -1
        board._flip_direction(0, 0, 0, 1)  # 方向为向右
        
        # 验证边缘棋子被翻转
        for i in range(1, 5):
            self.assertEqual(board.board[0][i], -1, f"Position (0, {i}) should be flipped to black")
    
    def test_flip_direction_boundary_corner(self):
        """测试角落位置的翻转"""
        board = OthelloBoard(size=6)
        
        # 设置一个角落翻转的局面
        board.board = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        board.current_player = -1  # 黑方
        
        # 手动调用 _flip_direction 函数
        board._flip_direction(0, 0, 1, 1)  # 方向为右下
        
        # 验证角落位置的翻转
        self.assertEqual(board.board[1][1], -1)
    
    def test_flip_direction_diagonal(self):
        """测试对角线方向的翻转"""
        board = OthelloBoard(size=6)
        
        # 设置一个对角线翻转的局面
        board.board = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, -1]
        ])
        board.current_player = 1  # 白方
        
        # 手动调用 _flip_direction 函数，对角线向左上翻转
        board._flip_direction(6, 6, -1, -1)  # 方向为左上
        
        # 验证对角线翻转
        self.assertEqual(board.board[5][5], 1)
    
    def test_flip_direction_long_distance(self):
        """测试长距离翻转"""
        board = OthelloBoard(size=6)
        
        # 设置一个长距离翻转的局面
        board.board = np.array([
            [0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        board.current_player = -1  # 黑方
        
        # 手动调用 _flip_direction 函数，从(7, 0)向右上方翻转
        board._flip_direction(7, 0, -1, 1)  # 方向为右上
        
        # 验证长距离翻转
        for i in range(1, 5):
            self.assertEqual(board.board[7-i][i], -1, f"Position ({7-i}, {i}) should be flipped to black")

class TestOthelloPassTurn(unittest.TestCase):
    """
    测试 OthelloBoard.pass_turn 函数的基本功能与边界情况
    """
    
    def test_pass_turn_simple(self):
        """测试基本的回合传递功能"""
        board = OthelloBoard(size=6)
        
        # 初始玩家是黑方(-1)
        self.assertEqual(board.current_player, -1)
        
        # 执行一次回合传递
        result = board.pass_turn()
        
        # 验证玩家已切换到白方(1)
        self.assertEqual(board.current_player, 1)
        # 验证pass_turn返回True表示游戏继续
        self.assertTrue(result)
        # 验证passed标记已设置
        self.assertTrue(board.passed)
        # 验证游戏未结束
        self.assertFalse(board.done)
    
    def test_pass_turn_twice(self):
        """测试连续两次回合传递导致游戏结束"""
        board = OthelloBoard(size=6)
        
        # 第一次回合传递
        board.pass_turn()
        # 确认当前玩家是白方(1)
        self.assertEqual(board.current_player, 1)
        
        # 第二次回合传递
        result = board.pass_turn()
        
        # 验证游戏已结束
        self.assertTrue(board.done)
        # 验证pass_turn返回False表示游戏结束
        self.assertFalse(result)
        # 确认winner已确定
        self.assertIsNotNone(board.winner)
    
    def test_pass_turn_then_move(self):
        """测试一次回合传递后再下棋"""
        board = OthelloBoard(size=6)
        
        # 初始玩家是黑方(-1)
        self.assertEqual(board.current_player, -1)
        
        # 黑方pass
        board.pass_turn()
        
        # 现在玩家是白方(1)
        self.assertEqual(board.current_player, 1)
        self.assertTrue(board.passed)
        
        # 白方下棋
        success = board.make_move(2, 4) # A valid move for white
        
        # 验证下棋成功
        self.assertTrue(success)
        # 验证passed标记已重置
        self.assertFalse(board.passed)
        # 验证当前玩家已切换回黑方(-1)
        self.assertEqual(board.current_player, -1)
    
    def test_pass_turn_no_valid_moves(self):
        """测试没有有效移动时的回合传递"""
        board = OthelloBoard(size=6)
        
        # 设置一个特殊的棋盘状态，使当前玩家没有有效移动，但对手有
        board.board = np.zeros((6, 6), dtype=np.int8)
        board.board[3][3] = 1  # White
        board.board[2:5, 2:5] = -1  # Black square
        board.board[3][3] = 1  # Restore white piece in the middle
        board.current_player = -1  # 黑方
        
        # 验证黑方没有有效移动
        self.assertEqual(len(board.get_valid_moves()), 0)
        
        # 黑方pass
        result = board.pass_turn()
        
        # 游戏应该继续，轮到白方
        self.assertTrue(result)
        self.assertEqual(board.current_player, 1)
        # 白方应该有棋可走
        self.assertGreater(len(board.get_valid_moves()), 0)
    
    def test_pass_turn_game_over_detection(self):
        """测试通过回合传递检测游戏结束"""
        board = OthelloBoard(size=6)  # 使用小尺寸棋盘便于测试
        
        # 设置一个即将结束的游戏状态
        board.board = np.array([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        board.current_player = 1  # 白方，没有可用的移动
        
        # 验证白方没有有效移动
        self.assertEqual(len(board.get_valid_moves()), 0)
        
        # 白方pass
        board.pass_turn()
        
        # 游戏应该结束，因为黑方也无棋可走
        self.assertTrue(board.done)
        self.assertEqual(board.winner, -1) # 黑棋多，黑胜
    
    def test_pass_turn_with_full_board(self):
        """测试棋盘已满时的回合传递"""
        board = OthelloBoard(size=6)
        
        # 设置一个已满的棋盘
        board.board = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ])
        board.current_player = -1  # 黑方
        
        # 手动设置游戏结束状态
        board.done = True
        board._determine_winner()
        
        # 验证游戏已结束
        self.assertTrue(board.done)
        
        # 在游戏结束后尝试执行回合传递
        result = board.pass_turn()
        
        # 验证pass_turn返回False（游戏已结束）
        self.assertFalse(result)
        # 验证赢家已确定（这种情况下是白方）
        self.assertEqual(board.winner, 1)

if __name__ == '__main__':
    unittest.main()