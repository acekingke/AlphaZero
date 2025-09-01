import unittest
import numpy as np
from env.othello import OthelloBoard, OthelloEnv

class TestOthelloGameOver(unittest.TestCase):
    def setUp(self):
        self.board = OthelloBoard(size=6)
        self.env = OthelloEnv(size=6)

    def test_game_over_board_full(self):
        self.board = OthelloBoard(size=6)
        # Fill the board with alternating pieces
        # This simulates a game where the board becomes full
        self.board.board = np.array([
            [ 1, -1,  1, -1,  1, -1],
            [-1,  1, -1,  1, -1,  1],
            [ 1, -1,  1, -1,  1, -1],
            [-1,  1, -1,  1, -1,  1],
            [ 1, -1,  1, -1,  1, -1],
            [-1,  1, -1,  1, -1,  0]  # Last position empty
        ])
        
        # Make the last move to fill the board
        self.board.current_player = 1
        success = self.board.make_move(5, 5)
        
        self.assertTrue(success, "Last move should be valid")
        self.assertTrue(self.board.is_done(), "Game should be done when board is full")
        self.assertIsNotNone(self.board.get_winner(), "Winner should be determined")

    def test_game_over_no_valid_moves(self):
        self.board = OthelloBoard(size=6)
        # Set up a board where neither player has valid moves
        # but the board is not full
        self.board.board = np.array([
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  0,  0,  1,  1],
            [ 1,  1,  0,  0,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1]
        ])
        
        # Try to make moves for both players
        self.board.current_player = 1  # White's turn
        valid_moves = self.board.get_valid_moves()
        self.assertEqual(len(valid_moves), 0, "White should have no valid moves")
        
        self.board.current_player = -1  # Black's turn
        valid_moves = self.board.get_valid_moves()
        self.assertEqual(len(valid_moves), 0, "Black should have no valid moves")
        
        # Make a pass move to trigger game over check
        self.board.pass_turn()
        self.assertTrue(self.board.is_done(), "Game should be done when neither player has valid moves")

    def test_game_over_consecutive_passes(self):
        self.board = OthelloBoard(size=6)
        # Set up a board where passing is necessary
        self.board.board = np.array([
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  0]  # Only one empty space left
        ])
        self.board.current_player = -1  # Black's turn
        
        # Verify no valid moves before first pass
        self.board.current_player = 1  # White's turn
        self.assertEqual(len(self.board.get_valid_moves()), 0, "Should have no valid moves before first pass")
        
        # First pass
        first_pass = self.board.pass_turn()
        self.assertTrue(first_pass, "First pass should be successful when no valid moves")
        self.assertTrue(self.board.passed, "Board should record the pass")
        self.assertFalse(self.board.is_done(), "Game should not be done after one pass")
        
        # Verify no valid moves before second pass
        self.assertEqual(len(self.board.get_valid_moves()), 0, "Should have no valid moves before second pass")
        
        # Second pass
        second_pass = self.board.pass_turn()
        self.assertFalse(second_pass, "Second pass should return False (game over)")
        self.assertTrue(self.board.is_done(), "Game should be done after consecutive passes")
        self.assertIsNotNone(self.board.get_winner(), "Winner should be determined")
        
        # Try to pass when there are valid moves
        self.board = OthelloBoard(size=6)  # New board with valid moves
        self.assertGreater(len(self.board.get_valid_moves()), 0, "New board should have valid moves")
        invalid_pass = self.board.pass_turn()
        self.assertFalse(invalid_pass, "Should not be able to pass when valid moves exist")

    def test_valid_game_not_over(self):
        self.board = OthelloBoard(size=6)
        # Test initial board is not done
        self.assertFalse(self.board.is_done(), "New game should not be done")
        self.assertIsNone(self.board.get_winner(), "New game should not have a winner")
        
        # Make a valid move
        valid_moves = self.board.get_valid_moves()
        self.assertGreater(len(valid_moves), 0, "Should have valid moves in new game")
        
        row, col = valid_moves[0]
        success = self.board.make_move(row, col)
        
        self.assertTrue(success, "Valid move should be successful")
        self.assertFalse(self.board.is_done(), "Game should not be done after valid move")
        self.assertIsNone(self.board.get_winner(), "Game should not have winner yet")

if __name__ == '__main__':
    unittest.main()