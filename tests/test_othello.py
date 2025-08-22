import unittest
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.othello import OthelloBoard, OthelloEnv

class TestOthelloBoard(unittest.TestCase):
    
    def test_initialization(self):
        """Test if the board is initialized correctly with the four center pieces."""
        board = OthelloBoard(size=8)
        state = board.get_state()
        
        # Check the four center pieces
        self.assertEqual(state[3][3], 1)  # White
        self.assertEqual(state[4][4], 1)  # White
        self.assertEqual(state[3][4], -1) # Black
        self.assertEqual(state[4][3], -1) # Black
        
        # Check current player is Black
        self.assertEqual(board.current_player, -1)
        
        # Check the game is not done
        self.assertFalse(board.done)
        self.assertIsNone(board.winner)
    
    def test_valid_moves(self):
        """Test if the valid moves are correctly identified."""
        board = OthelloBoard(size=8)
        valid_moves = board.get_valid_moves()
        
        # Standard opening moves for Black in 8x8 Othello
        expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]
        
        # Sort both lists to ensure order doesn't matter
        self.assertEqual(sorted(valid_moves), sorted(expected_moves))
    
    def test_make_move(self):
        """Test making a valid move and capturing pieces."""
        board = OthelloBoard(size=8)
        
        # Black (current player) places at (2, 3)
        success = board.make_move(2, 3)
        self.assertTrue(success)
        
        # Check if the piece was placed correctly
        state = board.get_state()
        self.assertEqual(state[2][3], -1)
        
        # Check if the white piece was flipped
        self.assertEqual(state[3][3], -1)
        
        # Check that it's now White's turn
        self.assertEqual(board.current_player, 1)
    
    def test_invalid_move(self):
        """Test making an invalid move."""
        board = OthelloBoard(size=8)
        
        # Try to place at a non-capturing position
        success = board.make_move(0, 0)
        self.assertFalse(success)
        
        # Current player should still be Black
        self.assertEqual(board.current_player, -1)
    
    def test_pass_turn(self):
        """Test passing a turn."""
        board = OthelloBoard(size=8)
        
        board.pass_turn()
        
        # Current player should now be White
        self.assertEqual(board.current_player, 1)
        self.assertTrue(board.passed)
        
        # If both players pass, the game should be over
        board.pass_turn()
        self.assertTrue(board.done)
    
    def test_game_over(self):
        """Test game over condition and winner determination."""
        board = OthelloBoard(size=4)  # Using a smaller board for this test
        
        # Manually set up a final board state where Black wins
        board.board = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, 1],
            [-1, 1, 1, 1],
            [-1, -1, -1, -1]
        ])
        
        # Check if the winner is correctly determined
        board._determine_winner()
        self.assertEqual(board.winner, -1)  # Black should win
    
    def test_get_observation(self):
        """Test if the observation is correctly formatted for the neural network."""
        board = OthelloBoard(size=8)
        obs = board.get_observation()
        
        # Check the shape of the observation
        self.assertEqual(obs.shape, (3, 8, 8))
        
        # Check the contents of the observation layers
        # Current player (Black) pieces
        self.assertEqual(np.sum(obs[0]), 2)  # 2 Black pieces initially
        # Opponent (White) pieces
        self.assertEqual(np.sum(obs[1]), 2)  # 2 White pieces initially
        # Current player indicator (1.0 if Black, 0.0 if White)
        self.assertEqual(np.sum(obs[2]), 64)  # All 1.0 for Black

    def test_canonical_state(self):
        """Test the canonical state representation."""
        board = OthelloBoard(size=8)
        canonical = board.get_canonical_state()
        
        # Black pieces should be 1, White pieces should be -1
        self.assertEqual(canonical[3][4], 1)   # Black becomes 1
        self.assertEqual(canonical[4][3], 1)   # Black becomes 1
        self.assertEqual(canonical[3][3], -1)  # White becomes -1
        self.assertEqual(canonical[4][4], -1)  # White becomes -1

class TestOthelloEnv(unittest.TestCase):
    
    def test_initialization(self):
        """Test if the environment is initialized correctly."""
        env = OthelloEnv(size=8)
        obs = env.reset()
        
        # Check the shape of the observation
        self.assertEqual(obs.shape, (3, 8, 8))
    
    def test_step_valid_move(self):
        """Test taking a valid step in the environment."""
        env = OthelloEnv(size=8)
        env.reset()
        
        # Convert (2, 3) to action number (one of the valid starting moves)
        action = env.get_action_from_coords(2, 3)
        
        # Take the action
        obs, reward, done, info = env.step(action)
        
        # Check that the observation is valid
        self.assertEqual(obs.shape, (3, 8, 8))
        
        # Game shouldn't be done after one move
        self.assertFalse(done)
        
        # Check that valid_moves are returned in info
        self.assertIn("valid_moves", info)
        
        # Current player should now be White (1)
        self.assertEqual(env.board.current_player, 1)
    
    def test_step_invalid_move(self):
        """Test taking an invalid step in the environment."""
        env = OthelloEnv(size=8)
        env.reset()
        
        # Convert (0, 0) to action number (an invalid move)
        action = env.get_action_from_coords(0, 0)
        
        # Take the action
        obs, reward, done, info = env.step(action)
        
        # Game should be marked as done for an invalid move
        self.assertTrue(done)
        
        # Reward should be negative for an invalid move
        self.assertEqual(reward, -10)
    
    def test_pass_action(self):
        """Test the pass action in the environment."""
        env = OthelloEnv(size=8)
        env.reset()
        
        # Pass action is size*size
        action = 8 * 8
        
        # Take the pass action
        obs, reward, done, info = env.step(action)
        
        # Game shouldn't be done after one pass
        self.assertFalse(done)
        
        # Current player should now be White (1)
        self.assertEqual(env.board.current_player, 1)
    
    def test_valid_moves_mask(self):
        """Test the valid moves mask."""
        env = OthelloEnv(size=8)
        env.reset()
        
        mask = env.get_valid_moves_mask()
        
        # Should be of length size*size+1 (including pass action)
        self.assertEqual(len(mask), 8*8+1)
        
        # Sum of valid moves should be 4 (the standard opening moves)
        self.assertEqual(np.sum(mask), 4)
        
        # Check specific positions that should be valid
        self.assertEqual(mask[env.get_action_from_coords(2, 3)], 1)
        self.assertEqual(mask[env.get_action_from_coords(3, 2)], 1)
        self.assertEqual(mask[env.get_action_from_coords(4, 5)], 1)
        self.assertEqual(mask[env.get_action_from_coords(5, 4)], 1)
        
        # Pass should not be valid at the start
        self.assertEqual(mask[8*8], 0)
    
    def test_action_coords_conversion(self):
        """Test conversion between actions and coordinates."""
        env = OthelloEnv(size=8)
        
        # Test action to coords
        self.assertEqual(env.get_coords_from_action(19), (2, 3))
        
        # Test coords to action
        self.assertEqual(env.get_action_from_coords(2, 3), 19)
        
        # Test pass action
        self.assertIsNone(env.get_coords_from_action(64))

if __name__ == '__main__':
    unittest.main()