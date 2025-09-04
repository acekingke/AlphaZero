import numpy as np

class OthelloBoard:
    """
    This class implements the Othello board and game rules.
    """
    # Directions: N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    def __init__(self, size=6):
        """
        Initialize the Othello board.
        
        Args:
            size: The size of the board (default 8x8)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        
        # Initialize the board with the four center pieces
        mid = size // 2
        self.board[mid-1][mid-1] = 1  # White
        self.board[mid][mid] = 1      # White
        self.board[mid-1][mid] = -1   # Black
        self.board[mid][mid-1] = -1   # Black
        
        # Black goes first
        self.current_player = -1
        self.passed = False
        self.done = False
        self.winner = None
    
    def get_valid_moves(self):
        """
        Get all valid moves for the current player.
        
        Returns:
            List of (row, col) tuples representing valid moves
        """
        valid_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self._is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves
    
    def _is_valid_move(self, row, col):
        """
        Check if placing a piece at (row, col) is a valid move.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Check bounds first to prevent IndexError
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
            
        # The cell must be empty
        if self.board[row][col] != 0:
            return False
        
        # Must capture at least one opponent's piece
        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            if not (0 <= r < self.size and 0 <= c < self.size):
                continue
                
            if self.board[r][c] == -self.current_player:  # Opponent's piece
                r += dr
                c += dc
                
                # Continue in this direction
                while 0 <= r < self.size and 0 <= c < self.size:
                    if self.board[r][c] == 0:  # Empty cell
                        break
                    if self.board[r][c] == self.current_player:  # Found our own piece
                        return True
                    r += dr
                    c += dc
                    
        return False
    
    def make_move(self, row, col):
        """
        Place a piece at the specified position and flip captured pieces.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if the move was successful, False if it was invalid
        """
        if not self._is_valid_move(row, col):
            return False
        
        # Place the piece
        self.board[row][col] = self.current_player
        
        # Flip captured pieces
        for dr, dc in self.DIRECTIONS:
            self._flip_direction(row, col, dr, dc)
        
        # Switch player
        self.current_player *= -1
        self.passed = False
        
        # Check if the board is full
        if np.all(self.board != 0):
            self.done = True
            self._determine_winner()
            return True
            
        # Check if the game is over or if the next player must pass
        if not self.get_valid_moves():
            self.current_player *= -1 # The other player has no moves, so current player plays again
            if not self.get_valid_moves(): # if current player also has no moves
                self.done = True
                self._determine_winner()
        
        return True
    
    def _flip_direction(self, row, col, dr, dc):
        """
        Flip pieces in a specific direction if they should be flipped.
        
        Args:
            row: Starting row index
            col: Starting column index
            dr: Row direction
            dc: Column direction
        """
        to_flip = []
        r, c = row + dr, col + dc
        
        # Collect pieces to flip
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == -self.current_player:
            to_flip.append((r, c))
            r += dr
            c += dc
        
        # If we found a piece of the current player, flip all pieces in between
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player:
            for flip_r, flip_c in to_flip:
                self.board[flip_r][flip_c] = self.current_player
    
    def pass_turn(self):
        """
        Pass the turn to the opponent.
        
        Returns:
            True if the pass is successful, False otherwise
        """
        # If game is already done, cannot pass
        if self.done:
            return False
            
        # If there are valid moves, cannot pass
        if len(self.get_valid_moves()) > 0:
            return False
            
        # First pass
        if not self.passed:
            self.passed = True
            self.current_player = -self.current_player
            
            # If the next player has valid moves, continue
            if len(self.get_valid_moves()) > 0:
                return True
                
            # If the next player has no valid moves either, end the game
            self.done = True
            self._determine_winner()
            return True
        
        # Second consecutive pass
        self.done = True
        self._determine_winner()
        return False
    
    def _check_game_over(self):
        """Check if the game is over."""
        # Game is over if no player has valid moves
        if not self.get_valid_moves():
            self.current_player *= -1
            if not self.get_valid_moves():
                self.done = True
                self._determine_winner()
            self.current_player *= -1 # switch back
        
        # Game is also over if the board is full
        if np.all(self.board != 0):
            self.done = True
            self._determine_winner()
        
    def _determine_winner(self):
        """Determine the winner based on piece count."""
        white_count = np.sum(self.board == 1)
        black_count = np.sum(self.board == -1)
        
        if white_count > black_count:
            self.winner = 1  # White wins
        elif black_count > white_count:
            self.winner = -1  # Black wins
        else:
            self.winner = 0  # Draw
    
    def get_state(self):
        """
        Get the current state of the board as a numpy array.
        
        Returns:
            A numpy array representing the board state
        """
        return self.board.copy()
    
    def get_observation(self):
        """
        Get observation for neural network input.
        
        Returns:
            A 3D numpy array with shape (3, size, size) where:
            - Channel 0 represents current player's pieces
            - Channel 1 represents opponent's pieces
            - Channel 2 is all ones if current player is Black (-1), all zeros if White (1)
        """
        observation = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Current player's pieces
        observation[0] = (self.board == self.current_player).astype(np.float32)
        # Opponent's pieces
        observation[1] = (self.board == -self.current_player).astype(np.float32)
        # Current player indicator (fill with 1.0 if player is Black, 0.0 if White)
        observation[2] = np.full((self.size, self.size), 1.0 if self.current_player == -1 else 0.0, dtype=np.float32)
        
        return observation
    
    def get_canonical_state(self):
        """
        Get the canonical state from the perspective of the current player.
        
        Returns:
            A numpy array representing the board from current player's perspective
        """
        return self.board * self.current_player
    
    def is_done(self):
        """Check if the game is done."""
        return self.done
    
    def get_winner(self):
        """Get the winner of the game."""
        return self.winner
    
    def __str__(self):
        """String representation of the board."""
        symbols = {0: '.', 1: 'W', -1: 'B'}
        s = '  ' + ' '.join([str(i) for i in range(self.size)]) + '\n'
        for i in range(self.size):
            s += str(i) + ' ' + ' '.join([symbols[self.board[i][j]] for j in range(self.size)]) + '\n'
        
        s += f"Current player: {'Black' if self.current_player == -1 else 'White'}\n"
        s += f"Valid moves: {self.get_valid_moves()}\n"
        
        return s
    
    def get_action_space_size(self):
        """Get the size of the action space."""
        return self.size * self.size + 1  # All positions + pass

class OthelloEnv:
    """
    Othello environment that follows a gym-like interface.
    """
    def __init__(self, size=6):
        self.size = size
        self.board = None
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            The initial observation
        """
        self.board = OthelloBoard(size=self.size)
        return self.board.get_observation()
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: An integer in range [0, size*size] where:
                   - 0 to size*size-1 corresponds to placing a piece at (action // size, action % size)
                   - size*size corresponds to passing
        
        Returns:
            observation: The observation after taking the action
            reward: The reward (-1, 0, 1)
            done: Whether the game is done
            info: Additional information (valid_moves)
        """
        was_done = self.board.is_done()
        
        # Remember who made the move before player switches
        player_who_moved = self.board.current_player
        
        if action == self.size * self.size:  # Pass
            if len(self.board.get_valid_moves()) > 0:
                # Invalid pass
                return self.board.get_observation(), -10, True, {"valid_moves": self.board.get_valid_moves()}
            # For pass, we need to preserve who actually made the pass action
            # because pass_turn() will switch the current_player
            pass_player = self.board.current_player
            success = self.board.pass_turn()
            if not success and not was_done:
                # Pass attempt failed, but game was not done
                return self.board.get_observation(), -10, True, {"valid_moves": self.board.get_valid_moves()}
            # Use the player who actually made the pass for reward calculation
            player_who_moved = pass_player
        else:
            row, col = action // self.size, action % self.size
            success = self.board.make_move(row, col)
            if not success:
                # Invalid move
                return self.board.get_observation(), -10, True, {"valid_moves": self.board.get_valid_moves()}
        
        # Check if the game is done
        done = self.board.is_done()
        reward = 0
        
        if done:
            winner = self.board.get_winner()
            if winner == 0:  # Draw
                reward = 0
            elif winner == player_who_moved:  # Player who just moved wins
                reward = 1
            else:  # Player who just moved loses
                reward = -1
        
        return self.board.get_observation(), reward, done, {"valid_moves": self.board.get_valid_moves()}
    
    def get_valid_moves_mask(self):
        """
        Get a binary mask of valid moves.
        
        Returns:
            A binary numpy array of shape (action_space_size,) where 1 indicates a valid move
        """
        mask = np.zeros(self.size * self.size + 1, dtype=np.int8)
        valid_moves = self.board.get_valid_moves()
        
        for row, col in valid_moves:
            mask[row * self.size + col] = 1
        
        # If no valid moves, allow pass
        if len(valid_moves) == 0:
            mask[self.size * self.size] = 1
            
        return mask
    
    def get_action_from_coords(self, row, col):
        """Convert (row, col) coordinates to action number."""
        return row * self.size + col
    
    def get_coords_from_action(self, action):
        """Convert action number to (row, col) coordinates."""
        if action == self.size * self.size:
            return None  # Pass action
        return action // self.size, action % self.size
    
    def get_board(self):
        """Return the current board object."""
        return self.board
    
    def render(self):
        """Render the current board state."""
        print(self.board)