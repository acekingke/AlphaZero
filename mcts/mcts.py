import math
import numpy as np
import torch
import copy

class Node:
    """Node in the MCTS tree."""
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
    
    def expanded(self):
        """Check if the node is expanded."""
        return len(self.children) > 0
    
    def value(self):
        """Get the mean value of the node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct):
        """Select a child according to the PUCT formula.

        PUCT: score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        where N(s) is the parent's visit count. On the very first simulation
        when no children have been visited yet, using sum_of_children_visits
        would give sqrt(0) = 0, completely zeroing the exploration term and
        picking the lowest-indexed action deterministically. We use max(1,
        sum_visits) to ensure the exploration term is non-zero on the first
        visit, so priors (including Dirichlet noise) actually bias the initial
        selection. This matches the intent of the AlphaZero paper which uses
        sqrt(parent_N) where parent_N >= 1 whenever we're selecting.
        """
        best_score = -float('inf')
        best_action = -1

        # Parent's visit count (proxy: sum of children visits).
        # max(1, ...) ensures exploration term is nonzero on first pass.
        sum_visits = sum(child.visit_count for child in self.children.values())
        sqrt_parent_visits = math.sqrt(max(1, sum_visits))

        for action, child in self.children.items():
            u = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            q = child.value()
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    
    def expand(self, state, policy):
        """Expand the node with the given policy."""
        self.state = state
        for action, prob in enumerate(policy):
            if prob > 0:
                self.children[action] = Node(prior=prob)

class MCTS:
    """
    Simplified MCTS implementation with canonical state support and Tree Visitor pattern.
    This version combines the simplicity of the git version with important improvements:
    1. Canonical state support for consistent training
    2. Tree Visitor pattern to avoid stack overflow
    3. Clean, maintainable architecture
    """
    
    def __init__(self, model, c_puct=2.0, num_simulations=800, dirichlet_alpha=0.5, dirichlet_weight=0.3):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        
    def search(self, state, env, temperature=1.0, add_noise=False):
        """
        Perform MCTS search starting from the given state.
        
        Args:
            state: Current observation state (already converted from canonical state)
            env: Game environment
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise to the prior probabilities
            
        Returns:
            Action probabilities based on visit counts
        """
        # Create root node
        root = Node(0)
        
        # Get canonical state for MCTS tree expansion
        canonical_state = env.board.get_canonical_state()
        
        # Use canonical state for policy evaluation (converted internally)
        policy, value = self._evaluate_state(canonical_state, env)
        
        # Add Dirichlet noise if requested
        if add_noise:
            policy = self._add_dirichlet_noise(policy, env)
        
        # Expand root node
        root.expand(canonical_state, policy)
        
        # Perform MCTS simulations using Tree Visitor pattern (iterative)
        for _ in range(self.num_simulations):
            self._simulate_iterative(root, env)
        
        # Calculate action probabilities based on visit counts
        return self._get_action_probabilities(root, temperature, env)
    
    def _evaluate_state(self, canonical_state, env):
        """Evaluate state using neural network with canonical state representation."""
        # Convert canonical state to observation format expected by neural network
        observation = self.canonical_to_observation(canonical_state, env)
        tensor_obs = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            try:
                device = next(self.model.parameters()).device
                if tensor_obs.device != device:
                    tensor_obs = tensor_obs.to(device)
                
                policy_logits, value = self.model(tensor_obs)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                value = value.item()
            except Exception as e:
                print(f"Error in neural network evaluation: {e}")
                # Fallback to random policy
                policy = np.ones(env.board.get_action_space_size()) / env.board.get_action_space_size()
                value = 0.0
        
        # Apply mask for valid actions
        valid_moves_mask = env.get_valid_moves_mask()
        masked_policy = policy * valid_moves_mask
        
        # Renormalize the policy
        policy_sum = np.sum(masked_policy)
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # If no valid moves, this should not happen in normal gameplay
            # but provide fallback anyway
            if np.sum(valid_moves_mask) > 0:
                masked_policy = valid_moves_mask / np.sum(valid_moves_mask)
            else:
                # Ultimate fallback - uniform distribution
                masked_policy = np.ones(env.board.get_action_space_size()) / env.board.get_action_space_size()
        
        return masked_policy, value
    
    def canonical_to_observation(self, canonical_state, env):
        """
        Convert canonical state to observation format for neural network.
        Since we're using canonical state, current player is always +1.
        """
        size = env.board.size
        observation = np.zeros((3, size, size), dtype=np.float32)
        
        # Current player's pieces (always +1 in canonical state)
        observation[0] = (canonical_state == 1).astype(np.float32)
        # Opponent's pieces (always -1 in canonical state)
        observation[1] = (canonical_state == -1).astype(np.float32)
        # Current player indicator (always 1 since current player is +1 in canonical state)
        observation[2] = np.ones((size, size), dtype=np.float32)
        
        return observation
    
    def _add_dirichlet_noise(self, policy, env):
        """Add Dirichlet noise to the policy for exploration."""
        valid_moves_mask = env.get_valid_moves_mask()
        noise = np.random.dirichlet([self.dirichlet_alpha] * np.sum(valid_moves_mask))
        noise_idx = 0
        noisy_policy = np.copy(policy)
        
        for i in range(len(policy)):
            if valid_moves_mask[i] == 1:
                noisy_policy[i] = policy[i] * (1 - self.dirichlet_weight) + noise[noise_idx] * self.dirichlet_weight
                noise_idx += 1
        
        return noisy_policy
    
    def _simulate_iterative(self, root, env):
        """
        Perform one MCTS simulation using Tree Visitor pattern (iterative).

        Value perspective convention: every value stored in a node is from
        the perspective of the player whose turn it is at that state. This
        matches the canonical_state convention used by the neural network.

        Backpropagation flips the sign ONLY when the actual player switches
        between nodes. In Othello, a move can trigger a forced pass (opponent
        has no valid moves), in which case current_player stays the same and
        the sign should NOT flip for that step.
        """
        # Tree Visitor pattern: explicit stack of (parent_node, action, player_at_parent)
        path_stack = []
        node = root
        env_copy = copy.deepcopy(env)

        # Selection phase - traverse down the tree
        while node.expanded():
            action = node.select_child(self.c_puct)
            # Record the player who is about to act at this parent node.
            # We'll use this during backprop to detect perspective changes.
            player_at_parent = env_copy.board.current_player
            path_stack.append((node, action, player_at_parent))

            env_copy.step(action)

            if action in node.children:
                node = node.children[action]
            else:
                break

        # Evaluation phase — both branches must produce value from the
        # leaf's current player's perspective.
        canonical_state = env_copy.board.get_canonical_state()
        game_ended = env_copy.board.is_done()

        if game_ended:
            # Terminal value from leaf's current player perspective.
            # This matches the NN convention where canonical_state is always
            # from the current player's perspective (current player = +1).
            # Both branches of the if/else must agree on the value reference
            # frame for backpropagation to work correctly.
            winner = env_copy.board.get_winner()
            if winner == 0:  # Draw
                value = 0.0
            else:
                leaf_current_player = env_copy.board.current_player
                value = 1.0 if winner == leaf_current_player else -1.0
        else:
            policy, value = self._evaluate_state(canonical_state, env_copy)
            node.expand(canonical_state, policy)

        # Backpropagation — walk from leaf back to root, flipping the sign
        # only when the player actually changes between adjacent nodes.
        leaf_player = env_copy.board.current_player
        node.value_sum += value
        node.visit_count += 1

        current_value = value
        prev_player = leaf_player

        for parent_node, action, parent_player in reversed(path_stack):
            # If parent's player differs from the child's (prev) player,
            # the perspective flips and we negate. Otherwise (forced pass
            # scenario) the perspective stays the same.
            if parent_player != prev_player:
                current_value = -current_value
            parent_node.value_sum += current_value
            parent_node.visit_count += 1
            prev_player = parent_player
    
    def _get_action_probabilities(self, root, temperature, env):
        """Calculate final action probabilities based on visit counts."""
        # Use environment's action space size for consistency
        action_space_size = env.board.get_action_space_size()
        
        action_probs = np.zeros(action_space_size)
        for action, child in root.children.items():
            if 0 <= action < action_space_size:  # Enhanced bounds check
                action_probs[action] = child.visit_count

        # Edge case: if no children have any visits (e.g., num_simulations=0
        # or expansion failure), fall back to a valid-moves uniform distribution
        # so downstream np.random.choice doesn't crash with "p does not sum to 1".
        if np.sum(action_probs) == 0:
            valid_moves_mask = env.get_valid_moves_mask()
            if np.sum(valid_moves_mask) > 0:
                action_probs = valid_moves_mask.astype(np.float32) / np.sum(valid_moves_mask)
            else:
                # No valid moves and no pass either — degenerate state.
                # Return uniform over entire action space as last resort.
                action_probs = np.ones(action_space_size, dtype=np.float32) / action_space_size
            return action_probs

        # Temperature annealing with numerical stability
        if temperature == 0:  # "Deterministic" selection with random tie-breaking
            # Random tie-breaking among best actions.
            # Matches alpha-zero-general MCTS.getActionProb (temp=0 branch):
            # when multiple actions have equal max visit counts (common for
            # weak networks where MCTS produces near-uniform visit distributions),
            # pick one at random instead of always the lowest-indexed one
            # (which np.argmax does). This is the ONLY source of non-determinism
            # in MCTS with a fixed network, and it's crucial for Arena to
            # produce statistically meaningful results (otherwise all N arena
            # games collapse to 2 unique outcomes).
            # Note: visit counts are integers stored as float, so equality
            # comparison is exact. If you ever change action_probs to involve
            # normalization BEFORE this branch, use np.isclose instead.
            max_val = action_probs.max()
            best_actions = np.flatnonzero(action_probs == max_val)
            best_action = int(np.random.choice(best_actions))
            action_probs = np.zeros(len(action_probs))
            action_probs[best_action] = 1
        else:  # Stochastic selection
            # Prevent numerical overflow by setting minimum temperature
            safe_temperature = max(temperature, 1e-6)

            # Additional numerical stability: normalize large values
            max_val = np.max(action_probs)
            if max_val > 1000:  # Prevent extreme values from causing overflow
                action_probs = action_probs / max_val

            action_probs = action_probs ** (1 / safe_temperature)
            action_probs /= np.sum(action_probs)

        return action_probs
# For backward compatibility
SimplifiedMCTS = MCTS
