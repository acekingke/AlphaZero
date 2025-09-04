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
        """Select a child according to the UCB formula."""
        best_score = -float('inf')
        best_action = -1
        
        # Sum of all visit counts of direct children
        sum_visits = sum(child.visit_count for child in self.children.values())
        
        for action, child in self.children.items():
            # UCB score calculation
            u = c_puct * child.prior * math.sqrt(sum_visits) / (1 + child.visit_count)
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
        observation = self._canonical_to_observation(canonical_state, env)
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
            masked_policy = valid_moves_mask / np.sum(valid_moves_mask)
        
        return masked_policy, value
    
    def _canonical_to_observation(self, canonical_state, env):
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
        Perform one MCTS simulation using Tree Visitor pattern (iterative implementation).
        This avoids recursion stack overflow issues.
        """
        # Tree Visitor pattern: use explicit stack to track path
        path_stack = []
        node = root
        env_copy = copy.deepcopy(env)
        
        # Selection phase - traverse down the tree
        while node.expanded():
            action = node.select_child(self.c_puct)
            path_stack.append((node, action))
            
            # Apply action to environment
            env_copy.step(action)
            
            if action in node.children:
                node = node.children[action]
            else:
                break
        
        # Evaluation phase
        canonical_state = env_copy.board.get_canonical_state()
        game_ended = env_copy.board.is_done()
        
        if game_ended:
            # Use game result as value
            winner = env_copy.board.get_winner()
            if winner == 0:  # Draw
                value = 0.0
            else:
                # 正确的价值计算：对于当前玩家的视角
                # 如果当前玩家是获胜者，value=1；否则value=-1
                value = 1.0 if winner == env_copy.board.current_player else -1.0
        else:
            # Expansion and evaluation
            policy, value = self._evaluate_state(canonical_state, env_copy)
            node.expand(canonical_state, policy)
        
        # Backpropagation phase - propagate value up the path
        node.value_sum += value
        node.visit_count += 1
        
        # Backpropagate through the path, flipping value for alternating players
        current_value = -value  # Flip for parent
        for parent_node, action in reversed(path_stack):
            parent_node.value_sum += current_value
            parent_node.visit_count += 1
            current_value = -current_value  # Flip for next parent
    
    def _get_action_probabilities(self, root, temperature, env):
        """Calculate final action probabilities based on visit counts."""
        # Use environment's action space size for consistency
        action_space_size = env.board.get_action_space_size()
        
        action_probs = np.zeros(action_space_size)
        for action, child in root.children.items():
            if action < action_space_size:
                action_probs[action] = child.visit_count
        
        # Temperature annealing
        if temperature == 0:  # Deterministic selection
            if np.sum(action_probs) > 0:
                best_action = np.argmax(action_probs)
                action_probs = np.zeros(len(action_probs))
                action_probs[best_action] = 1
        else:  # Stochastic selection
            if np.sum(action_probs) > 0:
                action_probs = action_probs ** (1 / temperature)
                action_probs /= np.sum(action_probs)
        
        return action_probs
# For backward compatibility
SimplifiedMCTS = MCTS
