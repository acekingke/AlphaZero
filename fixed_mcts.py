#!/usr/bin/env python3

import math
import numpy as np
import torch

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
        # Find the child with the highest UCB score
        best_score = -float('inf')
        best_action = -1
        
        # Sum of all visit counts of direct children
        sum_visits = sum(child.visit_count for child in self.children.values())
        
        for action, child in self.children.items():
            # UCB score calculation
            # Exploration: encourages exploring nodes with low visit counts
            # Exploitation: encourages visiting nodes with high value
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
            if prob > 0:  # Only add children for possible actions
                self.children[action] = Node(prob)

class FixedMCTS:
    """Fixed MCTS implementation with corrected value backpropagation."""
    
    def __init__(self, model, c_puct=1.0, num_simulations=100, dirichlet_alpha=0.3, dirichlet_weight=0.25):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        
    def search(self, state, env, temperature=1.0, add_noise=False):
        """
        Perform MCTS search starting from the given state.
        """
        # Create root node
        root = Node(0)
        root_player = env.board.current_player  # Remember who's turn it is at root
        
        # Get the initial policy and value from the neural network
        observation = env.board.get_observation()
        tensor_obs = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            try:
                device = next(self.model.parameters()).device
                if tensor_obs.device != device:
                    tensor_obs = tensor_obs.to(device)
                
                policy_logits, value = self.model(tensor_obs)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"Error in MCTS neural network evaluation: {e}")
                policy = np.ones(env.get_action_space_size()) / env.get_action_space_size()
        
        # Apply a mask for valid actions
        valid_moves_mask = env.get_valid_moves_mask()
        masked_policy = policy * valid_moves_mask
        
        # Renormalize the policy
        policy_sum = np.sum(masked_policy)
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            masked_policy = valid_moves_mask / np.sum(valid_moves_mask)
        
        # Add Dirichlet noise if requested
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * np.sum(valid_moves_mask))
            noise_idx = 0
            noisy_policy = np.copy(masked_policy)
            
            for i in range(len(masked_policy)):
                if valid_moves_mask[i] == 1:
                    noisy_policy[i] = masked_policy[i] * (1 - self.dirichlet_weight) + noise[noise_idx] * self.dirichlet_weight
                    noise_idx += 1
            
            root.expand(state, noisy_policy)
        else:
            root.expand(state, masked_policy)
        
        # Perform simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            env_copy = self._copy_env(env)
            
            # Selection
            while node.expanded():
                action = node.select_child(self.c_puct)
                env_copy.step(action)
                
                if action in node.children:
                    node = node.children[action]
                    search_path.append(node)
                else:
                    break
            
            # Get game state after selection
            observation = env_copy.board.get_observation()
            game_ended = env_copy.board.is_done()
            
            if game_ended:
                winner = env_copy.board.get_winner()
                if winner == 0:  # Draw
                    value = 0.0
                else:
                    # Value from ROOT player's perspective
                    value = 1.0 if winner == root_player else -1.0
            else:
                # Expansion and evaluation
                tensor_obs = torch.FloatTensor(observation).unsqueeze(0)
                
                with torch.no_grad():
                    try:
                        device = next(self.model.parameters()).device
                        if tensor_obs.device != device:
                            tensor_obs = tensor_obs.to(device)
                        
                        policy_logits, value_tensor = self.model(tensor_obs)
                        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                        network_value = value_tensor.item()
                        
                        # Convert network value to root player's perspective
                        # Network gives value from current player's perspective
                        if env_copy.board.current_player == root_player:
                            value = network_value
                        else:
                            value = -network_value
                            
                    except Exception as e:
                        print(f"Error in MCTS node evaluation: {e}")
                        policy = np.ones(env_copy.get_action_space_size()) / env_copy.get_action_space_size()
                        value = 0.0
                
                # Apply mask for valid moves
                valid_moves_mask = env_copy.get_valid_moves_mask()
                masked_policy = policy * valid_moves_mask
                
                # Renormalize
                policy_sum = np.sum(masked_policy)
                if policy_sum > 0:
                    masked_policy /= policy_sum
                else:
                    masked_policy = valid_moves_mask / np.sum(valid_moves_mask)
                
                node.expand(state, masked_policy)
            
            # Backpropagation - value is already from root's perspective
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                # No value flipping needed since we already converted to root perspective
        
        # Calculate action probabilities based on visit counts
        action_probs = np.zeros(len(masked_policy))
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        # Temperature annealing
        if temperature == 0:
            best_action = np.argmax(action_probs)
            action_probs = np.zeros(len(action_probs))
            action_probs[best_action] = 1
        else:
            action_probs = action_probs ** (1 / temperature)
            if np.sum(action_probs) > 0:
                action_probs /= np.sum(action_probs)
        
        return action_probs
    
    def _copy_env(self, env):
        """Create a copy of the environment for simulation."""
        import copy
        return copy.deepcopy(env)