"""Dict-based MCTS matching alpha-zero-general.

State is keyed by canonical board bytes. Statistics (Qsa/Nsa/Ns/Ps/Vs)
persist across moves within a game AND across games within an Arena session
(unless reset_tree() is called explicitly), which is how alpha-zero-general
achieves effective search depth >> num_simulations per move.

Public API (compatible with prior tree-based version):
    search(state, env, temperature, add_noise) -> action_probs
    advance_root(action) -> no-op (state-keyed, nothing to descend)
    reset_tree()         -> clears all dicts; call at session boundaries
    canonical_to_observation(canonical, env) -> shim used by callers
"""
import copy
import math
import numpy as np
import torch

EPS = 1e-8


class Node:
    """Deprecated shim kept so legacy imports don't break.

    Dict-based MCTS does not use Node objects. Kept here only so that
    `from mcts.mcts import Node` still succeeds for old tests.
    """
    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.state = None

    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self):
        return len(self.children) > 0

    def expand(self, state, policy):
        self.state = state
        for a, p in enumerate(policy):
            if p > 0:
                self.children[a] = Node(prior=float(p))


class MCTS:
    """Dict-based MCTS.

    Matches alpha-zero-general (MCTS.py) semantics but adapted to the
    env-based interface used in this project:
      - `env.board.get_canonical_state()` instead of `game.getCanonicalForm`
      - `env.step(action)` handles forced passes internally
      - `env.get_valid_moves_mask()` returns 0/1 vector incl. pass slot
    """

    def __init__(
        self,
        model,
        c_puct=1.0,
        num_simulations=25,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
    ):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        # alpha-zero-general's six dicts, keyed by canonical state bytes
        self.Qsa = {}   # (s_key, a) -> Q value (from s_key's current player's perspective)
        self.Nsa = {}   # (s_key, a) -> visit count
        self.Ns = {}    # s_key -> sum of child visits
        self.Ps = {}    # s_key -> masked+normalized policy prior vector
        self.Vs = {}    # s_key -> valid-moves mask

    # ---- Session management ----

    def reset_tree(self):
        """Clear all statistics. Call when starting a genuinely new session
        (e.g., between arena evaluations of different model pairs).
        Within a single self-play game or within a 40-game arena session,
        do NOT call this — accumulation across games is the whole point."""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Vs.clear()

    def advance_root(self, action):
        """No-op. Dict-keyed storage automatically shares statistics for any
        position reached by any path; no notion of 'root' to descend."""
        return

    # ---- Public search ----

    def canonical_to_observation(self, canonical_state, env):
        """Backward-compat shim — callers still pass `state` into search()
        but we now derive canonical internally from env."""
        return np.array(canonical_state, dtype=np.float32)

    def search(self, state, env, temperature=1.0, add_noise=False):
        """Run num_simulations MCTS simulations from env's current state,
        return action probabilities based on visit counts."""
        canonical = env.board.get_canonical_state()
        s_key = self._state_key(canonical)

        # Ensure root is expanded (stores Ps, Vs, Ns entry for s_key)
        if s_key not in self.Ps:
            self._expand(canonical, env, s_key)

        # Apply Dirichlet noise to root priors if requested. We re-apply each
        # call so self-play gets fresh noise every move (per AlphaZero paper).
        # Note: alpha-zero-general has no Dirichlet noise; this is our extension.
        if add_noise:
            self.Ps[s_key] = self._with_dirichlet_noise(self.Ps[s_key], self.Vs[s_key])

        for _ in range(self.num_simulations):
            self._simulate(env)

        return self._get_action_probabilities(s_key, temperature, env)

    # ---- Internals ----

    @staticmethod
    def _state_key(canonical_state):
        """Use raw bytes of int canonical board as hashable key. Fast and exact."""
        return canonical_state.astype(np.int8).tobytes()

    def _nn_predict(self, canonical_state, env):
        board = torch.FloatTensor(canonical_state.astype(np.float64))
        board = board.view(1, env.board.size, env.board.size)
        self.model.eval()
        with torch.no_grad():
            try:
                device = next(self.model.parameters()).device
                if board.device != device:
                    board = board.to(device)
                log_pi, value = self.model(board)
                policy = torch.exp(log_pi).data.cpu().numpy()[0]
                value = float(value.data.cpu().numpy()[0][0])
            except Exception as e:
                print(f"Error in neural network evaluation: {e}")
                policy = np.ones(env.board.get_action_space_size()) / env.board.get_action_space_size()
                value = 0.0
        return policy, value

    def _expand(self, canonical_state, env, s_key):
        """Evaluate NN at state, mask + normalize, store Ps/Vs/Ns.
        Returns the NN-predicted value (from current player's perspective)."""
        policy, value = self._nn_predict(canonical_state, env)
        valids = env.get_valid_moves_mask()
        masked = policy * valids
        total = float(masked.sum())
        if total > 0:
            masked = masked / total
        elif valids.sum() > 0:
            masked = valids.astype(np.float32) / float(valids.sum())
        else:
            masked = np.ones_like(policy, dtype=np.float32) / len(policy)

        self.Ps[s_key] = masked
        self.Vs[s_key] = valids.astype(np.int8)
        self.Ns[s_key] = 0
        return value

    def _with_dirichlet_noise(self, policy, valids):
        """Return a NEW policy array = (1-w)*policy + w*noise on valid actions."""
        num_valid = int(valids.sum())
        if num_valid == 0:
            return policy
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_valid)
        out = np.copy(policy)
        j = 0
        w = self.dirichlet_weight
        for i in range(len(policy)):
            if valids[i] == 1:
                out[i] = policy[i] * (1 - w) + noise[j] * w
                j += 1
        return out

    def _simulate(self, env):
        """One MCTS simulation: select → (expand | terminal) → backprop.

        Iterative implementation with explicit path stack so we can handle
        perspective flipping across forced passes where current_player
        doesn't change after env.step. That flipping is detected by
        comparing player_before at each step.
        """
        env_copy = copy.deepcopy(env)
        path = []  # list of (s_key, action, player_before_step)

        while True:
            if env_copy.board.is_done():
                winner = env_copy.board.get_winner()
                if winner == 0:
                    v = 0.0
                else:
                    leaf_player = env_copy.board.current_player
                    # Value from leaf's current_player POV. After the final
                    # env.step, current_player has flipped (normally), so
                    # winner == leaf_player means leaf_player was the LOSER
                    # in the more common sense... but the canonical POV is
                    # that of whoever is TO MOVE at this (terminal) node.
                    # If winner == leaf_player → from their POV, +1.
                    v = 1.0 if winner == leaf_player else -1.0
                break

            canonical = env_copy.board.get_canonical_state()
            s_key = self._state_key(canonical)

            if s_key not in self.Ps:
                # Leaf: expand and take NN's value
                v = self._expand(canonical, env_copy, s_key)
                break

            # PUCT action selection
            valids = self.Vs[s_key]
            Ns_s = self.Ns[s_key]
            Ps_s = self.Ps[s_key]
            sqrt_Ns = math.sqrt(Ns_s + EPS)

            best_score = -float("inf")
            best_action = -1
            for a in range(len(valids)):
                if valids[a] == 0:
                    continue
                if (s_key, a) in self.Qsa:
                    q = self.Qsa[(s_key, a)]
                    u = self.c_puct * Ps_s[a] * math.sqrt(Ns_s) / (1 + self.Nsa[(s_key, a)])
                else:
                    q = 0.0
                    u = self.c_puct * Ps_s[a] * sqrt_Ns
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = a

            player_before = env_copy.board.current_player
            path.append((s_key, best_action, player_before))
            env_copy.step(best_action)

        # Backprop. `v` is from the leaf's current_player POV.
        # Walking from leaf back to root, we flip sign at every step whose
        # parent_player differs from the perspective we currently hold.
        leaf_player = env_copy.board.current_player
        current_v = v
        prev_player = leaf_player

        for s_key, a, player_at_s in reversed(path):
            if player_at_s != prev_player:
                current_v = -current_v
            if (s_key, a) in self.Qsa:
                n_old = self.Nsa[(s_key, a)]
                self.Qsa[(s_key, a)] = (n_old * self.Qsa[(s_key, a)] + current_v) / (n_old + 1)
                self.Nsa[(s_key, a)] = n_old + 1
            else:
                self.Qsa[(s_key, a)] = current_v
                self.Nsa[(s_key, a)] = 1
            self.Ns[s_key] = self.Ns.get(s_key, 0) + 1
            prev_player = player_at_s

    def _get_action_probabilities(self, s_key, temperature, env):
        action_space_size = env.board.get_action_space_size()
        counts = np.zeros(action_space_size, dtype=np.float64)
        for a in range(action_space_size):
            key = (s_key, a)
            if key in self.Nsa:
                counts[a] = self.Nsa[key]

        if counts.sum() == 0:
            valids = env.get_valid_moves_mask()
            if valids.sum() > 0:
                return valids.astype(np.float32) / float(valids.sum())
            return np.ones(action_space_size, dtype=np.float32) / action_space_size

        if temperature == 0:
            max_val = counts.max()
            best_actions = np.flatnonzero(counts == max_val)
            best = int(np.random.choice(best_actions))
            probs = np.zeros(action_space_size, dtype=np.float32)
            probs[best] = 1.0
            return probs

        safe_temp = max(temperature, 1e-6)
        if counts.max() > 1000:
            counts = counts / counts.max()
        counts_pow = counts ** (1.0 / safe_temp)
        return counts_pow / counts_pow.sum()


# Backward-compat alias
SimplifiedMCTS = MCTS
