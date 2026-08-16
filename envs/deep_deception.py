"""
Deep Deception Grid (hardened 2D deceptive gridworld)
======================================================

This is the CURRENT environment for the confirmatory experiment. It supersedes
DeceptiveGrid2D (kept in this repo for provenance only).

WHY IT WAS HARDENED:
On the earlier 1-cell-barrier version (DeceptiveGrid2D), epsilon-greedy scored
0.987 -- the task was solvable by random walk, so it could not discriminate
between exploration methods at all. An environment every method solves measures
nothing. The barrier was thickened to 3 cells (a "ring" of k concentric square
rings) and the basin enlarged, so crossing requires SUSTAINED directed movement
against a value gradient, not a lucky wander.

FAIRNESS NOTE: this hardening is mechanism-agnostic. It penalises undirected
exploration in general and rewards ANY method that can sustain a committed
direction -- count-based, novelty search, RND and TEO can all in principle
benefit. It is not shaped around TEO's specific update rule, and the
FixedDirection control is retained to prove the goal still cannot be reached by
a hardcoded direction.

Layout logic (15x15 default):
  - start: dead centre
  - decoy: one cell from start, one-time +2.0 reward (prevents reward-farming
    by oscillating on it — repeat visits give nothing)
  - ring_cells: a k-cell-thick square ring at radius (centre - k - 1 + 1 .. )
    fully enclosing the basin (verified by flood-fill in tests/)
  - goal: one of 4 corners, RANDOMISED PER SEED so no fixed-direction policy
    can solve more than ~1/4 of seeds

`true_barrier_cost()` and `optimal_return()` are exposed for OFFLINE VALIDATION
ONLY. No agent or exploration strategy may read them -- they must estimate
barrier height from their own value function, never from the environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# action index -> (dx, dy)
ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
}


class DeepDeceptionGrid(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size: int = 15, max_steps: int = 300,
                 barrier_thickness: int = 3, valley_penalty: float = 0.4,
                 decoy_reward: float = 2.0, goal_reward: float = 10.0,
                 step_cost: float = 0.01, goal_corner: int | None = None,
                 seed: int | None = None):
        super().__init__()
        assert size % 2 == 1 and size >= 11
        self.size = size
        self.max_steps = max_steps
        self.k = barrier_thickness
        self.valley_penalty = valley_penalty
        self.decoy_reward = decoy_reward
        self.goal_reward = goal_reward
        self.step_cost = step_cost

        self._rng = np.random.default_rng(seed)
        self.centre = size // 2
        self.start = (self.centre, self.centre)
        self.decoy = (self.centre, self.centre - 1)

        # thick ring: k concentric rings starting at inner radius
        self.inner_r = self.centre - self.k - 1
        self.ring_cells = set()
        for t in range(self.k):
            self.ring_cells |= self._ring(self.inner_r + 1 + t)

        if goal_corner is None:
            goal_corner = int(self._rng.integers(0, 4))
        self.goal_corner = goal_corner
        self.goal = self._corner(goal_corner)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

        self._pos = self.start
        self._steps = 0
        self._decoy_collected = False

    def _ring(self, r):
        c, cells = self.centre, set()
        for i in range(c - r, c + r + 1):
            cells |= {(i, c - r), (i, c + r), (c - r, i), (c + r, i)}
        return cells

    def _corner(self, k):
        lo, hi = 0, self.size - 1
        return {0: (lo, lo), 1: (hi, lo), 2: (lo, hi), 3: (hi, hi)}[k]

    def _obs(self):
        x, y = self._pos
        return np.array([x / (self.size - 1), y / (self.size - 1)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self.start
        self._steps = 0
        self._decoy_collected = False
        return self._obs(), {}

    def step(self, action):
        self._steps += 1
        dx, dy = ACTIONS[int(action)]
        x, y = self._pos
        self._pos = (int(np.clip(x + dx, 0, self.size - 1)),
                     int(np.clip(y + dy, 0, self.size - 1)))

        reward = -self.step_cost
        terminated = False
        if self._pos == self.goal:
            reward += self.goal_reward
            terminated = True
        elif self._pos == self.decoy:
            if not self._decoy_collected:
                reward += self.decoy_reward
                self._decoy_collected = True
        elif self._pos in self.ring_cells:
            reward -= self.valley_penalty

        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {
            "pos": self._pos, "at_goal": self._pos == self.goal,
            "at_decoy": self._pos == self.decoy,
            "in_valley": self._pos in self.ring_cells,
            "goal_corner": self.goal_corner,
        }

    def true_barrier_cost(self):
        """Offline validation only -- never exposed to agents."""
        return self.valley_penalty * self.k

    def optimal_return(self):
        x0, y0 = self.start
        gx, gy = self.goal
        dxp, dyp = self.decoy
        steps = (abs(dxp - x0) + abs(dyp - y0)) + (abs(gx - dxp) + abs(gy - dyp))
        return (-self.step_cost * steps + self.decoy_reward
                - self.valley_penalty * self.k + self.goal_reward)
