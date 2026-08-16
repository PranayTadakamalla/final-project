"""
TEO v3 -- algorithmic redesign.

The v2 operator (agents/strategies.py: TEOClassical/TEOQuantum) underperformed
epsilon-greedy. Diagnosis of why, and the four fixes made here (each is a
genuine mechanism change, not a parameter tweak):

  PROBLEM 1: barrier estimate was noise-dominated.
    A raw value drop over a trailing window fires constantly early in training
    when Q-values are random, so the operator burned its interventions on noise.
  FIX 1: stagnation-gated firing. The operator only considers tunnelling when
    the agent is actually STUCK -- i.e. its recent state distribution has
    collapsed (low positional entropy) AND its value estimate has plateaued.
    This is what "trapped in a local optimum" actually looks like.

  PROBLEM 2: the push target was myopic.
    Choosing the least-visited immediate NEIGHBOUR is a one-step decision; in a
    thick barrier the agent re-enters the basin before crossing.
  FIX 2: frontier targeting. The operator maintains a visitation map, finds the
    least-visited reachable FRONTIER cell, and commits to a multi-step heading
    toward it -- a directed escape, which is the mechanism the project claims.

  PROBLEM 3: pushes were too short to clear a thick barrier.
  FIX 3: push length scales with the estimated barrier width, so a k-cell
    barrier gets a push long enough to actually clear it.

  PROBLEM 4: alpha was fixed, so P_tunnel was mis-scaled for the actual barrier
    magnitudes encountered.
  FIX 4: alpha is calibrated online from a running estimate of observed barrier
    heights, keeping P_tunnel in a useful range instead of saturating at 0 or 1.

  PROBLEM 5 (found after initial v3 testing): one seed fired 716 times and
    thrashed, collapsing goal rate from 1.00 to 0.39 -- the operator was
    interrupting the policy faster than it could learn.
  FIX 5: anti-thrash controls.
    (a) ANNEAL -- firing probability decays as training progresses, exactly as
        epsilon does; a converged agent has nothing left to escape.
    (b) BACKOFF -- if a tunnel event does not improve value, the cooldown grows
        multiplicatively, so unproductive firing self-limits instead of running
        away. A cap on max fires per episode backs this up.

Interface is unchanged from v2, so all baselines and the harness are untouched.
"""
import math
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 4
_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


def _key(obs, bins=15):
    return tuple(int(round(float(v) * (bins - 1))) for v in obs)


class _TEOv3Base:
    def __init__(self, seed=0, stagnation_window=30, entropy_thresh=1.6,
                 plateau_tol=0.05, push_scale=3.0, min_push=6, cooldown=25,
                 bins=15, anneal_episodes=150, max_fires_per_episode=6,
                 backoff_factor=1.8, max_cooldown=200, **kw):
        self.rng = random.Random(seed)
        self.bins = bins
        self.stagnation_window = stagnation_window
        self.entropy_thresh = entropy_thresh
        self.plateau_tol = plateau_tol
        self.push_scale = push_scale
        self.min_push = min_push
        self.base_cooldown = cooldown
        self.cooldown = cooldown

        # --- anti-thrash controls ---
        # Diagnosis: on some seeds the operator fired hundreds of times per run,
        # interrupting the policy faster than it could learn (goal rate collapsed
        # to 0.39 while well-behaved seeds hit 1.00 with <100 fires). Two fixes:
        #   (a) ANNEAL: exploration should decay as the policy converges, exactly
        #       as epsilon does -- a converged agent has nothing left to escape.
        #   (b) BACKOFF: if a tunnel event does not improve value, the cooldown
        #       grows multiplicatively, so unproductive firing self-limits
        #       instead of running away.
        self.anneal_episodes = anneal_episodes
        self.max_fires_per_episode = max_fires_per_episode
        self.backoff_factor = backoff_factor
        self.max_cooldown = max_cooldown
        self._episode = 0
        self._fires_this_episode = 0
        self._cur_cooldown = cooldown
        self._last_fire_value = None

        self.counts = defaultdict(int)
        self.recent_states = deque(maxlen=stagnation_window)
        self.recent_values = deque(maxlen=stagnation_window)
        self.barrier_samples = deque(maxlen=200)

        self._push_remaining = 0
        self._push_target = None
        self._cool = 0
        self.log = []
        self.n_fired = 0
        self.n_stagnant = 0

    def _anneal_factor(self):
        """Firing probability decays as training progresses."""
        return max(0.05, 1.0 - self._episode / max(1, self.anneal_episodes))

    # ---------- stagnation detection ----------
    def _positional_entropy(self):
        if len(self.recent_states) < self.stagnation_window:
            return 99.0
        c = defaultdict(int)
        for s in self.recent_states:
            c[s] += 1
        n = len(self.recent_states)
        return -sum((v / n) * math.log(v / n + 1e-12) for v in c.values())

    def _is_stagnant(self):
        if len(self.recent_values) < self.stagnation_window:
            return False
        ent = self._positional_entropy()
        vals = np.array(self.recent_values)
        half = len(vals) // 2
        drift = abs(vals[half:].mean() - vals[:half].mean())
        return ent < self.entropy_thresh and drift < self.plateau_tol

    # ---------- barrier estimate ----------
    def _estimate_delta(self, value_now):
        if len(self.recent_values) < 8:
            return 0.0
        recent = np.array(self.recent_values)
        return max(0.0, float(recent.max() - value_now))

    def _adaptive_alpha(self):
        """Calibrate alpha so P_tunnel sits in a useful band for the barrier
        magnitudes actually observed, instead of saturating at 0 or 1."""
        if len(self.barrier_samples) < 20:
            return 1.0
        med = float(np.median(self.barrier_samples))
        return math.log(2.0) / max(med, 1e-3)   # P_tunnel ~= 0.5 at median barrier

    # ---------- frontier targeting ----------
    def _frontier_target(self, obs):
        """Least-visited cell within a radius -- a directed escape target, not a
        myopic one-step choice. Uses only the operator's own visitation map."""
        cur = _key(obs, self.bins)
        best, best_score = None, None
        R = 6
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                if abs(dx) + abs(dy) < 3:
                    continue
                cell = (cur[0] + dx, cur[1] + dy)
                if not (0 <= cell[0] < self.bins and 0 <= cell[1] < self.bins):
                    continue
                dist = abs(dx) + abs(dy)
                score = self.counts[cell] + 0.35 * dist   # prefer unvisited, mildly prefer near
                if best_score is None or score < best_score:
                    best, best_score = cell, score
        return best

    def _action_toward(self, obs, target):
        cur = _key(obs, self.bins)
        dx, dy = target[0] - cur[0], target[1] - cur[1]
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2
        return 1 if dy > 0 else 0

    def _p_tunnel(self, delta, obs, value_now):
        raise NotImplementedError

    # ---------- main ----------
    def select_action(self, agent, obs, greedy_action):
        k = _key(obs, self.bins)
        self.counts[k] += 1
        self.recent_states.append(k)
        value_now = agent.value_estimate(obs)
        self.recent_values.append(value_now)

        if self._push_remaining > 0:
            self._push_remaining -= 1
            self.log.append((None, None, False))
            return self._action_toward(obs, self._push_target)

        if self._cool > 0:
            self._cool -= 1
            self.log.append((None, None, False))
            return greedy_action

        if not self._is_stagnant() or self._fires_this_episode >= self.max_fires_per_episode:
            self.log.append((0.0, None, False))
            return greedy_action

        self.n_stagnant += 1
        delta = self._estimate_delta(value_now)
        if delta > 1e-6:
            self.barrier_samples.append(delta)
        p = self._p_tunnel(delta, obs, value_now) * self._anneal_factor()

        fired = False
        if self.rng.random() < p:
            fired = True
            self.n_fired += 1
            self._fires_this_episode += 1
            # adaptive backoff: if the previous tunnel did not improve value,
            # grow the cooldown so unproductive firing self-limits
            if self._last_fire_value is not None:
                if value_now <= self._last_fire_value:
                    self._cur_cooldown = min(self.max_cooldown,
                                             self._cur_cooldown * self.backoff_factor)
                else:
                    self._cur_cooldown = self.base_cooldown
            self._last_fire_value = value_now
            self._push_target = self._frontier_target(obs)
            self._push_remaining = max(self.min_push, int(self.push_scale * max(delta, 0.5))) - 1
            self._cool = int(self._cur_cooldown)
            self._on_fire(obs, value_now, delta)

        self.log.append((delta, p, fired))
        if fired:
            return self._action_toward(obs, self._push_target)
        return greedy_action

    def _on_fire(self, obs, value_now, delta):
        pass

    def end_episode(self):
        self.recent_states.clear()
        self.recent_values.clear()
        self._push_remaining = 0
        self._cool = 0
        self._episode += 1
        self._fires_this_episode = 0


class TEOClassicalV3(_TEOv3Base):
    name = "teo_classical_v3"

    def __init__(self, alpha=None, **kw):
        super().__init__(**kw)
        self.fixed_alpha = alpha

    def _p_tunnel(self, delta, obs, value_now):
        a = self.fixed_alpha if self.fixed_alpha is not None else self._adaptive_alpha()
        return math.exp(-a * delta)


class TEOQuantumV3(_TEOv3Base):
    name = "teo_quantum_v3"

    def __init__(self, lr=0.05, train_every=32, outcome_horizon=12, n_layers=2, **kw):
        super().__init__(**kw)
        from agents.quantum_circuit import QuantumTunnelProbability
        seed = kw.get("seed", 0)
        self.qnet = QuantumTunnelProbability(seed=seed, n_layers=n_layers)
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.train_every = train_every
        self.outcome_horizon = outcome_horizon
        self._pending = deque()
        self._buffer = []
        self._steps = 0
        self.train_losses = []

    def _features(self, delta, obs, value_now):
        # normalise delta against the running median of observed barrier
        # heights (adaptive, mirroring _adaptive_alpha's calibration logic)
        # rather than a fixed constant, so features stay meaningful as the
        # environment/agent produces different barrier magnitudes over training
        med = float(np.median(self.barrier_samples)) if len(self.barrier_samples) > 10 else 1.0
        f_delta = min(delta / max(1e-3, med), 2.0) / 2.0
        f_visits = min(self.counts[_key(obs, self.bins)] / 50.0, 1.0)
        f_value = 1.0 / (1.0 + math.exp(-value_now))
        f_pos = float(np.mean(np.asarray(obs)))
        return torch.tensor([f_delta * math.pi, f_visits * math.pi,
                             f_value * math.pi, f_pos * math.pi],
                            dtype=torch.float32)

    def _p_tunnel(self, delta, obs, value_now):
        self._steps += 1
        feats = self._features(delta, obs, value_now)
        self._last_feats = feats
        with torch.no_grad():
            p = float(self.qnet(feats).item())
        self._resolve_pending(value_now)
        if self._steps % self.train_every == 0:
            self._train_step()
        return p

    def _on_fire(self, obs, value_now, delta):
        self._pending.append([self._last_feats, value_now, 0])

    def _resolve_pending(self, value_now):
        still = deque()
        for feats, v0, age in self._pending:
            age += 1
            if age >= self.outcome_horizon:
                label = 1.0 if value_now > v0 else 0.0
                self._buffer.append((feats, label))
            else:
                still.append([feats, v0, age])
        self._pending = still

    def _train_step(self):
        if len(self._buffer) < 8:
            return
        batch = self._buffer[-32:]
        self.opt.zero_grad()
        loss = 0.0
        for feats, label in batch:
            p = torch.clamp(self.qnet(feats), 1e-6, 1 - 1e-6)
            target = torch.tensor(label, dtype=p.dtype)
            loss = loss + nn.functional.binary_cross_entropy(p, target)
        loss = loss / len(batch)
        loss.backward()
        self.opt.step()
        self.train_losses.append(float(loss.item()))
