"""
Pluggable exploration strategies, all sharing one interface:

    strategy.select_action(agent, obs, greedy_action) -> action (int)
    strategy.end_episode()

This is what makes TEO a genuine plug-in: swapping strategies changes nothing
about DQNAgent itself. All strategies may read ONLY: agent.q_values(obs),
agent.value_estimate(obs), and their own internal state (visitation counts,
archives). None may read env internals, goal position, or true barrier cost.

Here, TEO chooses its push direction from its OWN visitation memory (steer
toward the least-visited neighbour). `FixedDirection` is retained permanently
in the matrix as a control that must NOT win.
"""
import math
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 4
# action index -> (dx, dy), must match envs.deep_deception.ACTIONS
_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


def _key(obs, bins=11):
    """Discretise a normalised observation into a visitation key."""
    return tuple(int(round(float(v) * (bins - 1))) for v in obs)


# ======================================================================
# Controls / baselines
# ======================================================================
class RandomPolicy:
    name = "random"

    def __init__(self, seed=0, **kw):
        self.rng = random.Random(seed)

    def select_action(self, agent, obs, greedy_action):
        return self.rng.randrange(N_ACTIONS)

    def end_episode(self):
        pass


class FixedDirection:
    """CONTROL / TRIPWIRE: ignores everything and always takes one action.
    Must NOT be competitive -- if it scores well, the environment is confounded
    and every result built on it is void. Kept permanently in the experiment
    matrix for exactly this reason."""
    name = "fixed_direction"

    def __init__(self, action=3, seed=0, **kw):
        self.action = action

    def select_action(self, agent, obs, greedy_action):
        return self.action

    def end_episode(self):
        pass


class EpsilonGreedy:
    name = "epsilon_greedy"

    def __init__(self, eps_start=1.0, eps_end=0.05, eps_decay_steps=20000, seed=0, **kw):
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.t = 0
        self.rng = random.Random(seed)

    def epsilon(self):
        frac = min(1.0, self.t / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, agent, obs, greedy_action):
        self.t += 1
        if self.rng.random() < self.epsilon():
            return self.rng.randrange(N_ACTIONS)
        return greedy_action

    def end_episode(self):
        pass


class CountBased:
    """Count-based intrinsic bonus: pick the action whose resulting state has
    been least visited, with probability proportional to an exploration weight."""
    name = "count_based"

    def __init__(self, beta=0.5, seed=0, **kw):
        self.counts = defaultdict(int)
        self.beta = beta
        self.rng = random.Random(seed)

    def select_action(self, agent, obs, greedy_action):
        k = _key(obs)
        self.counts[k] += 1
        q = agent.q_values(obs)
        bonus = np.array([
            self.beta / math.sqrt(1 + self.counts[_key(np.clip(
                np.array(obs) + np.array(_DELTAS[a]) / 10.0, 0, 1))])
            for a in range(N_ACTIONS)
        ])
        return int(np.argmax(q + bonus))

    def end_episode(self):
        pass


class RND:
    """Random Network Distillation (Burda et al. 2018).
    A fixed random target network and a trained predictor; prediction error is
    the novelty signal. Action chosen by Q + scaled intrinsic bonus."""
    name = "rnd"

    def __init__(self, obs_dim=2, hidden=64, out=16, lr=1e-3, beta=1.0, seed=0, **kw):
        torch.manual_seed(seed + 999)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.beta = beta
        self.rng = random.Random(seed)

    def _novelty(self, obs_batch):
        s = torch.tensor(np.array(obs_batch), dtype=torch.float32)
        with torch.no_grad():
            t = self.target(s)
            p = self.predictor(s)
        return ((t - p) ** 2).mean(dim=1).numpy()

    def _train(self, obs):
        s = torch.tensor(np.array([obs]), dtype=torch.float32)
        with torch.no_grad():
            t = self.target(s)
        p = self.predictor(s)
        loss = ((t - p) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def select_action(self, agent, obs, greedy_action):
        self._train(obs)
        cand = [np.clip(np.array(obs) + np.array(_DELTAS[a]) / 10.0, 0, 1)
                for a in range(N_ACTIONS)]
        bonus = self._novelty(cand)
        q = agent.q_values(obs)
        return int(np.argmax(q + self.beta * bonus))

    def end_episode(self):
        pass


class NoveltySearch:
    """Archive-based novelty: bonus = mean distance to k nearest archive entries."""
    name = "novelty_search"

    def __init__(self, k=8, beta=1.0, archive_max=2000, seed=0, **kw):
        self.archive = deque(maxlen=archive_max)
        self.k = k
        self.beta = beta
        self.rng = random.Random(seed)

    def _novelty(self, point):
        if len(self.archive) < self.k:
            return 1.0
        arr = np.array(self.archive)
        d = np.linalg.norm(arr - np.array(point), axis=1)
        return float(np.sort(d)[:self.k].mean())

    def select_action(self, agent, obs, greedy_action):
        self.archive.append(np.array(obs, dtype=np.float32))
        q = agent.q_values(obs)
        bonus = np.array([
            self._novelty(np.clip(np.array(obs) + np.array(_DELTAS[a]) / 10.0, 0, 1))
            for a in range(N_ACTIONS)
        ])
        return int(np.argmax(q + self.beta * bonus))

    def end_episode(self):
        pass


# ======================================================================
# TEO -- shared machinery (this is the v2 base; superseded by teo_v3.py for
# the actual confirmatory experiment, kept here for provenance / ablation)
# ======================================================================
class _TEOBase:
    """Common barrier estimation + novelty-directed push.

    Barrier height Delta is estimated ONLY from the agent's own value function:
    the drop in max_a Q(s,a) over a trailing window. This is a proxy, not ground
    truth -- documented as a limitation, never replaced by env internals.

    When a tunnel event fires the operator commits to a short directed push
    toward its LEAST-VISITED neighbouring region, chosen from its own visitation
    counts. It has no access to the goal location.
    """

    def __init__(self, history_len=8, push_len=6, seed=0, delta_floor=0.02, **kw):
        self.history_len = history_len
        self.push_len = push_len
        self.delta_floor = delta_floor
        self.rng = random.Random(seed)
        self.value_history = deque(maxlen=history_len + 1)
        self.counts = defaultdict(int)
        self._push_remaining = 0
        self._push_action = None
        self.log = []          # (delta, p_tunnel, fired)
        self.n_fired = 0

    def _estimate_delta(self, value_now):
        if len(self.value_history) <= self.history_len:
            return 0.0
        past = self.value_history[0]
        return max(0.0, past - value_now)

    def _least_visited_action(self, obs):
        scores = []
        for a in range(N_ACTIONS):
            nxt = np.clip(np.array(obs) + np.array(_DELTAS[a]) / 10.0, 0, 1)
            scores.append(self.counts[_key(nxt)])
        m = min(scores)
        best = [a for a, s in enumerate(scores) if s == m]
        return self.rng.choice(best)

    def _p_tunnel(self, delta, obs, value_now):
        raise NotImplementedError

    def select_action(self, agent, obs, greedy_action):
        self.counts[_key(obs)] += 1
        value_now = agent.value_estimate(obs)
        self.value_history.append(value_now)

        if self._push_remaining > 0:
            self._push_remaining -= 1
            self.log.append((None, None, False))
            return self._push_action

        delta = self._estimate_delta(value_now)
        p = self._p_tunnel(delta, obs, value_now)

        fired = False
        if delta > self.delta_floor and self.rng.random() < p:
            fired = True
            self.n_fired += 1
            self._push_action = self._least_visited_action(obs)
            self._push_remaining = self.push_len - 1
            self._on_fire(obs, value_now, delta)

        self.log.append((delta, p, fired))
        return self._push_action if fired else greedy_action

    def _on_fire(self, obs, value_now, delta):
        pass

    def end_episode(self):
        self.value_history.clear()
        self._push_remaining = 0


class TEOClassical(_TEOBase):
    """P_tunnel = exp(-alpha * Delta): physically faithful -- probability DECAYS
    as the barrier grows, mirroring the WKB transmission coefficient. Do not
    invert this relationship; it must match the physics."""
    name = "teo_classical"

    def __init__(self, alpha=1.5, **kw):
        super().__init__(**kw)
        self.alpha = alpha

    def _p_tunnel(self, delta, obs, value_now):
        return math.exp(-self.alpha * delta)


class TEOQuantum(_TEOBase):
    """P_tunnel produced by a 4-qubit variational quantum circuit (PennyLane).
    SIMULATED CLASSICALLY -- no hardware, no speedup claim. The research
    question is whether an entangled, trainable probability generator behaves
    differently from the closed-form exponential."""
    name = "teo_quantum"

    def __init__(self, lr=0.05, train_every=64, outcome_horizon=8, n_layers=2, **kw):
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
        f_delta = min(delta / 1.0, 1.0)
        f_visits = min(self.counts[_key(obs)] / 50.0, 1.0)
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


REGISTRY = {
    "random": RandomPolicy,
    "fixed_direction": FixedDirection,
    "epsilon_greedy": EpsilonGreedy,
    "count_based": CountBased,
    "rnd": RND,
    "novelty_search": NoveltySearch,
    "teo_classical": TEOClassical,
    "teo_quantum": TEOQuantum,
}
