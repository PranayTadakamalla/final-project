"""Standard DQN: replay buffer + target network. Kept deliberately small and
inspectable. Exposes value_estimate() and q_values() so exploration strategies
can read the agent's own value signal (they may read NOTHING else -- no ground
truth barrier cost, no goal position, no privileged environment state)."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim=2, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim=2, n_actions=4, lr=1e-3, gamma=0.99,
                 target_update_every=250, device="cpu", seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = device
        self.q = QNet(obs_dim, n_actions).to(device)
        self.q_target = QNet(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.n_actions = n_actions
        self.target_update_every = target_update_every
        self._updates = 0

    def q_values(self, obs):
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q(s).squeeze(0).cpu().numpy()

    def greedy_action(self, obs) -> int:
        return int(np.argmax(self.q_values(obs)))

    def value_estimate(self, obs) -> float:
        """max_a Q(s,a) -- this is the ONLY signal exploration strategies (TEO
        included) are permitted to use for barrier estimation. Never expose
        env.true_barrier_cost() or env.goal to any strategy."""
        return float(np.max(self.q_values(obs)))

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        s2 = torch.tensor(s2, device=self.device)
        d = torch.tensor(d, device=self.device)

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1)[0]
            target = r + self.gamma * (1.0 - d) * q_next
        loss = nn.functional.mse_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._updates += 1
        if self._updates % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())
