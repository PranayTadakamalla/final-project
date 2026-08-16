"""Runs the full matrix on DeepDeceptionGrid (thick barrier).
Baselines get the SAME tuning budget as TEO -- see experiments/sweep_baseline.py."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.deep_deception import DeepDeceptionGrid
from agents.dqn import DQNAgent
from agents.strategies import REGISTRY as BASE_REG
from agents.teo_v3 import TEOClassicalV3, TEOQuantumV3

REG = dict(BASE_REG)
REG["teo_classical_v3"] = TEOClassicalV3
REG["teo_quantum_v3"] = TEOQuantumV3


def run_episode(env, agent, strat, train=True):
    obs, _ = env.reset(); total = 0.0; steps = 0; reached = False
    term = trunc = False
    while not (term or trunc):
        g = agent.greedy_action(obs)
        a = strat.select_action(agent, obs, g)
        nobs, r, term, trunc, info = env.step(a)
        if train:
            agent.store(obs, a, r, nobs, float(term))
            agent.update(batch_size=64)
        obs = nobs
        total += r
        steps += 1
        reached = reached or info["at_goal"]
    strat.end_episode()
    return total, steps, reached


def train_run(name, seed, n_ep, kwargs=None, size=15, max_steps=300, thickness=3):
    env = DeepDeceptionGrid(size=size, max_steps=max_steps,
                            barrier_thickness=thickness, seed=seed)
    agent = DQNAgent(obs_dim=2, n_actions=4, seed=seed)
    strat = REG[name](seed=seed, **(kwargs or {}))
    rewards, goals, first = [], [], None
    for ep in range(n_ep):
        r, s, g = run_episode(env, agent, strat)
        rewards.append(r)
        goals.append(bool(g))
        if g and first is None:
            first = ep
    out = {"strategy": name, "seed": seed, "goal_corner": env.goal_corner,
           "optimal_return": env.optimal_return(), "episode_rewards": rewards,
           "goal_flags": goals, "first_goal_episode": first,
           "final50_avg_reward": float(np.mean(rewards[-50:])),
           "goal_rate_last100": float(np.mean(goals[-100:])),
           "env": "deep_deception", "config": kwargs or {}}
    for attr in ("n_fired", "n_stagnant"):
        if hasattr(strat, attr):
            out[attr] = int(getattr(strat, attr))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--strategies", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    res = []
    t0 = time.time()
    for s in a.strategies.split(","):
        for sd in range(a.seed_start, a.seed_start + a.seeds):
            t1 = time.time()
            r = train_run(s, sd, a.episodes)
            dt = time.time() - t1
            print(f"{s:<18} seed={sd:<3} corner={r['goal_corner']} "
                  f"goal={r['goal_rate_last100']:.2f} "
                  f"rew={r['final50_avg_reward']:7.2f} ({dt:.1f}s)")
            res.append(r)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump({"results": res}, open(a.out, "w"))
    print(f"Done in {time.time()-t0:.1f}s -> {a.out}")
