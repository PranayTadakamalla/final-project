"""Confirmatory run per PREREGISTRATION.md. n=40 seeds/arm, no early stopping.
Resumable: re-invoking continues from whatever seeds are already logged.

Usage:
    python3 experiments/confirm.py --arm teo_classical --budget 40
    python3 experiments/confirm.py --arm teo_quantum   --budget 40
    python3 experiments/confirm.py --arm baseline      --budget 40

Re-invoke repeatedly (each call is capped at --budget seeds so a single
invocation doesn't run indefinitely); it skips seeds already present in the
results file. Reports COMPLETE when an arm hits 40/40 and exits without
re-running anything.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_hard import train_run

BASE_CFG = {"eps_decay_steps": 5000, "eps_end": 0.10}
N_TARGET = 40

def load(path):
    if os.path.exists(path):
        return json.load(open(path))["results"]
    return []

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "teo_classical", "teo_quantum"])
    ap.add_argument("--budget", type=int, default=6, help="max seeds this invocation")
    a = ap.parse_args()

    spec = {
        "baseline": ("epsilon_greedy", BASE_CFG, "epsilon_greedy_tuned"),
        "teo_classical": ("teo_classical_v3", {}, "teo_classical_v3"),
        "teo_quantum": ("teo_quantum_v3", {}, "teo_quantum_v3"),
    }[a.arm]
    strat, cfg, label = spec
    path = f"results/confirm_{a.arm}.json"

    res = load(path)
    done = {r["seed"] for r in res}
    todo = [s for s in range(N_TARGET) if s not in done][:a.budget]
    if not todo:
        print(f"{a.arm}: COMPLETE ({len(done)}/{N_TARGET})")
        sys.exit(0)

    for sd in todo:
        r = train_run(strat, sd, 300, kwargs=cfg)
        r["strategy"] = label
        res.append(r)
        json.dump({"results": res, "prereg": True}, open(path, "w"))
        print(f"{label} seed={sd} corner={r['goal_corner']} "
              f"goal={r['goal_rate_last100']:.2f} rew={r['final50_avg_reward']:7.2f}")

    n_done = len(res)
    status = "COMPLETE" if n_done >= N_TARGET else f"{n_done}/{N_TARGET} (re-invoke to continue)"
    print(f"{a.arm}: {status}")
