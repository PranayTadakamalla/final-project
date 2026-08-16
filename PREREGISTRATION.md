# Pre-Registration — Confirmatory Experiment

This document locks the design of the confirmatory experiment BEFORE its
results exist. It exists because two earlier results in this project were
wrong and looked good: an under-tuned-baseline comparison (p=0.0105/0.0034,
n=10) and an optional-stopping comparison (p=0.00451/0.00118, n≈20). Both are
exploratory only and must never be cited as confirmatory findings.

## Hypotheses

- **H1 (primary):** TEO-Quantum-v3 achieves a higher goal-reach rate than the
  tuned epsilon-greedy baseline on DeepDeceptionGrid.
- **H2 (primary):** TEO-Classical-v3 achieves a higher goal-reach rate than the
  tuned epsilon-greedy baseline on DeepDeceptionGrid.
- **H3 (secondary, expected null):** TEO-Quantum and TEO-Classical do not
  differ. Prior evidence (p = 0.7539, n = 10) suggests no difference; this run
  tests it with more power.
- **H4 (variance):** TEO variants show lower across-seed standard deviation in
  goal-reach rate than the baseline.

## Fixed design — no deviations permitted

| Parameter | Committed value |
|---|---|
| Sample size | **n = 40 seeds per arm** (seeds 0–39) |
| Stopping rule | Run all 40 seeds per arm. **No interim analysis. No early stop, whatever the intermediate p-values look like.** |
| Environment | DeepDeceptionGrid, size 15, barrier_thickness 3, max_steps 300 |
| Episodes per run | 300 |
| Agent | DQN (obs_dim 2, n_actions 4, lr 1e-3, gamma 0.99) |
| Baseline config | epsilon_greedy with eps_decay_steps 5000, eps_end 0.10 (selected by a sweep run BEFORE this registration) |
| TEO config | v3 defaults as committed in `agents/teo_v3.py` at time of writing — no further tuning |
| Primary metric | goal-reach rate over the last 100 episodes |
| Secondary metric | mean reward over the final 50 episodes |
| Test | Welch's two-sided t-test |
| Alpha | 0.05, **Holm-Bonferroni corrected across H1 and H2** |

## Mandatory stratified check

A batch effect was observed in exploratory data: seeds 10+ favoured TEO
(0.837 → 0.963) while simultaneously disfavouring the baseline (0.585 → 0.496).
This may be chance, or may reflect goal-corner composition differing between
batches.

Therefore the confirmatory analysis **must** report goal-reach rate stratified
by `goal_corner` (4 levels) for every arm, and state whether the main effect
holds within each stratum. If the effect is driven by a single corner, that
must be reported as the finding, not averaged away.

## What will be reported regardless of outcome

- All 40 seeds per arm, including failures. No seed exclusions for any reason.
- The stratified table, whatever it shows.
- If H1/H2 fail: that will be reported as the result. A negative confirmatory
  outcome supersedes the positive exploratory one and will be stated plainly in
  the abstract, not buried in limitations.

## Provenance note

Exploratory results (n ≈ 10–22) remain in `FINAL_RESULTS.md` and are labelled
exploratory throughout. They generated these hypotheses; they do not test them.

## Status at time of last known checkpoint (previous session)

| Arm | Completed |
|---|---|
| teo_classical | 40/40 |
| teo_quantum | 27/40 |
| baseline | 17/40 |

**Total: 84/120.** This checkpoint's raw run data did not persist between
sessions (see PROJECT_CONTEXT.md for details) — the confirmatory run restarts
from 0/120 on the reconstructed codebase. The design above is unchanged.
