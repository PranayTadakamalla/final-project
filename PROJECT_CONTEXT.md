# PROJECT CONTEXT — Tunneling Through Deception

**Add this file to Claude Project knowledge.** It is the single source of
truth. Any future chat or agent should read this before proposing anything,
so it does not re-derive decisions already made or contradict findings
already established.

---

## Identity (fixed, do not drift)

| Field | Value |
|---|---|
| Title | Tunneling Through Deception: A Quantum-Inspired Exploration Operator for Agentic AI |
| Student | Tadakamalla Sai Pranay — Roll No. 2311CS020646 |
| Batch | E-BATCH-30 (single-student project) |
| Guide | Manasa Chandupatla |
| Coordinator | Dr. Anjaiah (signed the abstract) |
| Target | Scopus-indexed conference paper before January 2027 |

## What the project is

RL agents in deceptive reward landscapes get trapped in local optima. Standard
exploration (epsilon-greedy, entropy bonuses) escapes only by undirected
randomness. This project builds the **Tunneling Exploration Operator (TEO)**: a
pluggable exploration module that estimates a barrier height from the agent's
own value function and computes a directed escape probability — in two forms:

- **TEO-Classical:** P_tunnel = exp(-alpha * Delta), physically faithful to the
  WKB transmission coefficient.
- **TEO-Quantum:** the same probability produced by a 4-qubit variational
  quantum circuit (PennyLane, `lightning.qubit`). **Simulated classically. No
  hardware. No speedup claim.**

**The research question:** does the quantum-circuit-computed probability
produce measurably different exploration behaviour than the classical formula?
A null result is an acceptable, publishable answer and has been the answer
so far.

## Findings so far — READ BEFORE CITING ANYTHING

### Established (exploratory, n≈10-22)
- **Quantum vs classical: no significant difference** (p = 0.7539, n = 10).
- **TEO's variance is consistently ~half the baseline's**, the most robust
  observed property.
- **The environment is not confounded.** FixedDirection control scores 0.000
  (verified in `tests/test_env2d.py`, must be re-run after any code change).

### Exploratory only — DO NOT cite as confirmatory
- p = 0.0105 / 0.0034 (n=10): obtained against an **under-tuned baseline**. Void.
- p = 0.00451 / 0.00118 (n≈20): obtained via **optional stopping**. Void.

### Confirmatory status
A pre-registered confirmatory experiment (n=40/arm, no interim analysis,
mandatory goal-corner stratification) is defined in `PREREGISTRATION.md`.

**IMPORTANT:** A previous session reached 84/120 completed runs (teo_classical
40/40, teo_quantum 27/40, baseline 17/40), but that raw run data lived only in
that session's sandbox and did not persist. The codebase was reconstructed and
re-verified from the conversation transcript (see "Reconstruction" below) —
the confirmatory experiment restarts from 0/120 on this reconstructed code.
The pre-registered design itself is unchanged.

## Two mistakes already made — do not repeat them

1. **The hardcoded-direction confound.** v1 TEO "won" only because its push
   action pointed at the goal in a 1D corridor; a trivial always-right control
   beat it. Fixed by randomising goal corner per seed and making TEO choose
   direction from its own visitation counts. *The FixedDirection control is a
   permanent tripwire — if it ever scores well, results are void.*

2. **Optional stopping.** A null result at n=10 was followed by extending the
   sample until significance appeared. This is why the confirmatory run exists.

**Pattern to notice:** both wrong results looked *good*. Both were caught by
stopping to check, not by running more. Treat a surprisingly strong result as
a prompt to look for a confound, not a reason to celebrate.

## Environment (current)

`DeepDeceptionGrid`: 15×15, 4 actions, **3-cell-thick barrier ring** enclosing
the start basin, goal corner randomised per seed, one-time decoy (+2.0), goal
(+10.0), valley penalty −0.4/cell, step cost −0.01, 300 max steps.

The barrier was thickened from 1 cell because epsilon-greedy scored 0.987 on
the thin version — an environment every method solves discriminates nothing.
The hardening is mechanism-agnostic. **State this openly in the paper.**

## Reconstruction note (August 2026)

The codebase was rebuilt file-by-file from the source conversation transcript
after the working sandbox from a prior session was lost (per-session
filesystem, not pushed to persistent storage at the time). Every file was
re-verified against tests after reconstruction:
- Environment: flood-fill closure test passes, goal corners vary across seeds.
- DQN + all 8 strategies: run end-to-end without error.
- TEO v3 anti-thrash fix: re-tested on the historically problematic seed
  (seed 1, which originally fired 716 times and collapsed to 0.39 goal rate);
  confirmed convergence to 1.00 goal rate over 300 episodes, matching the
  original fix.
- FixedDirection tripwire: 0.00 goal rate over 20 seeds, confirming the
  environment is not confounded.

This repo is now the durable source of truth going forward — pushed to
GitHub specifically so this loss cannot recur.

## Codebase map

```
envs/deep_deception.py     current hard env (3-cell barrier)
agents/dqn.py               DQN with value_estimate() hook for TEO
agents/strategies.py        epsilon-greedy, count-based, RND, novelty, random,
                             FixedDirection control, TEO v2
agents/teo_v3.py            TEO v3: stagnation gating, frontier targeting,
                             push-length scaling, anneal + backoff
agents/quantum_circuit.py   PennyLane VQC (lightning.qubit, adjoint gradients)
experiments/run_hard.py     training harness for the hard env
experiments/confirm.py      pre-registered confirmatory runner (resumable)
tests/test_env2d.py         correctness tests incl. closed-ring flood fill
                             and FixedDirection tripwire
```

## Working agreements

- No fabricated, placeholder, or interpolated numbers, ever.
- No claim of quantum advantage.
- Every environment change gets correctness tests before any result built on
  it counts.
- Label exploratory vs confirmatory explicitly, always.
- If a result is negative, it goes in the abstract, not buried in limitations.

## Remaining work

1. Finish the confirmatory run (120 runs, from 0).
2. Failure analysis — what distinguishes high-goal-rate seeds from failures?
3. Tune and run the remaining baselines (count-based, novelty, RND) fairly.
4. Second environment (structurally different) for any generality claim.
5. Presentation/demo layer (visualizations, dashboard) — built from real
   results only, kept strictly separate from anything paper-facing.
6. Paper draft — built from `results/` only, after the confirmatory run.

## Likely framing for the paper

The most defensible headline is **reliability, not peak performance**: TEO
roughly halves across-seed variance. Whether it also raises the mean is what
the confirmatory run decides. Do not write the results section before that
run finishes.
