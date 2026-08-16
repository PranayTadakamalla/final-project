# Tunneling Through Deception: A Quantum-Inspired Exploration Operator for Agentic AI

**Student:** Tadakamalla Sai Pranay (2311CS020646) · **Batch:** E-BATCH-30
**Guide:** Manasa Chandupatla · Target: Scopus-indexed conference paper, Jan 2027

## What this is

A Tunneling Exploration Operator (TEO) for RL agents in deceptive reward
landscapes, tested in two parallel forms — a classical exponential-decay
formula and a simulated 4-qubit variational quantum circuit — against
epsilon-greedy and other standard exploration baselines.

See `PROJECT_CONTEXT.md` for full background, established findings, and the
two mistakes already made and fixed during development (hardcoded-direction
confound, optional stopping). Read it before proposing changes.

## Setup

```bash
pip install -r requirements.txt
```

## Verify the environment before trusting anything

```bash
python3 tests/test_env2d.py
```

This must show `[PASS] FixedDirection tripwire: 0.00 goal rate` — if that
number is meaningfully above 0, the environment is confounded and no result
built on it is valid.

## Run the confirmatory experiment

Per `PREREGISTRATION.md`: n=40 seeds/arm, no interim analysis, no early
stopping, no further tuning of any arm.

```bash
python3 experiments/confirm.py --arm teo_classical --budget 40
python3 experiments/confirm.py --arm teo_quantum   --budget 40
python3 experiments/confirm.py --arm baseline      --budget 40
```

Resumable — re-invoke to continue from wherever it left off; each arm reports
`COMPLETE` at 40/40. Do not compute or inspect any p-value before all three
arms are complete.

## Codebase map

```
envs/deep_deception.py     current environment (15x15, 3-cell barrier ring)
agents/dqn.py               DQN with value_estimate() hook for TEO
agents/strategies.py        8 strategies incl. FixedDirection tripwire, TEO v2
agents/teo_v3.py            TEO v3: stagnation gating, frontier targeting,
                             push-length scaling, anneal + backoff
agents/quantum_circuit.py   PennyLane VQC (lightning.qubit, adjoint gradients)
experiments/run_hard.py     training harness
experiments/confirm.py      pre-registered confirmatory runner (resumable)
tests/test_env2d.py         correctness tests incl. flood-fill + tripwire
PREREGISTRATION.md          locked experimental design
PROJECT_CONTEXT.md          full project history, findings, working agreements
```

## Working agreements

- No fabricated, placeholder, or interpolated numbers, ever.
- No claim of quantum hardware advantage — the circuit is classically simulated.
- Every environment change gets correctness tests before any result built on
  it counts.
- Label exploratory vs. confirmatory results explicitly, always.
- A negative confirmatory result is reported as the headline finding, not
  buried in limitations.
