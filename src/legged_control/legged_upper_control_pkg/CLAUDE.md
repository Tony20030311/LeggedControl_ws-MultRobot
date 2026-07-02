# CLAUDE.md — legged_upper_control_pkg (ADMM-CBF-DMPC layer)

This file governs the **ADMM-CBF-DMPC upper-control implementation** in this package.
For workspace-wide rules (C++ read-only, build/run, namespaces), see the repo-root CLAUDE.md.

> **This package is under active migration.** The L1–L4 stack described in the root
> CLAUDE.md (Pure Pursuit + single-integrator + velocity-level CBF filter) is being
> **replaced** by a trajectory-level ADMM-DMPC system. Where the root CLAUDE.md and this
> file disagree about the control architecture, **this file wins** for anything under
> this package.

---

## Source of truth (read before implementing)

The design lives in `docs/`. When documents conflict, **higher wins**:

1. `docs/spec_updates_v2.md` — spec patch, newest, **binding**
2. `docs/文件二_架構藍圖.md` — main architecture blueprint (C1–C6)
3. `docs/文件一_論文完整數學推導.md` — paper math, source of truth for derivations
4. `docs/opensource_reference.md` — open-source navigation (**reference, not spec**)

Open-source under `references/` is **reference implementation, never copy-paste target**.
`opensource_reference.md` states, per repo, what to borrow and what to avoid. Read it
before touching any stage.

## Build order (staged, gated)

Follow the stage map in `opensource_reference.md` (stages 0→5). **A stage must pass its
gate before the next begins.** The node-edge splitting (stage 2) is original work with no
open-source template — it must be built on the verified foundation of stages 0–1, not
ahead of it.

Each stage's gate is expressed as an executable check (assertions in a `verify_stage*.py`
or `test_*.py`), not prose. Implement the gate's assertions first, then the code, then run
to green. Behavioral gates that assertions can't fully capture (e.g. the *shape* of an
`r_prim` convergence curve) additionally produce a plot for human inspection.

## Architecture (target: ADMM-DMPC, not the legacy stack)

The system this package builds — distinct from the legacy Pure-Pursuit stack:

- **Model:** world-frame double integrator. State `X = [px, py, vx, vy]` (position first),
  input `u = [ax, ay]`. `X(t+1) = Ad·X(t) + Bd·u(t)`.
- **Formulation:** trajectory-level multiple-shooting. Decision vector
  `ξ^i ∈ R^{6N}`, states first (`4N`), accelerations after (`2N`).
- **Graph:** complete graph, 3 node QPs + 3 edge QPs.
- **Safety:** Xiong discrete second-order HOCBF (relative degree 2), inter-agent CBF on edges.
- **Formation:** normalized-Laplacian cost, linearized to a node-local gradient term.
- **Coordination:** ADMM node→edge→dual, 15–20 iterations, no wait-for-convergence.
- **Rate:** 10 Hz high level (was 20 Hz). Sends state trajectory to OCS2 via a custom adapter.

The four subproblem/coordinator classes (RTILinearizer, NodeSubproblem, EdgeSubproblem,
ADMMCoordinator) are being written from scratch. Python first (C++-style explicit index
loops), C++ migration later.

## Long-standing implementation constraints

These hold for the lifetime of this package (not task-specific, not expiring):

- **Solver:** OSQP with direct binding — **not cvxpy**. Pre-assemble the sparsity pattern;
  each ADMM iteration updates values only (`osqp.update(q=, l=, u=)`), never rebuilds.
  Edge QP always warm-starts.
- **Decision-variable layout is fixed:** `ξ = [x_1..x_N (4N), a_0..a_{N-1} (2N)]`.
  All matrix indexing uses `4*N`-based formulas — **never hardcode integer offsets**
  (N changes; a hardcoded `40` silently breaks when N goes 10→20).
- **z is stored per-edge, not per-node:** `z[edge][endpoint]`. A robot on multiple edges has
  an independent copy per edge. Node update's `Σ_{j∈N(i)}` is wrong if copies are merged by node.
- **Formation goes in node cost, never in the constraint set.** It is a soft objective;
  as a hard constraint it fights the CBF and goes infeasible. Formation is all-to-all
  (normalized Laplacian denominator couples the third robot) and **cannot be split into an
  edge** — linearize it to a frozen node gradient instead.
- **Inter-agent CBF goes on edges** (keeps both `a^i, a^j` coupled); obstacle/wall CBF is
  node-local.
- **Linearize once per control cycle, outside the ADMM loop.** Freeze CBF coefficients,
  formation gradient, Ad/Bd, and b_k; the 15–20 iterations share the frozen set.
- **yaw is a bypass, not an ADMM variable** — velocity-direction `atan2` with low-speed
  freeze and EMA, differenced through `wrap_to_pi`. Never let EMA cross ±π unwrapped.

## Coefficient discipline

Coefficients that are wrong don't crash — they make behavior subtly incorrect and are
painful to debug after the fact. They are guarded by `test_constants.py`; run it before
trusting any matrix assembly. Do not reason coefficients from memory or from a single
document's inline box — the guard test is authoritative.

## What NOT to carry over from the legacy stack

The legacy Pure-Pursuit stack carries many reactive patches (stuck recovery, slot freeze,
door mode, formation guards, target projection). These were built for a reactive velocity
QP and are **redundant/harmful under trajectory-level ADMM** — they fight the trajectory
optimizer. See the "reactive patch removal list" in `spec_updates_v2.md`. Do not port them
into the ADMM implementation.

## Working mode

This is original research code with no open-source template for the core (stage 2). Derive
the math together and confirm the design before writing the node-edge splitting — a wrong
formulation here propagates silently through every later stage. Read-only analysis of
`references/` and existing code is fine without asking.