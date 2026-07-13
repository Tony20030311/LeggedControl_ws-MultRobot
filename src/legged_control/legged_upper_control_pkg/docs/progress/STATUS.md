# ADMM-CBF-DMPC — Staged Progress Overview

Single-page status of the ADMM-DMPC rebuild (replaces the legacy Pure-Pursuit L1–L4).
Detail per stage lives in the sibling `docs/progress/stage*.md` and the commit messages.

## Stage map (0→5, gated)

| Stage | Content | Status | Commit |
|-------|---------|--------|--------|
| 0 | `constants.py` + `test_constants.py` coefficient guard | ✅ done | `46c5f26` |
| 1 | Single-dog node QP + HOCBF, cold-start soft warm-up, 5-scenario gate | ✅ done | `46c5f26` |
| 2 | Two-dog node-edge splitting (edge QP + coordinator + RTI) | ✅ done | `3dc0876` |
| 3 | Three-dog complete graph + Laplacian formation + node obstacle CBF | ✅ done | `2cc8000` |
| 4 | ADMM ξ → OCS2 24-D centroidal target adapter + Gazebo wiring | ✅ done | see below |
| 5 | C++ migration / full hardware integration | ⬜ not started | — |

## Stage 4 wiring (rungs 0–2 + arenas)

| Item | Status | Commit |
|------|--------|--------|
| Offline core `motion_adapter.py` (ξ→OCS2 24-D) | ✅ | `348d237` |
| rung 0 — pure chain proven on Gazebo | ✅ | `4f8dfae` |
| rung 1 — single-dog ξ→OCS2 + Route B path-yaw (orbit fix) | ✅ | `571b599` + `43ea1cc` |
| rung 2 — three-dog leaderless V formation (empty world) | ✅ | `47cd20e` + `ba960e7` |
| rung 2 — basic obstacle arena (2 circular pegs) | ✅ | `d2a8f2a` |
| edge-handoff safety (dynamic safe-prefix truncation) | ✅ | `4a85a68` |
| 梅花樁 (plum-blossom) dense arena — A* curved refs | ✅ | `fc47dc5` |
| door arena (real map: `obstacle_world`/`five_dogs`) — door + walls + A* | ✅ | (this commit) |

## rung-2 stress-test issues — all resolved

| # | Issue | Resolution |
|---|-------|------------|
| ① | edge-handoff safety hole (head-on collision) | Fixed: dynamic safe-prefix truncation (`4a85a68`); offline + 7 Gazebo runs |
| ② | three dogs "slow" (~0.06 m/s) / v=0.15 OCS2 crash | **Not a defect — machine-RTF artifact.** Headless RTF≈1.0: v-sweep 0.08–0.20 all clean, realized ≈87–93% of commanded, no crash. Old machine's slow/crash was RTF collapse. |
| ③ | 梅花樁 dense demo | **Done** (`fc47dc5`). Straight refs deadlock (missing L3 planner, not edge safety); fix = reuse `AStarPlanner` + `build_reference`. 15 Gazebo runs (v=0.10–0.20) all reach, no fall, zero red flags. Real-time demo video. |

## Where we are

**Python ADMM-DMPC is fully wired onto Gazebo** — single-dog → three-dog → V formation
→ obstacle arena → dense peg threading — reliable, safe, with video. All three post-stress
issues closed.

## Not yet done

1. **Stage 5** — C++ migration / hardware. The Python is written C++-style (explicit index
   loops, `4*N` formulas) precisely to port later.
2. **Optional polish (non-defects):** at-goal ~0.25 m residual jiggle (add at-goal micro-freeze
   if wanted); cold-start "slow regime" root cause (first/cold run ~2× slower, machine-state).
3. **Real target map (`obstacle_world`/`five_dogs.launch`) — DONE (3 dogs).** 2.5 m door at
   x=6 + 5 cylinders + boundary walls. `ARENAS["door"]` (door = A* rect walls + circular
   posts; boundaries = node-CBF wall half-planes — the **first working wall CBF**). Offline
   `verify_door.py` ALL GREEN; Gazebo reached in 49 s, no fall, min_pair 1.03, min_node
   +0.045, clean logs; real-time video. 5-dog extension (formation-for-5 + 10 edges) is the
   remaining stretch on this arena.

## Verification discipline

Every stage/arena has an offline gate (`verify_*.py` / `test_*.py`) that must be green before
Gazebo; Gazebo validation is by real headless runs with full-log red-flag scans. Verification
runs in the Docker container (osqp 0.6.3, ROS sourced). Recording pipeline (xvfb + software GL
+ ffmpeg x11grab) is in `scratchpad/arena/run_plum_record.sh`.

## C6 — ADMM C++ migration (Stage 5), 2026-07-13

Core (C0–C5) + dependencies/shell (C6a–f) ported to C++, offline bit-identical
(`test_cpp_parity.py`) and Gazebo statistically equivalent to the Python shell.

- **C6e** Gazebo smoke (rospy shell + C++ core, `ADMM_IMPL=cpp`): surfaced + fixed the
  `ADMMCoordinator.obstacles` binding gap (publisher crashed at `__init__` — offline parity
  can't see the shell↔core boundary). empty/door/plum all green after.
- **C6f** roscpp node `ocs2_fleet_publisher_node.cpp` (mirror of the Python publisher,
  links admm_core). Synthetic-loop gate (`scratchpad/arena/smoke_fleet_node.py`) passes
  against BOTH the rospy shell and the node. Committed 7e9f05c.
- **C6g** statistical-equivalence campaign, roscpp node (`PUB=cpp`), 3 runs × 3 maps, all
  green (fell/sqp/crash/pyerr/nanholds/gaps = 0):

  | map   | node min_pair (×3)      | Python-shell baseline |
  |-------|-------------------------|-----------------------|
  | empty | 0.829 / 0.848 / 0.809   | ~0.78                 |
  | plum  | 0.768 / 0.799 / 0.785   | ~0.75 (all threaded)  |
  | door  | 0.595 / 0.597 / 0.599   | 0.591 / 0.595         |

  min_pair distributions match within sim noise → node ≡ shell behaviourally.

- **Also this session** (r_prim data + stress campaigns, `PUB` scripts are scratchpad-local):
  plum 3× round-trip thread (all 3 dogs through the pegs), door 11-leg choreography (direct
  side-exit works but is marginal — one run exited both sides, another stalled from the right;
  no fall/crash). ADMM per-iter residual (`~log_hist_csv`): r_prim (true-body vs edge-copy) ≈
  1e-14 when spread, spikes to ~0.5 m initial when the inter-agent CBF binds but converges to
  ≤2 cm by iter 20; obstacle-CBF clearance grazed −0.01..−0.05 m at GT (tracking overshoot of
  the r_eff=0.60 barrier) with ~0.25 m physical margin to the obstacle surface, no collision.

- **C6h — DONE (migration complete):** deleted the Python ADMM core (constants, rti_linearizer,
  node/edge_subproblem, admm_coordinator, motion_adapter, reference), the rospy shell
  (`scripts/ocs2_fleet_publisher.py`), and the parity/perf tools (`test_cpp_parity`,
  `test_node_qp/edge_qp/rti`, `bench_timing`). `admm_impl.py` is now unconditional cpp (the
  `ADMM_IMPL` switch is gone); `admm_demo.launch` runs `type=ocs2_fleet_publisher_node`. C++ is
  the sole implementation. **Verified before deletion** (all in the SIL container, C++ shim):
  34/34 retained unit tests + 8/8 verify gates (stage1–4/arena/door/plum/edge_handoff) green,
  grep = 0 live refs to the deleted modules, `catkin build legged_upper_control` green, and a
  door confidence run via the NEW launch path (`admm_demo.launch` → node) — full through-door
  round trip, fell/sqp/crash/nanholds/gaps = 0, min_pair 0.736 ≥ 0.60 over 2228 ticks, fleet
  settled at the commanded centroid (0.09 m). Re-verified green AFTER deletion. The retained
  tests/verify still route through the `admm_impl` shim, now exercising the C++ port.
