# Formation Adaptation for Tight-Pocket Goals — Design Spec

**Date:** 2026-07-07
**Status:** Design agreed (approach 2a), ready to implement in a fresh session.
**Author:** Tony + Claude (brainstormed)

---

## Problem

The ADMM 3-dog stack threads the door arena (`obstacle_world` / `five_dogs.launch`)
to **open goal pockets** reliably (goal set "A", robust to v=0.25). But goals in
**tight/cluttered pockets** (sets B center-low, C high, D right-low) **fail**: a dog
topples, with 3× `[SqpSolver] Failed to solve QP`.

### Root cause (verified, not hypothesised)

Traced via the door reliability campaign (`scratchpad/arena/run_door_campaign.sh`) +
offline probe (`scratchpad/measure_door.py`) + track-CSV forensics:

1. It is **not** RTF collapse — all runs RTF ~1.0 (0.96–0.998).
2. It is **not** an unsafe plan — goal B threads CLEAN **offline** (DI plant, min
   inter-agent 0.674 m > D_MIN 0.60).
3. The fall is a **NaN**: at the squeeze, the **node QP goes infeasible** and OSQP
   returns a non-finite `xi`. A NaN is not a Python exception, so it slipped past the
   publisher's `try/except`; the NaN target reached OCS2 → SqpSolver crash → fall.
   (`track.csv` showed `tgt1x/tgt1y/cmd` all `nan` at the fall instant.)
4. **Why the node QP is infeasible:** a dog approaches the pillar (e.g. ~0.65 m from
   the (8,0.5) cylinder, ≈ node-CBF r_eff 0.60) at a speed/angle where the hard k=0
   obstacle-CBF row has no feasible acceleration within input/state bounds. **The
   formation cost is what drives the dog into that corner** — it pulls the dog toward
   its V slot, which sits behind/beside the pillar.

### Already fixed (committed `5f1977c`)

A **finite-guard** in `ocs2_fleet_publisher.py`: after `coord.step`, if any dog's `xi`
is non-finite, HOLD the last good target this cycle (never publish NaN to OCS2). This
converts the **dangerous FALL into a safe STALL** — verified: goal B before fell=1/sqp=3,
after fell=0/sqp=0 (guard caught 17–191 NaN cycles); goal A unaffected (holds=0, reaches).

**Remaining gap (this spec):** tight goals now safely **stall/jam** (dogs freeze,
drifting to min_pair ~0.39–0.49 — nearly touching, no physical collision, no fall) but
**do not reach**. We want them to reach (or settle safely spread out) instead of jamming.

---

## Chosen approach: 2a — obstacle-adaptive formation weight

**Idea:** the formation is a *soft cost* (weight `w_form`, currently 10, entering the
node QP as a linear `+w_form·g` term — see `admm-staged-progress` memory "formation =
bare +w_form·g node linear term"). **Fade a dog's formation pull toward 0 as it nears an
obstacle; restore it when clear.** With the formation no longer shoving the dog into the
pillar, the dog spreads and passes with room, the obstacle-CBF stays feasible, no NaN, no
jam — and the formation **reforms** once past.

Preventive, not curative: it stops the crowding *before* the QP becomes infeasible (the
formation term is soft, so it does not itself cause infeasibility — it causes the
*trajectory* that leads into the infeasible corner).

### Why this over the alternatives

- **2b (V→single-file morph):** much more machinery (corridor detection + shape
  morphing); overkill for scattered pillars + a wide 2.5 m door. Rejected for now.
- **2c (soft-CBF fallback on infeasibility):** directly cures the NaN but **loosens
  safety** (allows the dog inside r_eff) and breaks the stack's "k=0 never softened"
  rule. Against the design philosophy and against the user's safety concern. Rejected.

2a keeps **safety hard throughout** (the CBF is untouched) and is the smallest change.

### Plum-blossom requirement (decided)

Plum must **keep safely reaching** — it is acceptable for the formation to loosen a bit
near pegs (NOT byte-identical). This is what makes 2a clean: the fade triggers on
obstacle proximity, which *also* fires in plum (dogs ride the peg r_eff boundary), but
that is fine — plum already threads, and a looser formation near pegs only makes it
easier. **Must be validated: plum still reaches safely after the change.**

---

## Design sketch (to firm up during implementation)

- **Where:** the per-dog formation gradient is assembled in the coordinator
  (`admm_coordinator.py` `_formation_grad` → node `_q` linear term). Scale each dog i's
  formation gradient by a factor `w_i ∈ [0,1]` computed from its clearance to the
  nearest obstacle. CBF assembly is untouched.
- **Fade function:** `w_i = clamp((d_obs_i - d_relax) / (d_clear - d_relax), 0, 1)`
  where `d_obs_i` = clearance of dog i to the nearest obstacle centre minus r_eff.
  - `d_clear`: above this, full formation (w=1).
  - `d_relax`: at/below this, formation off (w=0).
  - Starting guess: `d_relax ≈ 0.0` (at the r_eff boundary), `d_clear ≈ 0.4` m. **Tune
    on door-B + plum.**
- **Operating-point consistency:** the formation gradient is frozen once per control
  cycle (outside the ADMM loop) — the fade factor is computed there too, from the
  current true-body positions. No per-iteration change (keeps the frozen-coefficient
  discipline).
- **rospy-free:** the fade is pure geometry on positions + obstacle list; keep it in the
  coordinator/formation layer, not the publisher, so the offline gates exercise it.
- **Open question:** fade near *other dogs* too (inter-agent proximity), or obstacles
  only? Start obstacles-only (simplest, targets the root cause); add inter-agent only if
  door-B still jams.

## Validation plan (gates before claiming done)

1. **Offline:** `verify_door.py` (goal A) still ALL GREEN; a new offline check that goal
   B/C/D now reach (or settle) with min inter-agent ≥ D_MIN and no infeasibility.
2. **Offline plum regression:** `verify_plum.py` still ALL GREEN (reaches, barriers ≥0).
3. **Gazebo door:** goal B/C/D — dogs reach or settle SPREAD (min_pair ≥ D_MIN 0.60), no
   fall, no NaN holds, zero SqpSolver failures. Full-log red-flag scan
   (`run_door_campaign.sh`).
4. **Gazebo plum regression:** one plum run — still reaches, no fall (looser formation OK).
5. **Real-time video** of a fixed door-B (dogs spread + reach/settle, upright, no pileup).

## Test / tooling assets (already exist)

- `scratchpad/measure_door.py` — offline door probe (goal set editable).
- `scratchpad/arena/run_door.sh` — headless door run (honours `GOAL_ARGS`,
  `~astar_robot_radius`, `MON_HOLD`, `MON_TIMEOUT`).
- `scratchpad/arena/run_door_campaign.sh` — multi-goal + multi-speed + red-flag scan.
- `scratchpad/arena/run_door_record.sh` — records (Xvfb + software GL + ffmpeg x11grab).
- `scratchpad/arena/monitor.py` — RTF, reached_t, post-reach drift, `~goalN_x/y` override.
- Verify runs in the Docker container (osqp 0.6.3, ROS sourced). `pkill -9 -f <name>`
  self-kills — use `pkill -x python3`. Container degrades after ~20 back-to-back
  Gazebo bring-ups → `docker restart <id>` for a clean slate.

## What is NOT in scope

- 5-dog extension of the door arena.
- 2b/2c approaches.
- Making EVERY conceivable tight goal reach — goals that are *geometrically* impossible
  for a 3-dog formation (gap < D_MIN + 2·r_eff) will still not reach; 2a only removes the
  formation-induced crowding, not physical impossibility. That is acceptable and honest.
