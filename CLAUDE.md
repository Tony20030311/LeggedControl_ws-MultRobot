# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

Three Unitree A1 quadruped robots in multi-robot formation control research.
Stack: ROS Noetic + Gazebo + OCS2 (MPC/WBC).

**Development scope: only modify Python — the upper-control layer (L1–L4). Two locations hold it:**
- `src/legged_control/legged_upper_control_pkg/` — the current package: the ADMM system
  (`scripts/ocs2_fleet_publisher.py` + `legged_upper_control/admm/`, the ACTIVE upper control)
  and the legacy modular Pure-Pursuit stack (`legged_upper_control/{core,controllers,fleet}`)
- `src/legged_control/legged_controllers/scripts/` — the RViz debug visualizer, the gait
  broadcaster, and `start_fleet.sh` (the monolithic single-file managers were **deleted** — see below)

**Never touch C++ under `src/` (OCS2, legged_control, pinocchio, hpp-fcl).** The L0 MPC/WBC is fixed.

## Working Mode

This is research code; the user wants a thinking partner, not an order-taker. On any non-trivial change, work it out *together* before writing code:

- **Derive the math together.** For any CBF / QP / MPC / formation change, write the formulation out step by step — barrier function and its derivatives, the HOCBF inequality, QP decision variables / objective / constraints, the integration step — and check it with the user before implementing.
- **Trace the data flow together.** State the path explicitly: which input topic/state, which frame transform, which layer (L4→L1), what gets published to `/dogN/cmd_vel`. Confirm it matches reality (`graphify query`, the C++ topic contracts) before changing it.
- **Work through the implementation together.** Propose the approach and the exact edit points first; give an honest assessment — risks, a simpler alternative, and *which* of the duplicated copies it touches and what will drift — instead of silently implementing.

Read-only experiments and analysis are fine without asking. Get agreement before editing code.

## Knowledge Graph (read this first)

A graphify knowledge graph of `src/legged_control` lives in `graphify-out/`. A `SessionStart` hook auto-loads `graphify-out/ARCH_PRIMER.md` into context each session. Before grepping/reading source **under `src/legged_control`** to understand the codebase, query the graph (a `PreToolUse` hook enforces this):

```bash
graphify query "<question>"        # scoped subgraph answer
graphify explain "<concept>"       # plain-language node explanation
graphify path "<A>" "<B>"          # shortest path between two symbols
```

**Exception:** files under `references/` (cloned open-source reference implementations) and `docs/` (design documents) are outside the graph — read them directly, no query needed. The `PreToolUse` hook exempts these paths.

Full audit: `graphify-out/GRAPH_REPORT.md`. Rebuild after large code changes: `/graphify src/legged_control --update`.

## Five-Layer Control Architecture

```
L4  Formation        — formation geometry (V-shape / Laplacian), follower targets
L3  A* Planner       — global grid path planning around obstacles (with inflation)
L2  Pure Pursuit     — holonomic waypoint tracking → (vx, vy, wz) commands
L1  CBF-QP           — safety filter, MODIFIABLE (slack, weights, stuck/door handling)
L0  MPC (OCS2 C++)   — NMPC + WBC at 400 Hz, NOT touched
```

L0 is C++; everything above is the Python upper-control layer you edit.

> **⚠️ L1–L4 are being refactored into a trajectory-level ADMM-DMPC system**
> (world-frame double integrator, node-edge splitting, 10 Hz, second-order HOCBF,
> reference tracks position only). The five layers above describe the **legacy
> Pure-Pursuit stack**, which still runs during migration. For the new architecture,
> its source-of-truth documents (`docs/`), the open-source navigation (`references/`),
> and all ADMM implementation rules, see **`legged_upper_control_pkg/CLAUDE.md`**.

## Two Upper-Control Implementations (important)

The same L1–L4 stack exists in two forms. Know which you are editing.

**A. Modular package — `legged_upper_control_pkg/legged_upper_control/` (current/preferred):**

| Module | Role |
|--------|------|
| `core/state.py` | World-frame pose/velocity from `/dogN/ground_truth/state` |
| `core/config.py` | YAML param loading (`get_config()`) |
| `core/geometry.py` | AABB / `closest_point_on_aabb()` and geometry helpers |
| `core/formation.py` | `FormationSwitcher`, formation offsets, door-passage state |
| `core/navigation.py` | `LeaderNavigator`, `LeaderCmdRelay`, stuck/recovery logic |
| `core/planning.py` | `AStarPlanner` (grid build, inflation, reachable-goal search) |
| `core/io.py` | `AccelCmdPublisher`, `CbfDebugPublisher`, `CmdVelPublisher` |
| `controllers/base.py` | `CBFControllerBase` abstract controller |
| `controllers/accel_hocbf.py` | `TwoOrderCBFQPController` — acceleration-level HOCBF QP |
| `controllers/velocity_qp.py` | `UnifiedQPController` — velocity-level CBF QP |
| `fleet/manager.py` | `FleetManagerUQP` — slot assignment, per-dog nominal velocity, top-level orchestration |
| `scripts/fleet_manager_node.py` | ROS entry point: instantiates `FleetManagerUQP().spin()` |
| `launch/formation_hocbf.launch` | Launches fleet manager + debug visualizer |
| `config/Cbf_params_twoOrderCBF.yaml` | Params for the modular HOCBF path |

Section A is the **legacy Pure-Pursuit stack**, now launched only by `formation_hocbf.launch`
and being retired as the ADMM system takes over. The active system is the ADMM publisher:

**B. ADMM upper-control (the ACTIVE system) — `legged_upper_control_pkg/scripts/ocs2_fleet_publisher.py` + `legged_upper_control/admm/`:**
trajectory-level ADMM-DMPC fleet publisher (world-frame double integrator, node-edge splitting,
second-order HOCBF, `/formation/goal` centroid command → OCS2 24D target). Run by the
`three_dogs_{empty,obstacles,door,plum}.launch` + `admm_demo.launch` flows. Full rules:
`legged_upper_control_pkg/CLAUDE.md`.

**Support scripts in `legged_controllers/scripts/`:** `formation_debug_visualizer.py` (legacy
RViz markers), `gait_broadcaster.py` (UDP gait, must start before fleet bringup; see Known
Issues), `start_fleet.sh`, `Cbf_params_uqp.yaml`.

> **DELETED 2026-07-09 (commit 522d77b/fb5aa6c) — don't look for these:** the monolithic
> `Formation_manager_unified_{twoOrderCBF,qp}.py`, their debug launches, `scripts/archive/`, the
> duplicate `scripts/Cbf_params_twoOrderCBF.yaml`, `ocs2_target_publisher.py`, `apps/`. They
> duplicated the modular package (now the single copy) or were orphaned.

## Build Commands

ROS1 catkin workspace (despite some ROS2-style conventions). Use `catkin_tools`:

```bash
source source.sh                                   # or: source devel/setup.bash
catkin config -DCMAKE_BUILD_TYPE=RelWithDebInfo    # one-time

# OCS2 deps (slow, ~10 min, once)
catkin build ocs2_legged_robot_ros ocs2_self_collision_visualization

catkin build legged_controllers legged_unitree_description
catkin build legged_upper_control                  # modular upper-control package
catkin build legged_gazebo                         # simulation
catkin build legged_controllers --no-deps          # rebuild one package
```

There are no automated tests **for the legacy stack**; its verification is by launching the system in Gazebo and observing behavior (real runs, both inside/outside formation, many points; watch synchronized `/cbf_debug/h_min_pair`). The ADMM-DMPC package under development introduces its own Python-level automated tests (coefficient guards and per-stage verification scripts) — see `legged_upper_control_pkg/CLAUDE.md`.

## Running the System

```bash
# 1. Gazebo with obstacle world
roslaunch legged_unitree_description obstacle_world.launch    # or five_dogs.launch

# 2. Load ROS controllers (staggered 0/15/30s delays built in)
roslaunch legged_controllers fleet_bringup.launch

# 3. Stand up + switch to trot gait
rosrun legged_controllers start_fleet.sh

# 4a. Legacy modular Pure-Pursuit stack (also starts the debug visualizer)
roslaunch legged_upper_control formation_hocbf.launch

# 4b. OR the ACTIVE ADMM system (kills C++ target + publisher + RViz rollout viz):
#     use three_dogs_plum.launch / three_dogs_door.launch in step 1, then:
roslaunch legged_upper_control admm_demo.launch
#     then command ONE centroid point:
#     rostopic pub -1 /formation/goal geometry_msgs/PoseStamped \
#       "{header: {frame_id: 'world'}, pose: {position: {x: 9.0, y: 0.0, z: 0.5}}}"

# 5a. Keyboard (publishes raw; CBF filters before cmd_vel)
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/dog1/cmd_vel_raw
# 5b. OR goal-based auto navigation
rostopic pub /dog1/goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 8.0, y: 0.0, z: 0.5}}}"
```

## Architecture Overview

```
Python / 10 Hz      ocs2_fleet_publisher.py (ACTIVE ADMM) — or the legacy FleetManagerUQP @ 20 Hz
                    └─ L4 Formation: V-shape / Laplacian offsets, follower targets
                    └─ L3 AStarPlanner: grid path planning with inflation
                    └─ L2 PurePursuitController: waypoint tracking → (vx, vy, wz)
                    └─ L1 CBF-QP: TwoOrderCBFQPController (HOCBF accel) / UnifiedQPController (vel)
                       — native OSQP solver, single-integrator model
                    └─ publishes safe velocities → /dogN/cmd_vel

C++ / 400 Hz        LeggedController (ros_control plugin)
                    └─ NMPC via OCS2 (SQP → HPIPM): state + gait optimization
                    └─ WBC hierarchical QP: contact > swing feet > torque min
                    └─ impedance commands (kp, kd, feedforward torque)

C++ / 1000 Hz       LeggedHWSim (Gazebo plugin): joint/IMU reads, hybrid impedance application
```

**Key C++ packages in `src/legged_control/`** (read-only): `legged_controllers` (ros_control plugin, primary C++ entry), `legged_wbc` (WeightedWbc/HierarchicalWbc, HoQP → qpoases), `legged_interface` (OCS2 NMPC setup, constraints, costs), `legged_estimation` (Kalman / FromTopic estimators), `legged_hw`/`legged_gazebo` (HW + sim). `legged_common` is the dependency base; `legged_controllers` is the top integrator (also pulls OCS2 ROS packages). External: `ocs2`, `pinocchio`, `hpp-fcl`.

## CBF Parameters

- Modular: `legged_upper_control_pkg/config/Cbf_params_twoOrderCBF.yaml`
- Legacy velocity-QP: `legged_controllers/scripts/Cbf_params_uqp.yaml` (`cbf_lookahead_tau`), still read by `formation_debug_visualizer.py`. (The monolith `Cbf_params_twoOrderCBF.yaml` copy was deleted; `config/Cbf_params_twoOrderCBF.yaml` is now the single copy.)
- Common keys: `cbf_enabled`, `cbf_d_min`, `cbf_slack_enabled`/`slack_lambda`, formation `offsets`, `obstacles`/`walls` (manually tuned to match the Gazebo arena), `followers_stationary`.
- `followers_stationary: true` is the safe default (followers hold position); set `false` for PID tracking only after the leader walks stably.

## Multi-Robot Namespace Isolation

Each dog (dog1/dog2/dog3) is isolated through 4 layers: (1) Gazebo `robotNamespace`, (2) launch `<group ns="dogN">`, (3) C++ private `NodeHandle`, (4) per-dog URDF generation. Topics follow `/dogN/<topic>` (`/dog1/cmd_vel`, `/dog2/ground_truth/state`, …).

## Known Issues / Design Notes

- **Code duplication (mostly resolved 2026-07-09):** the 3–4× copies of A*, `FleetManagerUQP`,
  the CBF controllers, and `Cbf_params_twoOrderCBF.yaml` (modular + monolith + `archive/`) were
  cut down — the monoliths and `archive/` are **deleted**; the modular package is the single
  copy. Remaining legacy duplication risk is small. The ADMM system (`ocs2_fleet_publisher.py` +
  `admm/`) is a separate, non-duplicated implementation and is where new work goes.
- `gait_broadcaster.py` must run before `fleet_bringup.launch` completes or the GaitReceiver UDP port binding races; the 15s stagger in `fleet_bringup.launch` mitigates this.
- The CBF QP uses a **single-integrator model** (velocity = input), intentionally — the low-level C++ controller handles full dynamics; CBF only needs velocity/acceleration-level safety.
- Obstacle/wall definitions in the param YAMLs are hand-tuned to the Gazebo arena (`five_dogs.launch`); changing the arena requires updating both.

## Key Config Files (C++ side, read-only)

| File | Purpose |
|------|---------|
| `legged_controllers/config/a1/task.info` | NMPC cost weights and constraints |
| `legged_controllers/config/a1/reference.info` | Target COM height, default joint poses, gait schedule |
| `legged_controllers/launch/fleet_bringup.launch` | Loads controllers for all 3 dogs |
| `legged_examples/.../five_dogs.launch` | Gazebo obstacle arena + 3 dogs |