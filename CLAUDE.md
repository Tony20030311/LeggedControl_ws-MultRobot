# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

Three Unitree A1 quadruped robots in multi-robot formation control research.
Stack: ROS Noetic + Gazebo + OCS2 (MPC/WBC).

**Development scope: only modify Python files under `scripts/`. Never touch C++ under `src/` (OCS2, legged_control).**

## Five-Layer Control Architecture

```
L4  FormationManager   — formation geometry, follower target computation
L3  A* Planner         — global path planning around obstacles (scripts/astar_planner.py)
L2  Pure Pursuit       — holonomic waypoint tracking → (vx, vy, wz) commands
L1  CBF-QP             — safety filter, MODIFIABLE for robustness work (slack, weights, stuck-handling)
L0  MPC (OCS2 C++)     — NMPC + WBC running at 400 Hz, not touched
```

Main script: `src/legged_control/legged_controllers/scripts/formation_managerCBF.py`
New module: `src/legged_control/legged_controllers/scripts/astar_planner.py`
Parameters: `src/legged_control/legged_controllers/scripts/cbf_params.yaml`

## Current Progress

- **Step 1**: Single-robot A* path planning (in progress)
- **Step 2–5**: See `architecture_walkthrough.md`

## Build Commands

This is a ROS1 catkin workspace (not ROS2 despite some conventions). Use `catkin_tools`:

```bash
# Source workspace
source source.sh   # or: source devel/setup.bash

# Configure build type (one-time)
catkin config -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build OCS2 dependencies (slow, ~10 min, only needed once)
catkin build ocs2_legged_robot_ros ocs2_self_collision_visualization

# Build main packages
catkin build legged_controllers legged_unitree_description
catkin build legged_gazebo   # simulation only

# Rebuild a single package
catkin build legged_controllers --no-deps
```

There are no automated tests. Verification is done by launching the system and observing behavior.

## Running the System (5 terminals)

```bash
# 1. Gazebo simulation with obstacle world
roslaunch legged_unitree_description obstacle_world.launch   # or five_dogs.launch

# 2. Load ROS controllers (staggered 0/15/30s delays built into launch file)
roslaunch legged_controllers fleet_bringup.launch

# 3. Stand up + switch to trot gait
rosrun legged_controllers start_fleet.sh

# 4. Formation manager with CBF safety filter
rosrun legged_controllers formation_managerCBF.py

# 5a. Keyboard control (publishes to raw topic, CBF filters before cmd_vel)
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/dog1/cmd_vel_raw

# 5b. OR goal-based auto navigation
rostopic pub /dog1/goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 8.0, y: 0.0, z: 0.5}}}"
```

## Architecture Overview

```
Python / 20 Hz      formation_managerCBF.py  +  astar_planner.py
                    └─ L4 FormationManager: V-shape offsets, follower targets
                    └─ L3 AStarPlanner: grid path planning (astar_planner.py)
                    └─ L2 PurePursuitController: waypoint tracking → (vx, vy, wz)
                    └─ L1 CBFSafetyFilter: QP safety (CVXPY + OSQP) — modifiable for slack/stuck recovery
                    └─ publishes safe velocities → /dogN/cmd_vel

C++ / 400 Hz        LeggedController (ros_control plugin)
                    └─ NMPC via OCS2 (SQP → HPIPM): state + gait optimization
                    └─ WBC hierarchical QP: contact constraints > swing feet > torque min
                    └─ Outputs impedance commands (kp, kd, feedforward torque)

C++ / 1000 Hz       LeggedHWSim (Gazebo plugin)
                    └─ Joint states + IMU reads
                    └─ Hybrid impedance command application
```

**Key packages in `src/legged_control/`:**
- `legged_controllers` — ROS Control plugin; the primary C++ entry point
- `legged_wbc` — Whole-Body Controller (WeightedWbc, HierarchicalWbc)
- `legged_interface` — OCS2 NMPC setup, constraints, cost functions
- `legged_estimation` — Kalman filter state estimator
- `legged_hw` / `legged_gazebo` — Hardware and simulation abstraction

**External libraries (in `src/`):** `ocs2` (optimal control), `pinocchio` (rigid body dynamics), `hpp-fcl` (collision detection).

## Key Files

| File | Purpose |
|------|---------|
| `src/legged_control/legged_controllers/scripts/formation_managerCBF.py` | Main formation + CBF script (865 lines, 9 classes) |
| `src/legged_control/legged_controllers/scripts/cbf_params.yaml` | All tunable parameters (CBF gains, formation offsets, obstacles, walls) |
| `src/legged_control/legged_controllers/launch/fleet_bringup.launch` | Loads controllers for all 3 dogs |
| `src/legged_control/legged_examples/legged_unitree/legged_unitree_description/launch/five_dogs.launch` | Gazebo world with obstacle arena + 3 dogs |
| `src/legged_control/legged_controllers/config/a1/task.info` | NMPC cost weights and constraints |
| `src/legged_control/legged_controllers/config/a1/reference.info` | Target COM height, default joint poses, gait schedule |

## formation_managerCBF.py — Class Map

| Class | Role |
|-------|------|
| `StateCollector` | Subscribes `/dogN/ground_truth/state`; outputs world-frame pose + velocity |
| `FormationPlanner` | Computes follower target positions from leader pose + YAML offsets |
| `NominalController` | PID tracking of formation targets; produces body-frame velocity commands |
| `AStarPlanner` | Grid path planning with obstacle inflation |
| `PurePursuitController` | Holonomic waypoint tracking; outputs (vx, vy, wz) |
| `LeaderNavigator` | Switches leader between KEYBOARD and AUTO (A* + PurePursuit) modes |
| `CBFSafetyFilter` | QP safety filter; 3 constraint types: robot-robot, obstacle, wall |
| `VelocityLimiter` | Hard clamp on vx, vy, wz magnitudes |
| `CmdVelPublisher` | Publishes final velocities to `/dogN/cmd_vel` |

## CBF Parameters (`cbf_params.yaml`)

- `cbf_enabled` / `cbf_d_min` / `cbf_gamma*` — enable CBF, minimum inter-robot distance, barrier aggressiveness
- `offsets` — formation geometry (body-frame x, y offsets per follower)
- `followers_stationary: true` — Phase 1 (followers hold position); `false` — Phase 2+ (PID tracking active)
- `obstacles` / `walls` — manual obstacle YAML (future: replaced by map_server)

## Multi-Robot Namespace Isolation

Each dog (dog1/dog2/dog3) is isolated through 4 layers:
1. Gazebo `robotNamespace` parameter
2. ROS launch `<group ns="dogN">`
3. C++ `NodeHandle` private namespaces
4. Per-dog URDF file generation

Topics follow the pattern `/dogN/<topic>` (e.g., `/dog1/cmd_vel`, `/dog2/ground_truth/state`).

## Known Issues / Design Notes

- `gait_broadcaster.py` must run before `fleet_bringup.launch` completes, or the GaitReceiver UDP port binding races. The 15s stagger in `fleet_bringup.launch` mitigates this.
- The CBF QP uses a **single integrator model** (velocity = input), not the full robot dynamics. This is intentional — the low-level controller handles dynamics; CBF only needs velocity-level safety.
- `followers_stationary: true` is the safe default. Switch to `false` only after verifying leader is walking stably.
- Obstacle and wall definitions in `cbf_params.yaml` are manually tuned to match the Gazebo arena in `five_dogs.launch`. Changing the arena requires updating both files.

## Research Roadmap (from cbf_notes.md)

- **Phase 1 (current)**: Manual obstacle YAML + keyboard/goal navigation
- **Phase 2**: PID formation tracking (`followers_stationary: false`)
- **Phase 3**: `map_server` → automatic obstacle detection from static map
- **Phase 4**: Global path planning (move_base or EGO planner)
- **Phase 5**: SLAM (gmapping + LiDAR simulation)
- **Phase 6**: CBBA distributed task allocation
- **Phase 7**: VLA (Vision Language Agent) instruction following
