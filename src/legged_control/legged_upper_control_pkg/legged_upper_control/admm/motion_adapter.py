"""Stage 4 -- ADMM ξ -> OCS2 centroidal target-trajectory adapter (Route B).

Maps one dog's ADMM decision vector ξ = [x_1..x_N (4N), a_0..a_{N-1} (2N)] (state
first, world-frame double integrator [px,py,vx,vy]) into an OCS2 legged_robot
centroidal *state* trajectory, publishable as ocs2_msgs/mpc_target_trajectories
(Path A: direct to OCS2, no legged_planner C++ in the loop).

OCS2 24-dim centroidal state layout (verified from the OCS2 C++, not the
legged_planner adapter's loose comment):
  getNormalizedMomentum = state[0:6]  = [normalized linear momentum (3),
                                          normalized angular momentum (3)]
  getBasePose           = state[6:12] = [base position x,y,z (3),
                                          base orientation euler-ZYX yaw,pitch,roll (3)]
  getJointAngles        = state[12:]  = joint angles (12)
  (ocs2_centroidal_model AccessHelperFunctionsImpl.h / CentroidalModelRbdConversions.cpp)

RATIFIED mapping (Tony):
  idx0,1 = vx,vy           normalized linear momentum == COM velocity (÷mass), direct
  idx2   = 0               vz
  idx3-5 = 0               normalized ANGULAR momentum -- NO reference; the MPC
                           self-computes the needed angular momentum from the yaw
                           (idx9) + linear velocity. Filling a yaw-RATE here is a
                           unit mismatch (kg·m²/s vs rad/s) and the wrong axis; the
                           spec C6.2 "idx3=ψ̇" is corrected to 0.
  idx6,7 = px,py           base position
  idx8   = COM_HEIGHT      base z (forced constant)
  idx9   = yaw             base orientation euler-Z (the yaw bypass target)
  idx10,11 = 0             pitch, roll
  idx12-23 = default_joints standing pose
  control/input trajectory = 0 (OCS2 recomputes GRFs from the state reference)

Padding uses EXPLICIT per-index assignment + a joint for-loop (never numpy fancy
reshape) so this file translates one-to-one to C++ later.

rospy-free: pure numpy + math. The ROS publisher (ocs2_msgs/mpc_target_trajectories)
is a thin outer wrapper added at Gazebo wiring; the mapping/yaw core stays testable
offline. yaw reuses core.geometry.wrap_to_pi (that module is pure math, ROS-free).
"""

import os
import sys
import math

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                        # constants
sys.path.insert(0, os.path.dirname(_HERE))       # core.geometry
import constants as C                             # noqa: E402
from core.geometry import wrap_to_pi              # noqa: E402  (pure math, ROS-free)


# --- OCS2 centroidal state index constants (named -> no magic ints in loops) ---
OCS2_STATE_DIM = 24
MOM_LIN_X, MOM_LIN_Y, MOM_LIN_Z = 0, 1, 2
MOM_ANG_X, MOM_ANG_Y, MOM_ANG_Z = 3, 4, 5
BASE_PX, BASE_PY, BASE_Z = 6, 7, 8
BASE_YAW, BASE_PITCH, BASE_ROLL = 9, 10, 11
JOINT_START, N_JOINTS = 12, 12


def pad_ocs2_state(px, py, vx, vy, yaw, com_height, default_joints):
    """4D ADMM state (+ bypass yaw) -> 24D OCS2 centroidal state. EXPLICIT index
    assignment (C++-translatable); every unset slot stays 0."""
    s = np.zeros(OCS2_STATE_DIM)
    s[MOM_LIN_X] = vx            # normalized linear momentum x = COM vx (÷mass)
    s[MOM_LIN_Y] = vy            # = COM vy
    # MOM_LIN_Z, MOM_ANG_X/Y/Z: stay 0 (no angular-momentum reference, ratified)
    s[BASE_PX] = px
    s[BASE_PY] = py
    s[BASE_Z] = com_height       # forced constant (flat ground)
    s[BASE_YAW] = yaw            # euler-Z; BASE_PITCH, BASE_ROLL stay 0
    for j in range(N_JOINTS):    # idx 12..23
        s[JOINT_START + j] = default_joints[j]
    return s


def yaw_step(vx, vy, prev_yaw, v_freeze, ema_alpha):
    """One yaw-bypass step: head toward the motion direction, freeze at low speed,
    EMA the WRAPPED increment (shortest arc) onto a CONTINUOUS accumulator.

    "Never let EMA cross ±π unwrapped" = never EMA raw atan2 (which jumps ±2π at the
    branch cut); we wrap the *increment* (raw - prev) instead, so the accumulator
    moves smoothly and MAY pass ±π continuously -- which a tracking reference needs.
    (Wrapped-vs-continuous for the OCS2 idx9 slot + seeding from the measured base
    yaw is confirmed at Gazebo wiring; continuity of the increment holds either way.)"""
    if math.hypot(vx, vy) < v_freeze:
        raw = prev_yaw                       # freeze -> avoid atan2(0,0) jitter
    else:
        raw = math.atan2(vy, vx)
    delta = wrap_to_pi(raw - prev_yaw)       # shortest-arc increment, never ±2π
    return prev_yaw + ema_alpha * delta      # accumulate continuously


def yaw_trajectory(vx_arr, vy_arr, seed_yaw, v_freeze, ema_alpha):
    """Sweep k over the horizon carrying prev_yaw. Returns (N,) yaws; the last one
    is the seed for the next control cycle (cross-cycle continuity)."""
    n = len(vx_arr)
    yaws = np.zeros(n)
    prev = seed_yaw
    for k in range(n):
        prev = yaw_step(vx_arr[k], vy_arr[k], prev, v_freeze, ema_alpha)
        yaws[k] = prev
    return yaws


def yaw_trajectory_path(px_arr, py_arr, seed_yaw, lookahead_M, pos_eps, ema_alpha):
    """Route B yaw: head along the PLANNED PATH (position lookahead), not the
    instantaneous velocity. yaw_k = atan2(p[k+M]-p[k]) over the SAME ξ positions.

    Why this kills the orbit that yaw_trajectory (velocity direction) causes: near
    the goal the toward-goal velocity -> 0 but the trot sway residual keeps rotating,
    so a velocity-direction yaw chases the sway -> body turns -> world velocity turns
    -> positive feedback -> orbit. A position lookahead is dominated by the (smooth,
    goal-directed, decelerating) path shape and averages the per-step sway out; when
    the trajectory has essentially stopped (‖p[k+M]-p[k]‖ < pos_eps) we FREEZE the
    heading instead of pointing at a vanishing-then-noisy direction -- that freeze is
    what breaks the residual->turn feedback chain.

    Same continuity contract as yaw_step: wrap only the INCREMENT and accumulate onto
    a continuous heading (may pass ±π), seed from the measured base yaw each cycle."""
    n = len(px_arr)
    yaws = np.zeros(n)
    prev = seed_yaw
    for k in range(n):
        j = min(k + lookahead_M, n - 1)          # clamp lookahead to horizon end
        dx = px_arr[j] - px_arr[k]
        dy = py_arr[j] - py_arr[k]
        if math.hypot(dx, dy) < pos_eps:
            raw = prev                            # trajectory ~stopped -> hold heading
        else:
            raw = math.atan2(dy, dx)
        prev = prev + ema_alpha * wrap_to_pi(raw - prev)   # continuous accumulate
        yaws[k] = prev
    return yaws


def safe_prefix_length(px_i, py_i, others_pos, d_safe, k_max):
    """Number of leading horizon steps (1..k_max) of dog i's predicted path that stay
    >= d_safe from EVERY other dog's predicted path at the SAME step.

    Truncating the OCS2 target to this prefix makes every sent x_ref point collision-
    safe by construction; past it OCS2 zero-order-hold CLAMPS to the last (safe) sent
    point (so the dog decelerates toward it rather than tracking the unsafe tail). This
    is the fix for the node-edge handoff hole: only k=0 is a HARD CBF step (a_max caps
    active hard steps ~2), so the ADMM plan itself dives unsafe at a close approach --
    we hand OCS2 only the part of the plan we can vouch for.

    px_i,py_i and each (px_j,py_j) in others_pos are step-aligned arrays (index m =
    horizon step m+1). Returns >= 1: the first step is ~current and kept safe by the
    k=0 hard CBF, so we always send at least it (never an empty target)."""
    K = 0
    kmax = min(int(k_max), len(px_i))
    for m in range(kmax):
        clear = True
        for (px_j, py_j) in others_pos:
            if math.hypot(px_i[m] - px_j[m], py_i[m] - py_j[m]) < d_safe:
                clear = False
                break
        if not clear:
            break
        K = m + 1
    return max(1, K)


class MotionAdapter:
    """Holds the padding config + per-dog yaw seed; turns one ADMM ξ into an OCS2
    target trajectory (times, states, zero inputs)."""

    def __init__(self, com_height, default_joints, n=None, ts=None,
                 v_freeze=0.05, ema_alpha=0.3, input_dim=24,
                 yaw_mode="path", lookahead_M=5, pos_eps=0.02, k_send=None):
        self.com_height = float(com_height)
        self.default_joints = np.asarray(default_joints, dtype=float)
        assert self.default_joints.shape == (N_JOINTS,), \
            f"default_joints must be {N_JOINTS}-vector"
        self.N = C.N if n is None else int(n)
        self.ts = C.TS if ts is None else float(ts)
        # send only the first k_send horizon steps to OCS2 (<= its 1.0s = 10-step
        # horizon); the far tail of xi is only soft-safe and drifts unsafe past ~k=10,
        # so we drop it and let OCS2 clamp to the last (safe) sent point. See C.K_SEND.
        self.k_send = C.K_SEND if k_send is None else int(k_send)
        self.v_freeze = float(v_freeze)
        self.ema_alpha = float(ema_alpha)
        self.input_dim = int(input_dim)
        # yaw_mode: 'path' (Route B, position lookahead -- default, kills the orbit) or
        # 'tangent' (legacy velocity direction; kept for arc mode + A/B diagnostics).
        self.yaw_mode = yaw_mode
        # ponytail: M=5 steps (0.5s @ 10Hz) / pos_eps=2cm are hardware knobs -- tune on
        # Gazebo (exposed as ~lookahead_m / ~pos_eps params in the publisher).
        self.lookahead_M = int(lookahead_M)
        self.pos_eps = float(pos_eps)
        self._seed = {}                       # per-dog yaw seed (cross-cycle)

    def build_target(self, xi, t0=0.0, seed_yaw=0.0, yaw_mode=None, k_send=None):
        """ξ (6N) -> dict(times(ns,), states(ns,24), inputs(ns,input_dim), next_seed),
        ns = k_send (default self.k_send). State x_k (k=1..ns) is sent at time
        t0 + k*Ts; accelerations are dropped. yaw is computed over the FULL horizon
        (its lookahead needs the real future) then truncated to the sent steps.
        yaw_mode overrides self.yaw_mode for this call ('path' | 'tangent')."""
        N = self.N
        ns = self.k_send if k_send is None else int(k_send)
        ns = max(1, min(ns, N))               # clamp to [1, N]
        xi = np.asarray(xi, dtype=float)
        px = np.array([xi[C.px_index(k, N)] for k in range(1, N + 1)])
        py = np.array([xi[C.py_index(k, N)] for k in range(1, N + 1)])
        vx = np.array([xi[C.vx_index(k, N)] for k in range(1, N + 1)])
        vy = np.array([xi[C.vy_index(k, N)] for k in range(1, N + 1)])
        mode = self.yaw_mode if yaw_mode is None else yaw_mode
        if mode == "tangent":
            yaws = yaw_trajectory(vx, vy, seed_yaw, self.v_freeze, self.ema_alpha)
        else:                                 # 'path' (Route B, default)
            yaws = yaw_trajectory_path(px, py, seed_yaw, self.lookahead_M,
                                       self.pos_eps, self.ema_alpha)

        times = np.zeros(ns)                  # send only the first ns (<=N) steps
        states = np.zeros((ns, OCS2_STATE_DIM))
        for j in range(ns):
            k = j + 1
            times[j] = t0 + k * self.ts
            states[j] = pad_ocs2_state(px[j], py[j], vx[j], vy[j], yaws[j],
                                       self.com_height, self.default_joints)
        inputs = np.zeros((ns, self.input_dim))   # zero control (OCS2 recomputes GRF)
        return dict(times=times, states=states, inputs=inputs,
                    next_seed=float(yaws[ns - 1]))

    def adapt(self, xi, dog, t0=0.0):
        """Stateful per-dog wrapper: uses/updates this dog's yaw seed across cycles."""
        out = self.build_target(xi, t0=t0, seed_yaw=self._seed.get(dog, 0.0))
        self._seed[dog] = out["next_seed"]
        return out
