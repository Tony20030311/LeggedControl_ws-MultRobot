"""Stage 4 unit gate -- ADMM ξ -> OCS2 target trajectory adapter.

These pin the padding index placement (the whole point of the adapter) and the yaw
bypass edge cases (low-speed freeze, +-pi wrap continuity). The ratified mapping
(Tony): vx->0, vy->1, px->6, py->7, z->COM_HEIGHT@8, yaw->9, joints->12..23, and
EVERYTHING ELSE 0 -- crucially idx3-5 (normalized angular momentum) stays 0, NOT a
yaw-rate (unit-mismatch; MPC self-computes the needed angular momentum).

rospy-free (pure numpy + math) -> runs on host or in the container.
"""

import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admm_impl import constants as C  # noqa: E402
from admm_impl import ma  # noqa: E402

DEFJ = np.arange(1.0, 13.0)    # distinct 12-vector -> any mis-placement shows


def _straight_xi(vx=0.3):
    """Synthetic ADMM ξ: moving +x at constant vx, y/vy = 0."""
    N = C.N
    xi = np.zeros(C.xi_dim(N))
    for k in range(1, N + 1):
        xi[C.px_index(k, N)] = 0.1 * k
        xi[C.vx_index(k, N)] = vx
    return xi


# ---- padding index placement (the ratified mapping table) --------------------
def test_pad_index_placement():
    s = ma.pad_ocs2_state(px=1.5, py=-2.5, vx=0.3, vy=-0.1, yaw=0.7,
                          com_height=0.42, default_joints=DEFJ)
    assert s.shape == (ma.OCS2_STATE_DIM,)
    assert s[0] == 0.3 and s[1] == -0.1           # normalized linear momentum = vx,vy
    assert s[6] == 1.5 and s[7] == -2.5           # base position px,py
    assert s[8] == 0.42                            # base z forced to COM_HEIGHT
    assert s[9] == 0.7                             # base yaw (euler-Z)
    np.testing.assert_array_equal(s[12:24], DEFJ)  # joints


def test_ratified_zero_slots():
    """vz(2), angular momentum(3,4,5), pitch/roll(10,11) MUST be 0 regardless of
    inputs -- idx3-5 especially: NOT a yaw-rate (Tony's correction to spec C6.2)."""
    s = ma.pad_ocs2_state(px=9.0, py=9.0, vx=9.0, vy=9.0, yaw=2.9,
                          com_height=0.3, default_joints=DEFJ)
    for idx in (ma.MOM_LIN_Z, ma.MOM_ANG_X, ma.MOM_ANG_Y, ma.MOM_ANG_Z,
                ma.BASE_PITCH, ma.BASE_ROLL):
        assert s[idx] == 0.0, f"idx {idx} must be 0"


# ---- yaw bypass edge cases ---------------------------------------------------
def test_yaw_straight_x_converges_to_zero():
    y = 0.9
    for _ in range(60):
        y = ma.yaw_step(0.3, 0.0, y, v_freeze=0.05, ema_alpha=0.3)
    assert abs(y) < 1e-3


def test_yaw_low_speed_freeze_no_nan():
    """|v| < freeze -> hold previous heading; atan2(0,0) never evaluated."""
    y = ma.yaw_step(0.0, 0.0, prev_yaw=1.234, v_freeze=0.05, ema_alpha=0.3)
    assert not math.isnan(y)
    assert y == 1.234


def test_yaw_wrap_continuous_crossing():
    """Heading ramping 150deg -> 210deg (crosses +180). The wrapped-delta EMA must
    accumulate CONTINUOUSLY: consecutive diffs small (no spurious ~2*pi jump), yaw
    increases monotonically, and it crosses pi smoothly to > pi (a continuous
    reference) instead of snapping back to -150deg. Wrap-vs-continuous for the OCS2
    idx9 reference is confirmed at wiring; continuity is the invariant either way."""
    angs = np.deg2rad(np.linspace(150.0, 210.0, 40))
    vx, vy = np.cos(angs), np.sin(angs)
    ys = ma.yaw_trajectory(vx, vy, seed_yaw=angs[0], v_freeze=0.05, ema_alpha=0.5)
    d = np.diff(ys)
    assert np.all(np.abs(d) < 0.5)                     # smooth: no ~2*pi jump
    assert np.all(d >= -1e-9)                           # monotone increasing heading
    assert ys[-1] > math.pi                             # crossed pi continuously
    assert ys[-1] < np.deg2rad(210.0) + 0.2             # tracks the true heading


def test_yaw_seed_continuity():
    """Feeding one trajectory's last yaw as the next seed is continuous."""
    vx = np.full(5, 0.3); vy = np.full(5, 0.1)
    a = ma.yaw_trajectory(vx, vy, seed_yaw=0.0, v_freeze=0.05, ema_alpha=0.4)
    b = ma.yaw_trajectory(vx, vy, seed_yaw=a[-1], v_freeze=0.05, ema_alpha=0.4)
    assert abs(ma_wrapdiff(b[0], a[-1])) <= 0.4 * math.pi + 1e-9


def ma_wrapdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ---- Route B: path-lookahead yaw (the orbit fix) -----------------------------
def test_path_yaw_ignores_velocity_sway():
    """THE orbit root-cause regression. Path is a straight +x line, but the velocity
    swings ±60deg (trot sway). Velocity-direction (tangent) yaw chases the sway ->
    orbit; position-lookahead (path) yaw follows the straight path and stays put."""
    n = 20
    px = np.array([0.1 * (k + 1) for k in range(n)])   # straight +x
    py = np.zeros(n)
    vx = np.full(n, 0.15)
    vy = np.array([0.30 * math.sin(k) for k in range(n)])   # rotating sway
    tangent = ma.yaw_trajectory(vx, vy, 0.0, v_freeze=0.05, ema_alpha=0.5)
    path = ma.yaw_trajectory_path(px, py, 0.0, lookahead_M=5, pos_eps=0.02, ema_alpha=0.5)
    assert np.ptp(path) < 0.05                          # path yaw ~constant along +x
    assert np.ptp(tangent) > 0.5                        # tangent yaw chases the sway


def test_path_yaw_slew_rate_cap():
    """Sharp corner (path turns ~135deg mid-horizon). Without a cap the yaw whipsaws
    (a big single-step jump -> trot spins -> topples). With max_dyaw the per-step yaw
    change is bounded to the cap -- the fix for the plum round-trip return topple."""
    n = 20
    # L-shaped path: go +x for 10 steps, then +y (a hard left turn) for 10.
    px = np.array([0.1 * (k + 1) if k < 10 else 1.0 for k in range(n)])
    py = np.array([0.0 if k < 10 else 0.1 * (k - 9) for k in range(n)])
    uncapped = ma.yaw_trajectory_path(px, py, 0.0, lookahead_M=5, pos_eps=0.02,
                                      ema_alpha=0.5, max_dyaw=None)
    capped = ma.yaw_trajectory_path(px, py, 0.0, lookahead_M=5, pos_eps=0.02,
                                    ema_alpha=0.5, max_dyaw=0.1)
    assert np.max(np.abs(np.diff(uncapped))) > 0.15    # corner causes a big yaw step
    assert np.all(np.abs(np.diff(capped)) <= 0.1 + 1e-9)  # every step within the cap
    assert abs(capped[0]) <= 0.1 + 1e-9                 # first (sent) step also bounded


def test_path_yaw_points_at_goal_and_freezes():
    """Decelerating-into-goal trajectory (steps shrink geometrically toward a 45deg
    goal). yaw converges to the goal heading, then FREEZES once ‖p[k+M]-p[k]‖ < eps,
    with no wind-up (small steps, bounded, no ~2pi jump)."""
    n = 20
    r = 0.7
    px = np.array([2.0 * (1.0 - r ** (k + 1)) for k in range(n)])   # -> goal_x = 2
    py = np.array([2.0 * (1.0 - r ** (k + 1)) for k in range(n)])   # -> goal_y = 2
    y = ma.yaw_trajectory_path(px, py, 0.0, lookahead_M=5, pos_eps=0.02, ema_alpha=0.5)
    assert abs(y[-1] - math.pi / 4) < 0.05             # settled at the 45deg goal heading
    assert np.all(np.abs(np.diff(y[-4:])) < 1e-9)      # tail frozen (trajectory stopped)
    assert np.all(np.abs(np.diff(y)) < 0.5)            # smooth: no ~2pi jump
    assert abs(y[-1]) < math.pi                         # bounded, no wind-up


def test_path_yaw_stationary_freezes_no_nan():
    """A trajectory sitting on the goal (all points equal) holds the seed heading --
    no atan2(0,0) NaN, no drift."""
    px = np.full(8, 3.0); py = np.full(8, -1.0)
    y = ma.yaw_trajectory_path(px, py, seed_yaw=1.234, lookahead_M=5,
                               pos_eps=0.02, ema_alpha=0.4)
    assert not np.any(np.isnan(y))
    assert np.all(y == 1.234)


def test_build_target_default_is_path():
    """The adapter defaults to Route B: build_target() with no yaw_mode uses path
    lookahead, and per-call 'tangent' recovers the legacy velocity yaw."""
    adap = ma.MotionAdapter(com_height=0.42, default_joints=DEFJ)
    assert adap.yaw_mode == "path"
    # straight +x ξ: both modes give ~0 yaw, but the code path must differ without error
    out_path = adap.build_target(_straight_xi(), t0=0.0, seed_yaw=0.0)
    out_tan = adap.build_target(_straight_xi(), t0=0.0, seed_yaw=0.0, yaw_mode="tangent")
    assert out_path["states"].shape == out_tan["states"].shape


# ---- full assembly from a real ξ ---------------------------------------------
def test_build_target_shapes_times_inputs():
    # build_target truncates to K_SEND steps (only OCS2's ~1s horizon is handed over).
    K = C.K_SEND
    adap = ma.MotionAdapter(com_height=0.42, default_joints=DEFJ, input_dim=24)
    out = adap.build_target(_straight_xi(), t0=1.0, seed_yaw=0.0)
    assert out["times"].shape == (K,)
    assert np.all(np.diff(out["times"]) > 0)
    assert abs(out["times"][0] - (1.0 + C.TS)) < 1e-9
    assert abs(out["times"][-1] - (1.0 + K * C.TS)) < 1e-9
    assert out["states"].shape == (K, ma.OCS2_STATE_DIM)
    assert out["inputs"].shape == (K, 24) and np.all(out["inputs"] == 0.0)
    # explicit k_send override wins
    assert adap.build_target(_straight_xi(), k_send=3)["times"].shape == (3,)


def test_build_target_places_admm_state():
    N = C.N
    xi = _straight_xi()
    adap = ma.MotionAdapter(com_height=0.42, default_joints=DEFJ)
    out = adap.build_target(xi, t0=0.0, seed_yaw=0.0)
    for j in range(C.K_SEND):                                  # only the sent steps
        k = j + 1
        assert out["states"][j, 6] == xi[C.px_index(k, N)]     # px -> idx6
        assert out["states"][j, 7] == xi[C.py_index(k, N)]     # py -> idx7
        assert out["states"][j, 0] == xi[C.vx_index(k, N)]     # vx -> idx0
        assert out["states"][j, 8] == 0.42                      # z forced
        assert out["states"][j, 3] == 0.0                       # angular mom stays 0


# ---- dynamic safe-prefix truncation (the node-edge handoff fix) ---------------
def test_safe_prefix_truncates_at_first_violation():
    # dog i marches +x (0.1,0.2,...); dog j sits at x=1.0. gap = |1.0 - 0.1(m+1)|.
    # m=0..2 gap .9/.8/.7 clear; m=3 gap .6 (not < .6, clear); m=4 gap .5 (<.6) -> K=4.
    px_i = np.array([0.1 * (m + 1) for m in range(C.N)]); py_i = np.zeros(C.N)
    px_j = np.full(C.N, 1.0); py_j = np.zeros(C.N)
    K = ma.safe_prefix_length(px_i, py_i, [(px_j, py_j)], d_safe=0.6, k_max=C.N)
    assert K == 4, K
    assert all(abs(1.0 - 0.1 * (m + 1)) >= 0.6 for m in range(K))       # every sent step clear
    assert abs(1.0 - 0.1 * (K + 1)) < 0.6                                # the next one violates


def test_safe_prefix_never_zero_and_caps_at_kmax():
    px = np.zeros(C.N); py = np.zeros(C.N)
    assert ma.safe_prefix_length(px, py, [(px, py)], 0.6, C.N) == 1      # overlap -> still >= 1
    far = np.full(C.N, 10.0)
    assert ma.safe_prefix_length(px, py, [(far, py)], 0.6, C.K_SEND) == C.K_SEND  # all clear -> cap


def test_safe_prefix_checks_all_others():
    px_i = np.array([0.1 * (m + 1) for m in range(C.N)]); py_i = np.zeros(C.N)
    far = np.full(C.N, 10.0); near = np.full(C.N, 0.5)
    K_far = ma.safe_prefix_length(px_i, py_i, [(far, py_i)], 0.6, C.N)
    K_both = ma.safe_prefix_length(px_i, py_i, [(far, py_i), (near, py_i)], 0.6, C.N)
    assert K_both < K_far                                                # nearest neighbour wins
