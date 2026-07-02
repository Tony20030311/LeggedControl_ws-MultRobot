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
import constants as C          # noqa: E402
import motion_adapter as ma    # noqa: E402

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


# ---- full assembly from a real ξ ---------------------------------------------
def test_build_target_shapes_times_inputs():
    N = C.N
    adap = ma.MotionAdapter(com_height=0.42, default_joints=DEFJ, input_dim=24)
    out = adap.build_target(_straight_xi(), t0=1.0, seed_yaw=0.0)
    assert out["times"].shape == (N,)
    assert np.all(np.diff(out["times"]) > 0)
    assert abs(out["times"][0] - (1.0 + C.TS)) < 1e-9
    assert out["states"].shape == (N, ma.OCS2_STATE_DIM)
    assert out["inputs"].shape == (N, 24) and np.all(out["inputs"] == 0.0)


def test_build_target_places_admm_state():
    N = C.N
    xi = _straight_xi()
    adap = ma.MotionAdapter(com_height=0.42, default_joints=DEFJ)
    out = adap.build_target(xi, t0=0.0, seed_yaw=0.0)
    for j in range(N):
        k = j + 1
        assert out["states"][j, 6] == xi[C.px_index(k, N)]     # px -> idx6
        assert out["states"][j, 7] == xi[C.py_index(k, N)]     # py -> idx7
        assert out["states"][j, 0] == xi[C.vx_index(k, N)]     # vx -> idx0
        assert out["states"][j, 8] == 0.42                      # z forced
        assert out["states"][j, 3] == 0.0                       # angular mom stays 0
