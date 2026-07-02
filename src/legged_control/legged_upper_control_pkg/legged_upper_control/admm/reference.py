"""Stage 1 reference: straight polyline -> arc-length anchors (x_des).

The node QP tracks states x_1..x_N to N position anchors sampled at (near-)
constant arc length along a waypoint polyline. Stage 1 feeds a single straight
segment [start, goal] aimed to pass *near* the obstacle so the node CBF has to
deflect it. Stages 3/4 swap in real A* waypoints -- same sampler, different path.

Position-only reference (spec B6/B7): velocity dims are 0. Velocity emerges from
tracking position + accel regularization + dynamics, so it stays smooth and
zeroes out at the goal without a feedback-law v_des.

Deceleration lives in the reference (spec B5): anchor spacing = v_ref * Ts with
v_ref = v_cruise * min(1, d_remain / d_brake) and d_remain the *remaining arc
length* to the final waypoint (not straight-line distance).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C  # noqa: E402


def _cumulative(waypoints):
    wp = np.asarray(waypoints, dtype=float)[:, :2]
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return wp, cum


def point_at_arclength(wp, cum, s):
    """Point on the polyline at arc length s (clamped to [0, L])."""
    L = cum[-1]
    s = min(max(s, 0.0), L)
    i = int(np.searchsorted(cum, s, side="right")) - 1
    i = min(max(i, 0), len(wp) - 2)
    seg_len = cum[i + 1] - cum[i]
    t = 0.0 if seg_len < 1e-12 else (s - cum[i]) / seg_len
    return wp[i] + t * (wp[i + 1] - wp[i])


def _project(pos, wp, cum):
    """Arc length of the closest point on the polyline to pos."""
    pos = np.asarray(pos, dtype=float)[:2]
    best_s, best_d2 = 0.0, float("inf")
    for i in range(len(wp) - 1):
        a, b = wp[i], wp[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float((pos - a) @ ab) / L2
        t = min(max(t, 0.0), 1.0)
        proj = a + t * ab
        d2 = float((pos - proj) @ (pos - proj))
        if d2 < best_d2:
            best_d2, best_s = d2, cum[i] + t * np.sqrt(L2)
    return best_s


def build_reference(cur_pos, waypoints, n=None, ts=None,
                    v_cruise=0.30, d_brake=0.80):
    """N position anchors ahead of cur_pos along the polyline.

    Returns x_des of shape (N, 4) = [px, py, 0, 0] targets for states x_1..x_N.
    """
    n = C.N if n is None else n
    ts = C.TS if ts is None else ts
    wp, cum = _cumulative(waypoints)
    L = cum[-1]
    s = _project(cur_pos, wp, cum)
    x_des = np.zeros((n, 4))
    for k in range(n):
        d_remain = max(0.0, L - s)
        v_ref = v_cruise * min(1.0, d_remain / d_brake) if d_brake > 1e-9 else v_cruise
        s = min(L, s + v_ref * ts)
        x_des[k, :2] = point_at_arclength(wp, cum, s)
    return x_des


if __name__ == "__main__":
    # start far from goal -> full cruise; spacing == v_cruise * Ts, y stays 0.
    xd = build_reference([0.0, 0.0], [[0.0, 0.0], [4.0, 0.0]])
    assert xd.shape == (C.N, 4)
    assert np.allclose(xd[:, 2:], 0.0), "velocity dims must be 0 (B7)"
    assert np.all(np.diff(xd[:, 0]) >= -1e-9), "anchors monotonic along path"
    assert abs((xd[1, 0] - xd[0, 0]) - 0.30 * C.TS) < 1e-9, "cruise spacing"
    assert np.allclose(xd[:, 1], 0.0), "straight line stays on y=0"
    # near goal -> anchors clamp at goal and decelerate (spacing shrinks).
    xd2 = build_reference([3.7, 0.0], [[0.0, 0.0], [4.0, 0.0]])
    assert xd2[-1, 0] <= 4.0 + 1e-9 and xd2[-1, 0] > 3.7
    assert (xd2[-1, 0] - xd2[-2, 0]) < 0.30 * C.TS + 1e-9, "decel near goal"
    print("reference.py self-check OK")
