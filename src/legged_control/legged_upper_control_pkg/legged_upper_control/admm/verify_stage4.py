"""Stage 4 gate -- ADMM ξ -> OCS2 target-trajectory adapter, on a real curved run.

Closed-loops the stage-1 node QP along a quarter-arc (a genuine turning trajectory,
so the yaw bypass actually sweeps), then feeds the result through MotionAdapter and
checks + plots the OCS2 mapping. The hard assertions live in test_motion_adapter.py;
this adds an integration pass on a real ξ plus a 3-panel figure for human inspection:

  (a) realized xy path + yaw quiver -> heading follows the motion direction
  (b) yaw(t): continuous EMA bypass vs raw atan2(vy,vx) -> smoothing (+ freeze)
  (c) a ±pi crossing demo -> the wrapped-delta EMA stays continuous, no 2*pi snap

Run inside the container (osqp 0.6.x; matplotlib):
  python3 legged_upper_control/admm/verify_stage4.py
"""

import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from admm_impl import constants as C  # noqa: E402
from admm_impl import nq  # noqa: E402
from admm_impl import ma  # noqa: E402
from admm_impl import build_reference  # noqa: E402

PNG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress",
                                    "stage4_adapter.png"))
# adapter config (placeholders; at wiring these load from reference.info:
# comHeight / defaultJointState). A1-ish standing pose, 12 joints.
COM_HEIGHT = 0.30
A1_DEFAULT_JOINTS = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                              0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
MAX_CYCLES = 90
GOAL_TOL = 0.15


def _arc_waypoints():
    """Quarter arc: starts at (0,0) heading +x, ends at (0.9,0.9) heading +y."""
    cx, cy, R = 0.0, 0.9, 0.9
    th = np.linspace(-np.pi / 2, 0.0, 12)
    return [(cx + R * math.cos(t), cy + R * math.sin(t)) for t in th]


def _run_arc():
    """Closed-loop the node QP along the arc; collect realized [px,py,vx,vy] and the
    last-cycle ξ (for the horizon-assembly integration check)."""
    wps = _arc_waypoints()
    goal = np.array(wps[-1])
    node = nq.NodeSubproblem(obstacles=[], walls=[])
    x = np.array([wps[0][0], wps[0][1], 0.0, 0.0])
    prev_xi, last_xi = None, None
    px, py, vx, vy = [], [], [], []
    for _ in range(MAX_CYCLES):
        x_des = build_reference(x[:2], wps)
        res = node.solve(x, x_des, xbar=prev_xi)
        if not res["status"].startswith("solved") or res["a0"] is None:
            break
        last_xi = res["xi"]
        px.append(x[0]); py.append(x[1]); vx.append(x[2]); vy.append(x[3])
        x = C.Ad @ x + C.Bd @ res["a0"]
        prev_xi = _shift(res["xi"], node.N)
        if np.linalg.norm(x[:2] - goal) < GOAL_TOL:
            break
    return (np.array(px), np.array(py), np.array(vx), np.array(vy), last_xi)


def _shift(xi, n):
    out = np.zeros(C.xi_dim(n))
    for k in range(1, n + 1):
        s = min(k + 1, n)
        out[C.x_index(k, n):C.x_index(k, n) + 4] = xi[C.x_index(s, n):C.x_index(s, n) + 4]
    for k in range(n):
        s = min(k + 1, n - 1)
        out[C.a_index(k, n):C.a_index(k, n) + 2] = xi[C.a_index(s, n):C.a_index(s, n) + 2]
    return out


def run():
    px, py, vx, vy, last_xi = _run_arc()
    adap = ma.MotionAdapter(com_height=COM_HEIGHT, default_joints=A1_DEFAULT_JOINTS)

    # yaw bypass over the whole realized run (for the plot)
    yaw = ma.yaw_trajectory(vx, vy, seed_yaw=0.0,
                            v_freeze=adap.v_freeze, ema_alpha=adap.ema_alpha)
    raw = np.array([math.atan2(vy[k], vx[k]) if math.hypot(vx[k], vy[k]) > adap.v_freeze
                    else np.nan for k in range(len(vx))])

    # integration check on a real horizon ξ (mirror of the unit asserts).
    # build_target sends only the first k_send (=C.K_SEND) steps — the safe near
    # horizon; the unsafe far tail is dropped (dynamic safe-prefix design).
    out = adap.build_target(last_xi, t0=0.0, seed_yaw=float(yaw[-1]))
    N = adap.N
    ns = adap.k_send
    checks = {
        "states_shape": out["states"].shape == (ns, ma.OCS2_STATE_DIM),
        "times_monotonic": bool(np.all(np.diff(out["times"]) > 0)),
        "inputs_zero": bool(np.all(out["inputs"] == 0.0)),
        "px_at_idx6": bool(np.allclose(out["states"][:, 6],
                                       [last_xi[C.px_index(k, N)] for k in range(1, ns + 1)])),
        "z_forced_comheight": bool(np.allclose(out["states"][:, 8], COM_HEIGHT)),
        "angmom_zero(idx3-5)": bool(np.all(out["states"][:, 3:6] == 0.0)),
        "pitch_roll_zero": bool(np.all(out["states"][:, 10:12] == 0.0)),
    }

    _plot(px, py, vx, vy, yaw, raw)

    print(f"[stage4] arc run: {len(px)} cycles, yaw {yaw[0]*57.3:+.1f} -> "
          f"{yaw[-1]*57.3:+.1f} deg (sweeps with the turn)")
    for k, v in checks.items():
        print(f"   [{'OK ' if v else 'FAIL'}] {k}")
    print(f"[stage4] wrote {PNG}")
    if not all(checks.values()):
        sys.exit(1)
    print("[stage4] GREEN")


def _plot(px, py, vx, vy, yaw, raw):
    t = np.arange(len(px)) * C.TS
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (a) path + yaw quiver
    a = ax[0]
    a.plot(px, py, "b.-", ms=3, lw=1, label="realized path (idx6,7)")
    step = max(1, len(px) // 12)
    a.quiver(px[::step], py[::step], np.cos(yaw[::step]), np.sin(yaw[::step]),
             color="r", scale=18, width=0.005, label="yaw (idx9)")
    a.set_aspect("equal"); a.grid(True); a.legend(fontsize=8)
    a.set_title("Path + yaw arrows (head follows motion)")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    # (b) yaw EMA vs raw
    b = ax[1]
    b.plot(t, np.rad2deg(yaw), "b.-", ms=3, lw=1.2, label="yaw bypass (EMA, continuous)")
    b.plot(t, np.rad2deg(raw), "0.6", ls=":", lw=1, label="raw atan2(vy,vx)")
    b.grid(True); b.legend(fontsize=8)
    b.set_title("yaw(t): EMA smoothing (freeze where |v|<thresh)")
    b.set_xlabel("t [s]"); b.set_ylabel("yaw [deg]")

    # (c) +-pi crossing continuity demo
    c = ax[2]
    angs = np.deg2rad(np.linspace(150.0, 210.0, 40))
    dvx, dvy = np.cos(angs), np.sin(angs)
    ys = ma.yaw_trajectory(dvx, dvy, seed_yaw=angs[0], v_freeze=0.05, ema_alpha=0.5)
    rawc = np.rad2deg([math.atan2(dvy[k], dvx[k]) for k in range(len(angs))])
    c.plot(np.rad2deg(ys), "b.-", ms=3, lw=1.2, label="EMA (continuous, crosses 180)")
    c.plot(rawc, "0.6", ls=":", lw=1, label="raw atan2 (wraps -> jumps)")
    c.axhline(180, color="r", ls="--", lw=1)
    c.grid(True); c.legend(fontsize=8)
    c.set_title("±180 crossing: wrapped-delta EMA stays continuous")
    c.set_xlabel("step"); c.set_ylabel("yaw [deg]")

    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.tight_layout(); fig.savefig(PNG, dpi=110); plt.close(fig)


if __name__ == "__main__":
    run()
