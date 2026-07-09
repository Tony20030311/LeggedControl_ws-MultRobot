"""Stage 2 gate -- two-dog ADMM node-edge splitting, closed loop.

Two dogs walk head-on (small +-0.1 y offset to break the symmetric deadlock, the
var3 lesson) and must avoid each other via the inter-agent edge CBF. Each control
cycle runs the full ADMM (node->edge->dual, P_ITERS), takes a0^i, advances both
true plants X_{t+1}=Ad X+Bd a0, repeats.

Gate (the node-edge fingerprint; do NOT tune params to force it -- a bad curve is a
splitting bug to report, spec):
  1. realized inter-agent h >= 0 every cycle   (||p^1-p^2||^2 - D_MIN^2, no collision)
  2. r_prim descends over ADMM iterations       (iron proof node-edge is correct)
  3. r_dual descends too                         (r_prim alone is not enough, 文件一 6.2)
  4. h2_viol small after convergence             (DIAGNOSTIC only, see caveat below)
  5. both dogs reach their goals

h2_viol CAVEAT: h2_viol is the three-point HOCBF value; its coefficients sum to 0.09,
so it UNDERSTATES the realized barrier breach ~11x (h2_viol ~= 0.09*h). The safety
gate (1) uses the RAW realized barrier ||p1-p2||^2 - D_MIN^2 -- that is the truth;
h2_viol is only a convergence diagnostic.

Edge k=0 is HARD (k>=1 soft), mirroring the stage-1 node CBF: all-soft could not
produce realized avoidance (weak 3Ts^2 gradient, slack absorbs the violation), so the
dogs passed straight through. Hard k=0 forces the immediate avoidance and still
converges (z is unbounded -> always feasible). See docs/progress/stage2.md.

Writes a 2x2 figure: trajectories, realized h(t), r_prim/r_dual convergence,
h2_viol convergence (all cycles faint + the strongest-coupling cycle highlighted).

Run inside the container: python3 legged_upper_control/admm/verify_stage2.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admm_impl import constants as C  # noqa: E402
from admm_impl import ac  # noqa: E402
from admm_impl import reference as ref  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress",
                                    "stage2_residuals.png"))

START = {1: np.array([-3.0, 0.10]), 2: np.array([3.0, -0.10])}
GOAL = {1: np.array([3.0, 0.10]), 2: np.array([-3.0, -0.10])}
MAX_CYCLES = 300
GOAL_TOL = 0.15


def realized_h(p1, p2):
    e = np.asarray(p1) - np.asarray(p2)
    return float(e @ e) - C.D_MIN ** 2


def run():
    coord = ac.ADMMCoordinator()
    x = {i: np.array([START[i][0], START[i][1], 0.0, 0.0]) for i in (1, 2)}
    traj = {1: [x[1][:2].copy()], 2: [x[2][:2].copy()]}
    h_hist, cyc_hists = [], []
    reached = {1: False, 2: False}
    cyc = -1

    for cyc in range(MAX_CYCLES):
        xnow = {i: x[i] for i in (1, 2)}
        xdes = {i: ref.build_reference(x[i][:2], [START[i], GOAL[i]]) for i in (1, 2)}
        xi, hist = coord.step(xnow, xdes)
        cyc_hists.append(hist)
        h_hist.append(realized_h(x[1][:2], x[2][:2]))
        for i in (1, 2):
            a0 = xi[i][C.a_index(0, C.N):C.a_index(0, C.N) + 2]
            x[i] = C.Ad @ x[i] + C.Bd @ a0
            traj[i].append(x[i][:2].copy())
            if np.linalg.norm(x[i][:2] - GOAL[i]) < GOAL_TOL:
                reached[i] = True
        if reached[1] and reached[2]:
            break

    h_hist = np.array(h_hist)
    traj = {i: np.array(traj[i]) for i in (1, 2)}
    # strongest-coupling cycle = largest initial primal residual (closest approach)
    rep = int(np.argmax([h["r_prim"][0] for h in cyc_hists]))
    rp = np.array(cyc_hists[rep]["r_prim"])
    rd = np.array(cyc_hists[rep]["r_dual"])
    hv = np.array(cyc_hists[rep]["h2_viol"])

    reached_both = reached[1] and reached[2]
    min_h = float(h_hist.min())
    res = dict(cyc=cyc + 1, reached=reached_both, min_h=min_h, rep=rep,
               rp=rp, rd=rd, hv=hv, traj=traj, h_hist=h_hist, cyc_hists=cyc_hists)

    _plot(res)                                          # plot first (honest: always)

    # gate checks (report, never tune to force)
    checks = {
        "reach": reached_both,
        "safety(h>=0)": min_h >= -1e-6,
        "r_prim_descends": rp[-1] < rp[0] and rp[-1] < 0.3 * rp.max(),
        "r_dual_descends": rd[-1] < rd[0] and rd[-1] < 0.3 * rd.max(),
        "h2_viol_small": hv[-1] < 5e-2,
    }
    print(f"[stage2] cycles={cyc+1}  reached_both={reached_both}  "
          f"realized min inter-agent h={min_h:+.4f}")
    print(f"[stage2] rep cycle {rep}: r_prim {rp[0]:.3e}->{rp[-1]:.3e}  "
          f"r_dual {rd[0]:.3e}->{rd[-1]:.3e}  h2_viol {hv[0]:.3e}->{hv[-1]:.3e}")
    for k, v in checks.items():
        print(f"   [{'OK ' if v else 'FAIL'}] {k}")
    print(f"[stage2] wrote {PNG}")
    if not all(checks.values()):
        sys.exit(1)


def _plot(res):
    traj, h_hist = res["traj"], res["h_hist"]
    cyc_hists, rep = res["cyc_hists"], res["rep"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) trajectories
    a = ax[0, 0]
    a.plot(traj[1][:, 0], traj[1][:, 1], "b.-", ms=3, lw=1, label="dog1")
    a.plot(traj[2][:, 0], traj[2][:, 1], "r.-", ms=3, lw=1, label="dog2")
    for i, c in ((1, "b"), (2, "r")):
        a.plot(*START[i], c + "d", ms=8)
        a.plot(*GOAL[i], c + "*", ms=13)
    a.set_aspect("equal"); a.grid(True); a.legend(fontsize=8)
    a.set_title("Two-dog head-on: trajectories (avoid via edge CBF)")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    # (b) realized inter-agent barrier over the closed loop
    b = ax[0, 1]
    t = np.arange(len(h_hist)) * C.TS
    b.plot(t, h_hist, "k-", lw=1.5, label="realized ||p1-p2||^2 - D_MIN^2")
    b.axhline(0.0, color="r", ls="--", lw=1)
    b.grid(True); b.legend(fontsize=8)
    b.set_title(f"Realized inter-agent h(t): min={h_hist.min():+.4f} (>= 0 = safe)")
    b.set_xlabel("t [s]"); b.set_ylabel("h")

    # (c) primal & dual residual convergence (all cycles faint, rep highlighted)
    c = ax[1, 0]
    for h in cyc_hists:
        c.semilogy(h["r_prim"], color="0.8", lw=0.6)
        c.semilogy(h["r_dual"], color="0.9", lw=0.6)
    c.semilogy(cyc_hists[rep]["r_prim"], "b.-", lw=1.5, label="r_prim (rep cycle)")
    c.semilogy(cyc_hists[rep]["r_dual"], "g.-", lw=1.5, label="r_dual (rep cycle)")
    c.grid(True, which="both"); c.legend(fontsize=8)
    c.set_title("ADMM residuals vs iteration (both must descend)")
    c.set_xlabel("ADMM iteration"); c.set_ylabel("residual (log)")

    # (d) h2_viol convergence
    d = ax[1, 1]
    for h in cyc_hists:
        d.plot(h["h2_viol"], color="0.85", lw=0.6)
    d.plot(cyc_hists[rep]["h2_viol"], "m.-", lw=1.5, label="h2_viol (rep cycle)")
    d.axhline(0.0, color="r", ls="--", lw=1)
    d.grid(True); d.legend(fontsize=8)
    d.set_title("h2_viol vs iteration (-> ~0 after convergence)")
    d.set_xlabel("ADMM iteration"); d.set_ylabel("max(-Hbar)^+")

    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.tight_layout(); fig.savefig(PNG, dpi=110); plt.close(fig)


if __name__ == "__main__":
    run()
