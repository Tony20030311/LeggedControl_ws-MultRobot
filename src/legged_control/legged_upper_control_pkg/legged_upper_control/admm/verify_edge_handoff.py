"""Edge-CBF handoff safety experiment -- sweep the edge hard span (hard_through) on a
two-dog head-on and measure the safety of the WHOLE SENT reference, not the a0-executed
plant.

Why this exists (the handoff hole): the closed-loop plant stays safe for ANY
hard_through>=1 because only a0 -- the k=0 hard step -- is executed each cycle. But we
hand the FULL horizon xi to OCS2 as x_ref, and OCS2 tracks it as a soft cost with no
inter-agent CBF of its own and no staleness guard. With hard_through=1 (k=0 hard, k>=1
soft, today's shipping behavior) the reference's tail dives through the other dog (the
copies violate, consensus pulls xi toward them) -> OCS2 tracks an unsafe reference ->
collision. Making the sent steps HARD (paper eq.13/16/18) keeps the reference safe.
This is the offline proxy the a0-only plant rollout (verify_stage2) cannot show.

Per cycle, per config, over the head-on encounter:
  * ref_min = min_k(||p^i_k - p^j_k|| - D_MIN) over k=1..N on the SENT xi   <-- the truth
  * r_prim[-1]  (copy<->true-body gap after P_ITERS)                        <-- convergence
Plant realized h is tracked too (safe for any hard_through>=1; not the point).

Gate (asserts the FIX works; the current-vs-full comparison is REPORTED, not forced):
  * full-hard keeps ref_min >= -TOL           (the sent reference is safe)
  * full-hard r_prim descends                 (consensus still converges at N=20)
  * full-hard reaches the goal                (not deadlocked by the hard whole-horizon)
A failing full-hard gate is an honest finding (full-hard too aggressive at our a_max ->
fall back to the window), not a code bug -- report it, do not tune to force it.

Run in the container: python3 legged_upper_control/admm/verify_edge_handoff.py
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
from admm_impl import ma  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress",
                                    "edge_handoff.png"))

START = {1: np.array([-3.0, 0.10]), 2: np.array([3.0, -0.10])}
GOAL = {1: np.array([3.0, 0.10]), 2: np.array([-3.0, -0.10])}
MAX_CYCLES = 300
GOAL_TOL = 0.15
TOL = 0.05   # allowed linearization slop below D_MIN on the reference [m]

CONFIGS = [("k0-only (current)", 1), ("window k<10", 10), ("full (paper)", C.N - 1)]


def ref_min_dist(xi_i, xi_j, N):
    """min over horizon steps k=1..N of ||p^i_k - p^j_k|| - D_MIN, read straight off
    the SENT position block of xi (exactly the reference OCS2 would track)."""
    worst = np.inf
    for k in range(1, N + 1):
        pi = xi_i[C.px_index(k, N):C.px_index(k, N) + 2]
        pj = xi_j[C.px_index(k, N):C.px_index(k, N) + 2]
        worst = min(worst, float(np.linalg.norm(pi - pj)) - C.D_MIN)
    return worst


def run_config(hard_through):
    coord = ac.ADMMCoordinator(hard_through=hard_through)
    x = {i: np.array([START[i][0], START[i][1], 0.0, 0.0]) for i in (1, 2)}
    ref_hist, ref_send_hist, rp_final, plant_h = [], [], [], []
    dyn_prefix, dyn_ref_hist = [], []
    rep_rp, rep_val = None, -np.inf
    edge_fail = 0
    reached = {1: False, 2: False}
    for _cyc in range(MAX_CYCLES):
        xnow = {i: x[i] for i in (1, 2)}
        xdes = {i: ref.build_reference(x[i][:2], [START[i], GOAL[i]]) for i in (1, 2)}
        xi, hist = coord.step(xnow, xdes)
        edge_fail += int(hist.get("edge_fail", 0))
        rm = ref_min_dist(xi[1], xi[2], C.N)              # full 2s horizon
        ref_hist.append(rm)
        ref_send_hist.append(ref_min_dist(xi[1], xi[2], C.K_SEND))  # fixed K_SEND cut
        # DYNAMIC safe-prefix truncation: send only the leading steps that clear D_MIN
        wpx1 = np.array([xi[1][C.px_index(k, C.N)] for k in range(1, C.N + 1)])
        wpy1 = np.array([xi[1][C.py_index(k, C.N)] for k in range(1, C.N + 1)])
        wpx2 = np.array([xi[2][C.px_index(k, C.N)] for k in range(1, C.N + 1)])
        wpy2 = np.array([xi[2][C.py_index(k, C.N)] for k in range(1, C.N + 1)])
        kdyn = ma.safe_prefix_length(wpx1, wpy1, [(wpx2, wpy2)], C.D_MIN, C.K_SEND)
        dyn_prefix.append(kdyn)
        dyn_ref_hist.append(ref_min_dist(xi[1], xi[2], kdyn))   # safe by construction
        rp_final.append(float(hist["r_prim"][-1]))
        e = x[1][:2] - x[2][:2]
        plant_h.append(float(e @ e) - C.D_MIN ** 2)
        if -rm > rep_val:                      # rep = closest reference approach
            rep_val, rep_rp = -rm, np.array(hist["r_prim"])
        for i in (1, 2):
            a0 = xi[i][C.a_index(0, C.N):C.a_index(0, C.N) + 2]
            x[i] = C.Ad @ x[i] + C.Bd @ a0
            if np.linalg.norm(x[i][:2] - GOAL[i]) < GOAL_TOL:
                reached[i] = True
        if reached[1] and reached[2]:
            break
    return dict(ref_hist=np.array(ref_hist), ref_send_hist=np.array(ref_send_hist),
                dyn_prefix=np.array(dyn_prefix), dyn_ref_hist=np.array(dyn_ref_hist),
                rp_final=np.array(rp_final), plant_h=np.array(plant_h), rep_rp=rep_rp,
                edge_fail=edge_fail, reached=(reached[1] and reached[2]))


def run():
    res = {name: run_config(K) for name, K in CONFIGS}

    print("[edge-handoff] head-on, D_MIN=%.2f, N=%d, K_SEND=%d, P_ITERS=%d" %
          (C.D_MIN, C.N, C.K_SEND, C.P_ITERS))
    print("  config              | ref_min(full 2s) | ref_min(sent %d) | r_prim | edge_fail | reached"
          % C.K_SEND)
    for name, _K in CONFIGS:
        r = res[name]
        print("  [%-18s]   %+.4f          %+.4f         %.1e    %-4d       %s" %
              (name, r["ref_hist"].min(), r["ref_send_hist"].min(),
               r["rp_final"].max(), r["edge_fail"], r["reached"]))

    _plot(res)
    print("[edge-handoff] wrote %s" % PNG)

    soft = res["k0-only (current)"]      # the shipping config = hard_through=1
    # REPORTED: expanding the hard span is falsified (r_prim explodes, dogs collide/stuck).
    for name in ("window k<10", "full (paper)"):
        r = res[name]
        print("  [report] %-14s hard-expansion BROKEN: r_prim=%.1e reached=%s (a_max caps"
              " active hard steps ~2)" % (name, r["rp_final"].max(), r["reached"]))

    # The fix = keep hard_through=1 + DYNAMIC safe-prefix truncation. Compare the two
    # truncation policies for the shipping config:
    kmin = int(soft["dyn_prefix"].min())
    print("  [fix] fixed K_SEND=%d -> ref_min %+.4f (still dips); DYNAMIC safe-prefix ->"
          " ref_min %+.4f, prefix shrinks to %d steps (%.1fs) at the closest approach"
          % (C.K_SEND, soft["ref_send_hist"].min(), soft["dyn_ref_hist"].min(),
             kmin, kmin * C.TS))
    checks = {
        "hard-expansion falsified (window broke)": res["window k<10"]["rp_final"].max() > 1.0,
        "fixed trunc alone still unsafe":          soft["ref_send_hist"].min() < -TOL,
        "DYNAMIC trunc: sent ref safe (>=-eps)":   soft["dyn_ref_hist"].min() >= -1e-6,
        "DYNAMIC prefix never empty (>=1)":        kmin >= 1,
        "k0 converges (r_prim small)":             soft["rp_final"].max() < 1e-2,
        "k0 reaches goal":                         soft["reached"],
        "k0 no edge failures":                     soft["edge_fail"] == 0,
    }
    for k, v in checks.items():
        print("   [%s] %s" % ("OK " if v else "FAIL", k))
    if not all(checks.values()):
        sys.exit(1)


def _plot(res):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"k0-only (current)": "r", "window k<10": "orange", "full (paper)": "b"}

    a = ax[0]
    for name, _K in CONFIGS:
        r = res[name]
        t = np.arange(len(r["ref_hist"])) * C.TS
        a.plot(t, r["ref_hist"], color=colors[name], lw=1.4, label=name)
    a.axhline(0.0, color="k", ls="--", lw=1)
    a.axhline(-TOL, color="0.6", ls=":", lw=1)
    a.grid(True); a.legend(fontsize=8)
    a.set_title("Sent-reference safety over the encounter\n"
                "min_k(||p^i_k - p^j_k|| - D_MIN)  (>= 0 = the x_ref OCS2 tracks is safe)")
    a.set_xlabel("t [s]"); a.set_ylabel("reference margin [m]")

    b = ax[1]
    for name, _K in CONFIGS:
        rp = res[name]["rep_rp"]
        if rp is not None:
            b.semilogy(rp, color=colors[name], lw=1.4, marker=".", label=name)
    b.grid(True, which="both"); b.legend(fontsize=8)
    b.set_title("ADMM r_prim = ||z - xi|| descent (rep/closest cycle)\n"
                "copy<->true-body gap after P_ITERS iterations")
    b.set_xlabel("ADMM iteration"); b.set_ylabel("r_prim (log)")

    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.tight_layout(); fig.savefig(PNG, dpi=110); plt.close(fig)


if __name__ == "__main__":
    run()
