"""Stage 1 gate -- single-dog node QP closed-loop verification.

Drives the node QP in closed loop against the true world-frame double integrator
(X_{t+1} = Ad X + Bd a_0*), a straight reference, obstacle(s) and a wall.
Asserts the stage-1 gate per scenario, then writes a trajectory PNG for each.

The BASELINE scenario is the original stage-1 gate (one off-line obstacle + one
wall, PNG stage1_traj.png) -- unchanged, still reproducible. Three variants are
added as an OVER-FITTING check: the node QP must stay safe on geometries it was
never tuned against, not just the single baseline.

  var1 off-center  obstacle offset to the far side -> avoid the OTHER way
  var2 two-obstacle slalom -> both barriers active (controller scales n_feat)
  var3 start hugging an obstacle -> hard CBF from a tight initial condition

Gate (per scenario, must all hold):
  1. reaches the goal:                 ||p_final - goal|| < GOAL_TOL
  2. avoids every obstacle + wall:     realized min h over ALL features >= 0
  3. QP solved every cycle
Predicted-horizon min h is DIAGNOSTIC only (soft far-steps, single
linearization) -- measured & reported, never asserted (spec B9).

Breaches are reported honestly with the actual min h -- geometry is NEVER
re-tuned to hide a dip; a variant that breaches is the signal we want.

Run inside the container (matplotlib works there; PNGs land in the shared
workspace): python3 legged_upper_control/admm/verify_stage1.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admm_impl import constants as C  # noqa: E402
from admm_impl import nq  # noqa: E402
from admm_impl import build_reference  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress"))

ROBOT_MARGIN = 0.30
MAX_CYCLES = 400
GOAL_TOL = 0.15

# straight y=0 wall reused by every scenario (safe line at y = -0.70)
_WALL = {"normal": (0.0, 1.0), "point": (0.0, -1.0), "d_safe": 0.30}

# baseline first (keeps stage1_traj.png identical), then the 3 variants.
SCENARIOS = [
    dict(name="baseline", png="stage1_traj.png",
         start=(0.0, 0.0), goal=(4.0, 0.0),
         obstacles=[{"pos": (2.0, 0.4), "radius": 0.35}],
         walls=[_WALL]),
    # var1: obstacle offset to the FAR side (occupies y in [-1.2, +0.1] at x=2)
    #       -> reference line is inside it, robot must deflect UP (opposite side).
    dict(name="var1_offcenter", png="stage1_var1_offcenter.png",
         start=(0.0, 0.0), goal=(4.0, 0.0),
         obstacles=[{"pos": (2.0, -0.55), "radius": 0.35}],
         walls=[_WALL]),
    # var2: slalom -> go below A (y<-0.15) then above B (y>0.15); both active.
    dict(name="var2_two_obs", png="stage1_var2_two_obs.png",
         start=(0.0, 0.0), goal=(4.0, 0.0),
         obstacles=[{"pos": (1.6, 0.45), "radius": 0.30},
                    {"pos": (2.6, -0.45), "radius": 0.30}],
         walls=[_WALL]),
    # var3: obstacle right in front of the start; h0 ~= +0.057 (6cm outside
    #       margin) -> the k=0 hard CBF starts from a tight initial condition.
    #       SAFETY GATE: this is the cold-start-fix acceptance (feasible every
    #       cycle + realized h>=0). It does NOT reach the goal: a near-head-on
    #       obstacle + a straight reference is a CBF deadlock, and the geometry
    #       that triggers cold-start infeasibility (margin reaching back to
    #       within ~2 steps of the start with h0>=0) is *necessarily* near
    #       head-on -- the two can't be separated. Goal-reach is proven
    #       recoverable by var3b (a bent reference), so we do NOT bolt on a
    #       reactive deadlock-breaker here (spec_updates_v2 C: removal list).
    dict(name="var3_near_start", png="stage1_var3_near_start.png", gate="safety",
         start=(0.0, 0.0), goal=(4.0, 0.0),
         obstacles=[{"pos": (0.7, -0.1), "radius": 0.35}],
         walls=[_WALL]),
    # var3b: SAME head-on obstacle, but a bent reference that routes over the
    #        top of the margin (what A* + inflation would hand us in stage 3).
    #        Proves the var3 deadlock is a straight-reference artifact, not a
    #        controller limitation: with a curved reference the deadlock is gone
    #        and the goal is reached with realized h>=0. Full gate.
    dict(name="var3b_bent_ref", png="stage1_var3b_bent_ref.png",
         start=(0.0, 0.0), goal=(4.0, 0.0),
         obstacles=[{"pos": (0.7, -0.1), "radius": 0.35}],
         walls=[_WALL],
         waypoints=[(0.0, 0.0), (0.25, 0.55), (0.7, 0.8), (1.15, 0.6),
                    (1.5, 0.25), (2.0, 0.0), (4.0, 0.0)]),
]


def shift_xi(xi, n):
    """Operating point for next cycle: previous solution shifted one step."""
    out = np.zeros(C.xi_dim(n))
    for k in range(1, n + 1):
        src = min(k + 1, n)                          # x_k <- prev x_{k+1}
        out[C.x_index(k, n):C.x_index(k, n) + 4] = \
            xi[C.x_index(src, n):C.x_index(src, n) + 4]
    for k in range(n):
        src = min(k + 1, n - 1)                      # a_k <- prev a_{k+1}
        out[C.a_index(k, n):C.a_index(k, n) + 2] = \
            xi[C.a_index(src, n):C.a_index(src, n) + 2]
    return out


# ---- realized barrier over ALL features of a scenario (the gate quantity) ----
def _min_h_obs(p, sc):
    return min((nq.h_obstacle(p, o["pos"], o["radius"] + ROBOT_MARGIN)
                for o in sc["obstacles"]), default=np.inf)


def _min_h_wall(p, sc):
    return min((nq.h_wall(p, w["normal"], w["point"], w["d_safe"])
                for w in sc["walls"]), default=np.inf)


def _min_h(p, sc):
    return min(_min_h_obs(p, sc), _min_h_wall(p, sc))


def run_scenario(sc):
    """Closed-loop the node QP over one scenario. Returns a result dict; never
    raises on a QP failure/breach -- records it so every scenario still runs."""
    start = np.array(sc["start"], float)
    goal = np.array(sc["goal"], float)
    node = nq.NodeSubproblem(obstacles=sc["obstacles"], walls=sc["walls"],
                             robot_margin=ROBOT_MARGIN,
                             q_pos=10.0, q_v=1.0, r_accel=0.5, w_pred=20.0)
    x = np.array([start[0], start[1], 0.0, 0.0])     # start at rest
    prev_xi = None
    traj = [x[:2].copy()]
    h_obs_hist, h_wall_hist, pred_min_hist = [], [], []
    reached = False
    fail_msg = None
    cycle = -1
    waypoints = sc.get("waypoints", [start, goal])   # bent path for var3b

    for cycle in range(MAX_CYCLES):
        x_des = build_reference(x[:2], waypoints)
        res = node.solve(x, x_des, xbar=prev_xi)
        if not res["status"].startswith("solved") or res["a0"] is None:
            fail_msg = f"cycle {cycle}: QP status={res['status']!r}"   # gate 3
            break

        # diagnostic: min predicted h over the horizon (soft far-steps may dip)
        pred_min_hist.append(min(_min_h(p, sc) for p in res["x_pred"][:, :2]))
        # realized barrier at the current state (gate 2)
        h_obs_hist.append(_min_h_obs(x[:2], sc))
        h_wall_hist.append(_min_h_wall(x[:2], sc))

        x = C.Ad @ x + C.Bd @ res["a0"]              # advance the true plant
        traj.append(x[:2].copy())
        prev_xi = shift_xi(res["xi"], node.N)

        if np.linalg.norm(x[:2] - goal) < GOAL_TOL:
            reached = True
            break

    h_obs = np.array(h_obs_hist) if h_obs_hist else np.array([np.inf])
    h_wall = np.array(h_wall_hist) if h_wall_hist else np.array([np.inf])
    pred = np.array(pred_min_hist) if pred_min_hist else np.array([np.inf])
    min_h_obs = float(h_obs.min())
    min_h_wall = float(h_wall.min())
    breach = (min_h_obs < -1e-6) or (min_h_wall < -1e-6)
    gate = sc.get("gate", "full")
    if gate == "safety":         # cold-start-fix acceptance: feasible + h>=0
        ok = (fail_msg is None) and not breach
    else:                        # full gate: also reach the goal
        ok = reached and not breach and fail_msg is None

    return dict(sc=sc, ok=ok, gate=gate, reached=reached, cycles=cycle + 1,
                fail_msg=fail_msg, breach=breach,
                min_h_obs=min_h_obs, min_h_wall=min_h_wall,
                pred_min=float(pred.min()),
                traj=np.array(traj), h_obs=h_obs, h_wall=h_wall, pred=pred)


def _plot(r):
    sc = r["sc"]
    start = np.array(sc["start"], float)
    goal = np.array(sc["goal"], float)
    traj = r["traj"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ref = np.array(sc.get("waypoints", [start, goal]), dtype=float)
    ax.plot(ref[:, 0], ref[:, 1], "b--", lw=1, label="reference")
    ax.plot(traj[:, 0], traj[:, 1], "k.-", lw=1.2, ms=3, label="realized")
    ax.plot(*start, "gd", ms=9, label="start")
    ax.plot(*goal, "r*", ms=13, label="goal")
    th = np.linspace(0, 2 * np.pi, 100)
    for i, o in enumerate(sc["obstacles"]):
        oc = np.array(o["pos"]); rad = o["radius"]; r_eff = rad + ROBOT_MARGIN
        ax.plot(oc[0] + rad * np.cos(th), oc[1] + rad * np.sin(th),
                color="orange", lw=2,
                label="obstacle" if i == 0 else None)
        ax.plot(oc[0] + r_eff * np.cos(th), oc[1] + r_eff * np.sin(th),
                color="orange", ls=":", lw=1,
                label="obstacle + margin" if i == 0 else None)
    for w in sc["walls"]:
        ax.axhline(w["point"][1] + w["d_safe"], color="gray", ls="-.",
                   lw=1, label="wall safe line")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    status = "OK" if r["ok"] else ("BREACH" if r["breach"] else "FAIL")
    if r["ok"] and not r["reached"]:
        status += ", safety gate (no goal)"
    ax.set_title(f"Stage 1 [{sc['name']}] node QP + HOCBF  ({status})")

    if np.isfinite(r["h_obs"]).any():
        t = np.arange(len(r["h_obs"])) * C.TS
        ax2.plot(t, r["h_obs"], label="realized min h_obstacle")
        ax2.plot(t, r["h_wall"], label="realized min h_wall")
        ax2.plot(t, r["pred"], color="0.6", ls=":", lw=1,
                 label="min predicted h (diagnostic, soft)")
        ax2.legend(fontsize=8)
    ax2.axhline(0.0, color="r", ls="--", lw=1)
    ax2.grid(True)
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("h")
    ax2.set_title("Barrier h(t): realized must stay >= 0")

    png = os.path.join(PNG_DIR, sc["png"])
    os.makedirs(PNG_DIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
    return png


def main():
    results = []
    for sc in SCENARIOS:
        r = run_scenario(sc)
        png = _plot(r)
        results.append(r)
        tag = "OK  " if r["ok"] else ("BREACH" if r["breach"] else "FAIL")
        reach = "reached" if r["reached"] else "NO-GOAL"
        line = (f"[{tag}] {sc['name']:<16} {r['gate']:<6} {reach} "
                f"cyc={r['cycles']:<4} realized min h_obs={r['min_h_obs']:+.4f} "
                f"min h_wall={r['min_h_wall']:+.4f} "
                f"| pred(diag)={r['pred_min']:+.4f}")
        if r["fail_msg"]:
            line += f"  <- {r['fail_msg']}"
        print(line)
        print(f"         wrote {png}")

    n_ok = sum(r["ok"] for r in results)
    print(f"\n[stage1] {n_ok}/{len(results)} scenarios green "
          f"(baseline, 3 variants, var3b bent-ref; var3 on the safety gate)")
    if n_ok != len(results):
        # honest failure: a breach/infeasibility is the signal, not a bug to hide
        sys.exit(1)


if __name__ == "__main__":
    main()
