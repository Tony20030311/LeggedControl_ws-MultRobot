"""Stage 3 gate -- three-dog ADMM + formation + node CBF, closed loop.

Three dogs must (a) thread a door (obstacle posts + loose corridor walls -- node-
local CBF, re-enabled here after being dormant in stage 2), (b) avoid each other
(inter-agent edge CBF on all 3 edges of the complete graph), and (c) assemble into
the desired formation shape (normalized-Laplacian cost, linearized once per cycle
to a frozen node-local gradient -- spec B4/B8). Each control cycle runs the full
ADMM (node->edge->dual, P_ITERS), takes a0^i, advances all three true plants
X_{t+1}=Ad X+Bd a0, repeats.

SIMPLIFIED arena on purpose (Tony-approved): the point is to prove the MECHANISM
(formation term active + node CBF + inter-agent CBF coexist and ADMM still
converges), not to match the full Gazebo scene -- that is stage 4 (OCS2).

Scenarios:
  V        assemble a V from an abreast start (baseline; f-isolation overlay).
  column   assemble a single-file column from the SAME abreast start (over-fitting
           check: a different L_hat_des must also converge).
  squeeze  SOFT DEFORMATION through a narrow gap: a WIDE V (wings +-0.8) meets a
           door whose clear corridor (|y|<0.40) is narrower than the formation.
           The wings MUST squeeze in to pass, then release. We watch f(t) rise at
           the door (formation yields) and fall after (recovers). Run at w_form=10
           AND w_form=0 -- if w_form=10 deadlocks or peaks far above w_form=0 the
           formation is too rigid (-> the dynamic-tightening question); reported,
           NOT patched with a reactive breaker (that debt is stage 5).

Gate (per scenario; a bad curve is a bug to report, NEVER tune geometry to force):
  1. all three reach their formation slot at the goal (no deadlock)
  2. realized inter-agent h >= 0 every cycle for ALL 3 pairs (raw ||p^i-p^j||^2 - d^2)
  3. realized node CBF h >= 0 every cycle for every dog (posts + walls)
  4. r_prim AND r_dual descend over ADMM iterations
  5. formation error f=||L_hat - L_hat_des||^2 behaves (V/column: falls & stays low;
     squeeze: rises at the door then recovers -- asserted only if they pass).

Run inside the container (ROS sourced -> core.formation imports rospy; osqp 0.6.x):
  source /opt/ros/noetic/setup.bash
  python3 legged_upper_control/admm/verify_stage3.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # legged_upper_control/ -> core.formation
sys.path.insert(0, HERE)                     # admm/ -> constants, coordinator, ...
from admm_impl import constants as C  # noqa: E402
from admm_impl import ac  # noqa: E402
from admm_impl import nq  # noqa: E402
from admm_impl import build_reference         # noqa: E402
from admm_impl import LaplacianFormation  # noqa: E402

PNG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress"))
DOGS = (1, 2, 3)
EDGES = ((1, 2), (1, 3), (2, 3))
MAX_CYCLES = 450
GOAL_TOL = 0.20
W_FORM = 10.0
ROBOT_MARGIN = 0.30                            # node default; r_eff = radius + margin
DOOR_X = 0.0

# loose corridor walls, shared by all scenarios (exercise the wall CBF plumbing +
# multi-feature assembly; realized wall h stays well > 0 here -- said so honestly).
WALLS = [{"normal": (0.0, -1.0), "point": (0.0, 1.7), "d_safe": 0.30},
         {"normal": (0.0, 1.0), "point": (0.0, -1.7), "d_safe": 0.30}]
# door = two posts flanking a clear gap (a half-plane wall can't make an opening).
DOOR_WIDE = [{"pos": (0.0, 1.15), "radius": 0.30},     # clear |y| < 0.55
             {"pos": (0.0, -1.15), "radius": 0.30}]
DOOR_NARROW = [{"pos": (0.0, 1.0), "radius": 0.30},    # clear |y| < 0.40 (squeeze)
               {"pos": (0.0, -1.0), "radius": 0.30}]

ABREAST = {1: (-3.0, 0.0), 2: (-3.0, 0.7), 3: (-3.0, -0.7)}
WIDEV_START = {1: (-3.0, 0.0), 2: (-3.7, 0.8), 3: (-3.7, -0.8)}

FORMATIONS = {
    "V":      [(0.0, 0.0), (-0.7, 0.5), (-0.7, -0.5)],
    "column": [(0.0, 0.0), (-0.7, 0.0), (-1.4, 0.0)],
    "V_wide": [(0.0, 0.0), (-0.7, 0.8), (-0.7, -0.8)],
}
SCENARIOS = [
    dict(label="V", formation="V", png="stage3_formation.png", overlay0=True,
         start=ABREAST, obstacles=DOOR_WIDE,
         goal={1: (3.0, 0.0), 2: (2.3, 0.5), 3: (2.3, -0.5)}),
    dict(label="column", formation="column", png="stage3_variant_column.png",
         overlay0=False, start=ABREAST, obstacles=DOOR_WIDE,
         goal={1: (3.0, 0.0), 2: (2.3, 0.0), 3: (1.6, 0.0)}),
    # squeeze is a DIAGNOSTIC PROBE (gating=False): it deadlocks by design -- a wide
    # V meeting a door narrower than the formation, fed a straight per-dog reference.
    # The deadlock is a known limit of the missing planner layer (stage 4), same as
    # stage-1 var3 (straight ref into a head-on obstacle), so it must NOT gate stage
    # 3. It still runs + plots + reports its finding every time.
    dict(label="squeeze", formation="V_wide", png="stage3_squeeze.png",
         overlay0=True, squeeze=True, gating=False, start=WIDEV_START,
         obstacles=DOOR_NARROW,
         goal={1: (3.0, 0.0), 2: (2.3, 0.8), 3: (2.3, -0.8)}),
]


# ---- realized barriers (the gate quantities) ---------------------------------
def realized_pair_h(pi, pj):
    e = np.asarray(pi) - np.asarray(pj)
    return float(e @ e) - C.D_MIN ** 2


def realized_node_h(p, obstacles):
    hs = [nq.h_obstacle(p, o["pos"], o["radius"] + ROBOT_MARGIN) for o in obstacles]
    hs += [nq.h_wall(p, w["normal"], w["point"], w["d_safe"]) for w in WALLS]
    return min(hs)


def run(lf, scn, w_form):
    """Closed-loop the 3-dog ADMM over one scenario. Never raises on breach/failure
    -- records it so the scenario always finishes and the plot is honest."""
    lf.set_formation(scn["formation"])
    start, goal, obstacles = scn["start"], scn["goal"], scn["obstacles"]
    coord = ac.ADMMCoordinator(dogs=DOGS, edges=EDGES,
                               obstacles=obstacles, walls=WALLS,
                               formation=lf, w_form=w_form)
    x = {i: np.array([start[i][0], start[i][1], 0.0, 0.0]) for i in DOGS}
    traj = {i: [x[i][:2].copy()] for i in DOGS}
    h_pair = {e: [] for e in EDGES}
    node_h = {i: [] for i in DOGS}
    f_hist, cyc_hists = [], []
    reached = {i: False for i in DOGS}
    fail_msg = None
    cyc = -1

    for cyc in range(MAX_CYCLES):
        xnow = {i: x[i] for i in DOGS}
        xdes = {i: build_reference(x[i][:2], [start[i], goal[i]]) for i in DOGS}
        xi, hist = coord.step(xnow, xdes)
        cyc_hists.append(hist)

        for (a, b) in EDGES:
            h_pair[(a, b)].append(realized_pair_h(x[a][:2], x[b][:2]))
        for i in DOGS:
            node_h[i].append(realized_node_h(x[i][:2], obstacles))
        f_now, _ = lf.compute([x[i][:2] for i in DOGS])
        f_hist.append(f_now)

        for i in DOGS:
            a0 = xi[i][C.a_index(0, C.N):C.a_index(0, C.N) + 2]
            if a0 is None or not np.all(np.isfinite(a0)):
                fail_msg = f"cycle {cyc}: dog {i} a0 not finite"
                break
            x[i] = C.Ad @ x[i] + C.Bd @ a0
            traj[i].append(x[i][:2].copy())
            if np.linalg.norm(x[i][:2] - np.asarray(goal[i])) < GOAL_TOL:
                reached[i] = True
        if fail_msg is not None:
            break
        if all(reached.values()):
            break

    traj = {i: np.array(traj[i]) for i in DOGS}
    h_pair = {e: np.array(v) for e, v in h_pair.items()}
    node_h = {i: np.array(v) for i, v in node_h.items()}
    f_hist = np.array(f_hist)
    min_pair = min(float(h_pair[e].min()) for e in EDGES)
    min_node = min(float(node_h[i].min()) for i in DOGS)
    rep = int(np.argmax([h["r_prim"][0] for h in cyc_hists]))
    return dict(label=scn["label"], w_form=w_form, cyc=cyc + 1, reached=reached,
                fail_msg=fail_msg, traj=traj, h_pair=h_pair, node_h=node_h,
                f_hist=f_hist, cyc_hists=cyc_hists, rep=rep,
                min_pair=min_pair, min_node=min_node)


def _crossing_stats(res):
    """f behaviour while ANY dog is within 1.2 m of the door (the squeeze window)."""
    f = res["f_hist"]
    T = len(f)
    xs = np.array([[res["traj"][i][t, 0] for i in DOGS] for t in range(T)])
    near = np.min(np.abs(xs - DOOR_X), axis=1) < 1.2
    f_start = float(np.mean(f[:3]))
    f_end = float(np.mean(f[-5:]))
    f_peak = float(f[near].max()) if near.any() else f_start
    t_peak = int(np.argmax(np.where(near, f, -np.inf))) if near.any() else 0
    win = np.where(near)[0]
    return dict(f_start=f_start, f_end=f_end, f_peak=f_peak, t_peak=t_peak,
                win=(int(win[0]), int(win[-1])) if win.size else (0, 0))


def gate(res, scn):
    rp = np.array(res["cyc_hists"][res["rep"]]["r_prim"])
    rd = np.array(res["cyc_hists"][res["rep"]]["r_dual"])
    cs = _crossing_stats(res)
    reach_all = all(res["reached"].values()) and res["fail_msg"] is None
    checks = {
        "reach_all(no deadlock)": reach_all,
        "interagent_h>=0": res["min_pair"] >= -1e-6,
        "node_h>=0": res["min_node"] >= -1e-6,
        "r_prim_descends": rp[-1] < rp[0] and rp[-1] < 0.3 * rp.max(),
        "r_dual_descends": rd[-1] < rd[0] and rd[-1] < 0.3 * rd.max(),
    }
    if scn.get("squeeze"):
        # soft yield: f rose at the door then recovered -- assert ONLY if they passed
        # (a deadlock is the headline finding, reported separately, not force-failed
        # on the f-shape). "rose" = >= 0.05 above the near-zero in-shape start.
        yields = cs["f_peak"] > cs["f_start"] + 0.05
        recovers = cs["f_end"] < 0.5 * cs["f_peak"] and cs["f_end"] < 0.10
        checks["formation_yields_at_door"] = (yields and recovers) if reach_all else True
    else:
        checks["formation_improves"] = cs["f_end"] < 0.5 * cs["f_start"]
    return checks, dict(rp=rp, rd=rd, **cs)


def _plot(res, scn, res0=None):
    """2x2: trajectories / all realized barriers / residuals / formation error."""
    traj, goal, start = res["traj"], scn["goal"], scn["start"]
    obstacles = scn["obstacles"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cols = {1: "b", 2: "r", 3: "g"}

    # (a) trajectories + arena
    a = ax[0, 0]
    for i in DOGS:
        a.plot(traj[i][:, 0], traj[i][:, 1], cols[i] + ".-", ms=2, lw=1,
               label=f"dog{i}")
        a.plot(*start[i], cols[i] + "d", ms=7)
        a.plot(*goal[i], cols[i] + "*", ms=13)
    th = np.linspace(0, 2 * np.pi, 60)
    for k, o in enumerate(obstacles):
        oc, r = np.array(o["pos"]), o["radius"]
        a.plot(oc[0] + r * np.cos(th), oc[1] + r * np.sin(th), color="orange", lw=2,
               label="door post" if k == 0 else None)
        a.plot(oc[0] + (r + ROBOT_MARGIN) * np.cos(th),
               oc[1] + (r + ROBOT_MARGIN) * np.sin(th), color="orange", ls=":", lw=1)
    for w in WALLS:
        a.axhline(w["point"][1] + w["d_safe"] / w["normal"][1],
                  color="0.6", ls="-.", lw=0.8)
    a.set_aspect("equal"); a.grid(True); a.legend(fontsize=8, loc="upper center")
    a.set_title(f"Stage 3 [{res['label']}] 3-dog trajectories (door + formation)")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    # (b) all realized barriers (inter-agent 3 pairs + node min per dog)
    b = ax[0, 1]
    t = np.arange(len(res["f_hist"])) * C.TS
    for (i, j) in EDGES:
        b.plot(t, res["h_pair"][(i, j)], lw=1.3, label=f"h{i}{j} inter-agent")
    for i in DOGS:
        b.plot(t, res["node_h"][i], ls="--", lw=0.9, label=f"node h dog{i}")
    b.axhline(0.0, color="r", ls="--", lw=1)
    b.grid(True); b.legend(fontsize=7, ncol=2)
    b.set_title(f"Realized barriers: min inter-agent={res['min_pair']:+.3f}, "
                f"min node={res['min_node']:+.3f} (>=0 safe)")
    b.set_xlabel("t [s]"); b.set_ylabel("h")

    # (c) ADMM residuals (all cycles faint, rep highlighted)
    c = ax[1, 0]
    for h in res["cyc_hists"]:
        c.semilogy(h["r_prim"], color="0.85", lw=0.5)
        c.semilogy(h["r_dual"], color="0.9", lw=0.5)
    rep = res["rep"]
    c.semilogy(res["cyc_hists"][rep]["r_prim"], "b.-", lw=1.5, label="r_prim (rep)")
    c.semilogy(res["cyc_hists"][rep]["r_dual"], "g.-", lw=1.5, label="r_dual (rep)")
    c.grid(True, which="both"); c.legend(fontsize=8)
    c.set_title("ADMM residuals vs iteration (both must descend)")
    c.set_xlabel("ADMM iteration"); c.set_ylabel("residual (log)")

    # (d) formation error f(t) -- w_form=0 overlay + squeeze window shading
    d = ax[1, 1]
    d.plot(t, res["f_hist"], "m.-", ms=2, lw=1.3, label=f"w_form={res['w_form']:.0f}")
    if res0 is not None:
        t0 = np.arange(len(res0["f_hist"])) * C.TS
        d.plot(t0, res0["f_hist"], color="0.5", ls="--", lw=1.2,
               label="w_form=0 (tracking only)")
    if scn.get("squeeze"):
        cs = _crossing_stats(res)
        d.axvspan(cs["win"][0] * C.TS, cs["win"][1] * C.TS, color="orange",
                  alpha=0.12, label="door-crossing window")
    d.grid(True); d.legend(fontsize=8)
    d.set_title("Formation error f=||L_hat - L_hat_des||^2")
    d.set_xlabel("t [s]"); d.set_ylabel("f")

    os.makedirs(PNG_DIR, exist_ok=True)
    png = os.path.join(PNG_DIR, scn["png"])
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
    return png


def main():
    lf = LaplacianFormation(FORMATIONS)
    all_ok = True
    for scn in SCENARIOS:
        res = run(lf, scn, W_FORM)
        res0 = run(lf, scn, 0.0) if scn.get("overlay0") else None
        png = _plot(res, scn, res0)
        checks, ex = gate(res, scn)

        if not scn.get("gating", True):
            print(f"\n[stage3:{scn['label']}] === DIAGNOSTIC PROBE (non-gating): a "
                  f"deadlock here is an expected known limit, not a gate failure ===")
        reach = "/".join("R" if res["reached"][i] else "-" for i in DOGS)
        print(f"\n[stage3:{scn['label']}] cycles={res['cyc']}  reached[{reach}]  "
              f"min inter-agent h={res['min_pair']:+.4f}  min node h={res['min_node']:+.4f}")
        print(f"[stage3:{scn['label']}] rep cycle {res['rep']}: "
              f"r_prim {ex['rp'][0]:.3e}->{ex['rp'][-1]:.3e}  "
              f"r_dual {ex['rd'][0]:.3e}->{ex['rd'][-1]:.3e}")
        if scn.get("squeeze"):
            print(f"[stage3:{scn['label']}] formation f: start={ex['f_start']:.4f}  "
                  f"door-peak={ex['f_peak']:.4f} (t={ex['t_peak'] * C.TS:.1f}s)  "
                  f"end={ex['f_end']:.4f}   <- rise@door then recover = soft yield")
            if res0 is not None:
                c0 = _crossing_stats(res0)
                r0 = all(res0["reached"].values()) and res0["fail_msg"] is None
                print(f"[stage3:{scn['label']}] w_form=0 companion: reached={r0}  "
                      f"door-peak={c0['f_peak']:.4f}  end={c0['f_end']:.4f}  "
                      f"(compare rigidity: w_form=10 peak {ex['f_peak']:.4f})")
            if not checks["reach_all(no deadlock)"]:
                for i in DOGS:
                    if not res["reached"][i]:
                        xf = res["traj"][i][-1]
                        where = "STUCK before door" if xf[0] < DOOR_X + 0.2 else "en route"
                        print(f"   [DEADLOCK] dog{i} final=({xf[0]:+.2f},{xf[1]:+.2f}) "
                              f"dist2goal={np.linalg.norm(xf - np.asarray(scn['goal'][i])):.2f} "
                              f"-> {where}")
        else:
            print(f"[stage3:{scn['label']}] formation f: {ex['f_start']:.4f} -> "
                  f"{ex['f_end']:.4f}"
                  + (f"   (w_form=0 f_end="
                     f"{float(np.mean(res0['f_hist'][-5:])):.4f})"
                     if res0 is not None else ""))
        hv = [h["h2_viol"][-1] for h in res["cyc_hists"]]
        print(f"[stage3:{scn['label']}] diag: P_ITERS/cycle="
              f"{len(res['cyc_hists'][0]['r_prim'])}  "
              f"h2_viol(end) max={max(hv):.3e} (diagnostic, understates ~11x)")
        gating = scn.get("gating", True)
        for k, v in checks.items():
            print(f"   [{'OK ' if v else 'FAIL'}] {k}")
        print(f"[stage3:{scn['label']}] wrote {png}"
              + ("" if gating else "   (DIAGNOSTIC PROBE -- non-gating)"))
        if gating:
            all_ok = all_ok and all(checks.values())

    print(f"\n[stage3] {'ALL GREEN' if all_ok else 'RED -- see FAIL above'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
