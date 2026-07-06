"""rung-2 basic-arena gate -- three-dog ADMM + V formation + node CIRCULAR obstacle
CBF, closed loop, OPEN FIELD (no walls, no door).

This is the offline sanity for wiring the node obstacle CBF onto Gazebo
(three_dogs_obstacles.launch). It closes the loop on the SAME arena the fleet
publisher publishes -- it imports ARENAS straight from
scripts/ocs2_fleet_publisher.py so the CBF geometry can never drift from what runs
on the robot (the arena obstacle "pos" must also match the cylinders in
legged_gazebo/worlds/three_dogs_obstacles.world; that pairing is checked by eye /
Gazebo, this script checks the CBF math).

Mechanism proven: obstacles placed OFF each dog's straight per-dog reference (no A*
yet -> a head-on obstacle would deadlock, stage-1 var3 lesson) -> the node CBF
deflects the whole V laterally, edge CBF keeps the dogs apart, formation cost holds
the shape, ADMM still converges, everyone reaches the V slot.

Gate (a bad curve is a bug to report, NEVER tune geometry to force):
  1. all three reach their formation slot at the goal (no deadlock)
  2. realized inter-agent h >= 0 every cycle for all 3 pairs (raw ||p^i-p^j||^2 - d^2)
  3. realized node CBF h >= 0 every cycle for every dog (||p-p_obs||^2 - r_eff^2)
  4. r_prim AND r_dual descend over ADMM iterations
  5. formation error f falls and stays low

Run inside the container (ROS sourced -> core.formation + the publisher import rospy;
osqp 0.6.x):
  source /opt/ros/noetic/setup.bash
  python3 legged_upper_control/admm/verify_arena.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # legged_upper_control/ -> core.formation
sys.path.insert(0, HERE)                                   # admm/ -> constants, coordinator, ...
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))  # publisher
import constants as C                                       # noqa: E402
import admm_coordinator as ac                               # noqa: E402
import node_subproblem as nq                                # noqa: E402
from reference import build_reference                       # noqa: E402
from core.formation import LaplacianFormation               # noqa: E402
# single source of truth for the arena (obstacles + goals) + formation shapes:
from ocs2_fleet_publisher import ARENAS, FORMATIONS         # noqa: E402

PNG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress"))
DOGS = (1, 2, 3)
EDGES = ((1, 2), (1, 3), (2, 3))
MAX_CYCLES = 550
GOAL_TOL = 0.20
W_FORM = 10.0
ROBOT_MARGIN = 0.30                                          # node default; r_eff = radius + margin
ARENA = "obstacles"
# Gazebo V spawn (three_dogs_obstacles.launch): dog1(2,0) dog2(1,1) dog3(1,-1).
START = {1: (2.0, 0.0), 2: (1.0, 1.0), 3: (1.0, -1.0)}


def realized_pair_h(pi, pj):
    e = np.asarray(pi) - np.asarray(pj)
    return float(e @ e) - C.D_MIN ** 2


def realized_node_h(p, obstacles):
    return min(nq.h_obstacle(p, o["pos"], o["radius"] + ROBOT_MARGIN) for o in obstacles)


def _assert_arena_shape(obstacles):
    """Structural guard: the arena obstacles must be the {"pos":(x,y),"radius":r}
    dicts the node CBF expects (node_subproblem._features reads exactly these keys)."""
    assert obstacles, "arena has no obstacles"
    for o in obstacles:
        assert set(("pos", "radius")) <= set(o), "obstacle missing pos/radius: %r" % o
        assert len(o["pos"]) == 2, "obstacle pos must be (x,y): %r" % o
        float(o["radius"])
    print("[arena] structural guard OK: %d obstacles, keys {pos,radius}" % len(obstacles))


def run(lf, obstacles, goal, w_form):
    lf.set_formation("V")
    coord = ac.ADMMCoordinator(dogs=DOGS, edges=EDGES,
                               obstacles=obstacles, walls=[],
                               formation=lf, w_form=w_form)
    x = {i: np.array([START[i][0], START[i][1], 0.0, 0.0]) for i in DOGS}
    traj = {i: [x[i][:2].copy()] for i in DOGS}
    h_pair = {e: [] for e in EDGES}
    node_h = {i: [] for i in DOGS}
    f_hist, cyc_hists = [], []
    reached = {i: False for i in DOGS}
    fail_msg = None
    cyc = -1

    for cyc in range(MAX_CYCLES):
        xnow = {i: x[i] for i in DOGS}
        xdes = {i: build_reference(x[i][:2], [START[i], goal[i]]) for i in DOGS}
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
                fail_msg = "cycle %d: dog %d a0 not finite" % (cyc, i)
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
    return dict(cyc=cyc + 1, reached=reached, fail_msg=fail_msg, traj=traj,
                h_pair=h_pair, node_h=node_h, f_hist=f_hist, cyc_hists=cyc_hists,
                min_pair=min(float(h_pair[e].min()) for e in EDGES),
                min_node=min(float(node_h[i].min()) for i in DOGS),
                rep=int(np.argmax([h["r_prim"][0] for h in cyc_hists])))


def gate(res):
    rp = np.array(res["cyc_hists"][res["rep"]]["r_prim"])
    rd = np.array(res["cyc_hists"][res["rep"]]["r_dual"])
    f = res["f_hist"]
    return {
        "reach_all(no deadlock)": all(res["reached"].values()) and res["fail_msg"] is None,
        "interagent_h>=0": res["min_pair"] >= -1e-6,
        "node_h>=0": res["min_node"] >= -1e-6,
        "r_prim_descends": rp[-1] < rp[0] and rp[-1] < 0.3 * rp.max(),
        "r_dual_descends": rd[-1] < rd[0] and rd[-1] < 0.3 * rd.max(),
        "formation_improves": float(np.mean(f[-5:])) < 0.5 * float(np.mean(f[:3])),
    }


def _plot(res, obstacles, goal, png):
    traj = res["traj"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cols = {1: "b", 2: "r", 3: "g"}
    a = ax[0, 0]
    for i in DOGS:
        a.plot(traj[i][:, 0], traj[i][:, 1], cols[i] + ".-", ms=2, lw=1, label="dog%d" % i)
        a.plot(*START[i], cols[i] + "d", ms=7)
        a.plot(*goal[i], cols[i] + "*", ms=13)
    th = np.linspace(0, 2 * np.pi, 60)
    for k, o in enumerate(obstacles):
        oc, r = np.array(o["pos"]), o["radius"]
        a.plot(oc[0] + 0.20 * np.cos(th), oc[1] + 0.20 * np.sin(th), color="0.4", lw=2,
               label="obstacle (phys r=0.20)" if k == 0 else None)
        a.plot(oc[0] + (r + ROBOT_MARGIN) * np.cos(th),
               oc[1] + (r + ROBOT_MARGIN) * np.sin(th), color="orange", ls=":", lw=1,
               label="CBF r_eff=%.2f" % (r + ROBOT_MARGIN) if k == 0 else None)
    a.set_aspect("equal"); a.grid(True); a.legend(fontsize=8, loc="upper left")
    a.set_title("rung2 basic arena: 3-dog trajectories (open field + circular obstacles)")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    b = ax[0, 1]
    t = np.arange(len(res["f_hist"])) * C.TS
    for (i, j) in EDGES:
        b.plot(t, res["h_pair"][(i, j)], lw=1.3, label="h%d%d inter-agent" % (i, j))
    for i in DOGS:
        b.plot(t, res["node_h"][i], ls="--", lw=0.9, label="node h dog%d" % i)
    b.axhline(0.0, color="r", ls="--", lw=1)
    b.grid(True); b.legend(fontsize=7, ncol=2)
    b.set_title("Realized barriers: min inter-agent=%+.3f, min node=%+.3f (>=0 safe)"
                % (res["min_pair"], res["min_node"]))
    b.set_xlabel("t [s]"); b.set_ylabel("h")

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

    d = ax[1, 1]
    d.plot(t, res["f_hist"], "m.-", ms=2, lw=1.3, label="w_form=%.0f" % W_FORM)
    d.grid(True); d.legend(fontsize=8)
    d.set_title("Formation error f=||L_hat - L_hat_des||^2")
    d.set_xlabel("t [s]"); d.set_ylabel("f")

    os.makedirs(PNG_DIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
    return png


def main():
    arena = ARENAS[ARENA]
    obstacles = [{"pos": o["pos"], "radius": o["radius"]} for o in arena["obstacles"]]
    goal = arena["goals"]
    _assert_arena_shape(obstacles)

    lf = LaplacianFormation(FORMATIONS)
    res = run(lf, obstacles, goal, W_FORM)
    png = _plot(res, obstacles, goal, os.path.join(PNG_DIR, "rung2_arena.png"))
    checks = gate(res)

    reach = "/".join("R" if res["reached"][i] else "-" for i in DOGS)
    print("\n[arena] cycles=%d  reached[%s]  min inter-agent h=%+.4f  min node h=%+.4f"
          % (res["cyc"], reach, res["min_pair"], res["min_node"]))
    ex = res["cyc_hists"][res["rep"]]
    print("[arena] rep cycle %d: r_prim %.3e->%.3e  r_dual %.3e->%.3e"
          % (res["rep"], ex["r_prim"][0], ex["r_prim"][-1], ex["r_dual"][0], ex["r_dual"][-1]))
    print("[arena] formation f: %.4f -> %.4f"
          % (float(np.mean(res["f_hist"][:3])), float(np.mean(res["f_hist"][-5:]))))
    if not checks["reach_all(no deadlock)"]:
        for i in DOGS:
            if not res["reached"][i]:
                xf = res["traj"][i][-1]
                print("   [DEADLOCK] dog%d final=(%+.2f,%+.2f) dist2goal=%.2f"
                      % (i, xf[0], xf[1], np.linalg.norm(xf - np.asarray(goal[i]))))
    for k, v in checks.items():
        print("   [%s] %s" % ("OK " if v else "FAIL", k))
    print("[arena] wrote %s" % png)

    ok = all(checks.values())
    print("\n[arena] %s" % ("ALL GREEN" if ok else "RED -- see FAIL above"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
