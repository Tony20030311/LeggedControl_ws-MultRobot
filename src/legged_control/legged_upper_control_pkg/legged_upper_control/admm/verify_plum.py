"""rung-2 梅花樁 (plum-blossom) gate -- three-dog ADMM + V formation threads a dense
quincunx peg field via per-dog A* CURVED references.

Why this gate exists (and why A* is mandatory here): the ADMM node/edge CBF only makes
the NEXT step (k=0) hard, and the per-dog reference is a straight line. A peg sitting on
that straight line -> the reactive single-linearization CBF cannot find the global detour
-> DEADLOCK (safety holds, liveness fails). This is the stage-1 var3 / stage-3 squeeze
lesson. The fix is the EXISTING core.planning.AStarPlanner feeding the EXISTING
build_reference (which already samples a polyline) -- no ADMM-core change.

Imports ARENAS["plum"] straight from scripts/ocs2_fleet_publisher.py so the peg geometry
+ goals can NEVER drift from what the fleet publisher runs on the robot (and from the
Gazebo cylinders in legged_gazebo/worlds/three_dogs_plum.world, paired by eye).

Gate (a bad curve is a bug to REPORT, never tune geometry to force):
  1. A* finds a non-trivial path for every dog (waypoints > 2 -> not a straight fallback)
  2. all three reach their goal (no deadlock) with A* refs
  3. realized inter-agent h >= -eps every cycle for all 3 pairs
  4. realized node CBF h >= -eps every cycle for every dog
  5. r_prim AND r_dual descend
Also REPORTED (documents necessity, not gated -- a negative test is fragile): straight
refs on the SAME field DEADLOCK.

Run in the container (ROS sourced -> core.formation + the publisher import rospy):
  source /opt/ros/noetic/setup.bash
  python3 legged_upper_control/admm/verify_plum.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # legged_upper_control/ -> core.*
sys.path.insert(0, HERE)                                   # admm/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))  # publisher
from admm_impl import constants as C  # noqa: E402
from admm_impl import ac  # noqa: E402
from reference import build_reference                       # noqa: E402
from admm_impl import LaplacianFormation  # noqa: E402
from core.planning import AStarPlanner                      # noqa: E402
import verify_arena as va                                   # noqa: E402  (reuse realized_* + gate + _plot)
from ocs2_fleet_publisher import ARENAS, FORMATIONS         # noqa: E402  (single source of truth)

PNG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress"))
DOGS = (1, 2, 3)
EDGES = ((1, 2), (1, 3), (2, 3))
# Gazebo V spawn (three_dogs_plum.launch): dog1(2,0) dog2(1,1) dog3(1,-1).
START = {1: (2.0, 0.0), 2: (1.0, 1.0), 3: (1.0, -1.0)}
MAX_CYCLES = 800
GOAL_TOL = 0.20
W_FORM = 10.0
TOL = 1e-6
# A* inflation: peg radius 0.30 + robot_radius 0.35 = 0.65 keeps the reference centre
# >= node-CBF r_eff (0.30 + ROBOT_MARGIN 0.30 = 0.60) from every peg -> no deadlock.
ASTAR_ROBOT_RADIUS = 0.35


def _run(coord, lf, obstacles, goal, paths):
    """Closed loop with per-dog reference polylines (paths[i]); returns a verify_arena-
    shaped res dict so va.gate()/va._plot() apply unchanged."""
    x = {i: np.array([START[i][0], START[i][1], 0.0, 0.0]) for i in DOGS}
    traj = {i: [x[i][:2].copy()] for i in DOGS}
    h_pair = {e: [] for e in EDGES}
    node_h = {i: [] for i in DOGS}
    f_hist, cyc_hists = [], []
    reached = {i: False for i in DOGS}
    cyc = -1
    for cyc in range(MAX_CYCLES):
        xnow = {i: x[i] for i in DOGS}
        xdes = {i: build_reference(x[i][:2], paths[i], v_cruise=0.15) for i in DOGS}
        xi, hist = coord.step(xnow, xdes)
        cyc_hists.append(hist)
        for (a, b) in EDGES:
            h_pair[(a, b)].append(va.realized_pair_h(x[a][:2], x[b][:2]))
        for i in DOGS:
            node_h[i].append(va.realized_node_h(x[i][:2], obstacles))
        f_now, _ = lf.compute([x[i][:2] for i in DOGS])
        f_hist.append(f_now)
        for i in DOGS:
            a0 = xi[i][C.a_index(0, C.N):C.a_index(0, C.N) + 2]
            x[i] = C.Ad @ x[i] + C.Bd @ a0
            traj[i].append(x[i][:2].copy())
            if np.linalg.norm(x[i][:2] - np.asarray(goal[i])) < GOAL_TOL:
                reached[i] = True
        if all(reached.values()):
            break
    traj = {i: np.array(traj[i]) for i in DOGS}
    h_pair = {e: np.array(v) for e, v in h_pair.items()}
    node_h = {i: np.array(v) for i, v in node_h.items()}
    return dict(cyc=cyc + 1, reached=reached, fail_msg=None, traj=traj,
                h_pair=h_pair, node_h=node_h, f_hist=np.array(f_hist),
                cyc_hists=cyc_hists,
                min_pair=min(float(h_pair[e].min()) for e in EDGES),
                min_node=min(float(node_h[i].min()) for i in DOGS),
                rep=int(np.argmax([h["r_prim"][0] for h in cyc_hists])))


def main():
    arena = ARENAS["plum"]
    obstacles = [{"pos": o["pos"], "radius": o["radius"]} for o in arena["obstacles"]]
    goal = arena["goals"]
    va._assert_arena_shape(obstacles)

    lf = LaplacianFormation(FORMATIONS)

    # --- REPORTED: straight refs on the same field deadlock (documents A* necessity) ---
    lf.set_formation("V")
    coord_s = ac.ADMMCoordinator(dogs=DOGS, edges=EDGES, obstacles=obstacles,
                                 walls=[], formation=lf, w_form=W_FORM)
    straight = {i: [list(START[i]), list(goal[i])] for i in DOGS}
    res_s = _run(coord_s, lf, obstacles, goal, straight)
    straight_deadlock = not all(res_s["reached"].values())
    print("[plum] REPORT straight-ref: reached[%s] deadlock=%s  (A* is required)"
          % ("/".join("R" if res_s["reached"][i] else "-" for i in DOGS), straight_deadlock))

    # --- GATED: A* curved refs thread the field ---
    lf.set_formation("V")
    coord = ac.ADMMCoordinator(dogs=DOGS, edges=EDGES, obstacles=obstacles,
                               walls=[], formation=lf, w_form=W_FORM)
    planner = AStarPlanner(resolution=0.15, robot_radius=ASTAR_ROBOT_RADIUS,
                           obstacles=obstacles, x_min=0.0, x_max=10.0,
                           y_min=-4.0, y_max=4.0)
    paths, wp_ok = {}, True
    for i in DOGS:
        p = planner.plan(START[i], goal[i])
        paths[i] = p if p else [list(START[i]), list(goal[i])]
        wp_ok = wp_ok and len(p) > 2
        print("[plum] dog%d A* waypoints=%d" % (i, len(p)))
    res = _run(coord, lf, obstacles, goal, paths)

    png = va._plot(res, obstacles, goal, os.path.join(PNG_DIR, "rung2_plum.png"))
    checks = va.gate(res)
    checks["astar_found_curved_paths(wp>2)"] = wp_ok
    checks["reach_all(no deadlock)"] = all(res["reached"].values())

    reach = "/".join("R" if res["reached"][i] else "-" for i in DOGS)
    print("\n[plum] %d pegs, cycles=%d  reached[%s]  min inter-agent h=%+.4f  min node h=%+.4f"
          % (len(obstacles), res["cyc"], reach, res["min_pair"], res["min_node"]))
    for i in DOGS:
        xf = res["traj"][i][-1]
        print("   dog%d final=(%+.2f,%+.2f)  dist2goal=%.2f  %s"
              % (i, xf[0], xf[1], np.linalg.norm(xf - np.asarray(goal[i])),
                 "REACHED" if res["reached"][i] else "STUCK"))
    for k, v in checks.items():
        print("   [%s] %s" % ("OK " if v else "FAIL", k))
    print("[plum] wrote %s" % png)

    ok = all(checks.values())
    print("\n[plum] %s" % ("ALL GREEN" if ok else "RED -- see FAIL above"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
