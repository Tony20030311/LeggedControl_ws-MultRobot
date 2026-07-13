"""rung-2 door gate — three-dog ADMM threads the REAL target map (obstacle_world /
five_dogs.launch): a 2.5m DOOR at x=6 + 5 field cylinders + boundary WALLS, via per-dog
A* curved references. First gate that exercises the node-CBF WALL half-plane.

Why the door needs both A* rects AND circular posts (stage-3 lesson "half-plane walls
can't make an opening"): a half-plane at x=6 would block the door gap. So A* gets the full
wall boxes as rect obstacles (routes through the 2.5m gap) and the node CBF gets circular
door-POSTS at the corners. Boundary walls (right/top/bottom, no gap) are node-CBF
half-planes — valid, and the first time a wall CBF is checked in the loop.

Imports ARENAS["door"] (obstacles + walls + rects + goals) through the admm_impl shim
(admm_core_cpp fleet_config) so geometry can't drift from what the robot runs (and from
the cylinders/walls in legged_gazebo/worlds/obstacle_world.world, paired by eye).

Gate:
  1. A* finds a curved path for every dog (wp > 2)
  2. all three reach their (reachable-snapped) goal — no deadlock
  3. realized inter-agent h >= -eps every cycle
  4. realized node CBF h (cylinders + door posts) >= -eps
  5. realized WALL h (boundary half-planes) >= -eps
  6. r_prim AND r_dual descend

Run in the container: source /opt/ros/noetic/setup.bash
  python3 legged_upper_control/admm/verify_door.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from admm_impl import constants as C  # noqa: E402
from admm_impl import ac  # noqa: E402
from admm_impl import nq  # noqa: E402
from admm_impl import build_reference                       # noqa: E402
from admm_impl import LaplacianFormation  # noqa: E402
from admm_impl import AStarPlanner                      # noqa: E402
import verify_arena as va                                   # noqa: E402
from admm_impl import ARENAS, FORMATIONS         # noqa: E402  (single source of truth)

PNG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "progress"))
DOGS = (1, 2, 3)
EDGES = ((1, 2), (1, 3), (2, 3))
START = {1: (2.0, 0.0), 2: (1.0, 1.0), 3: (1.0, -1.0)}     # obstacle_world V spawn
MAX_CYCLES = 900
GOAL_TOL = 0.25
W_FORM = 10.0


def realized_wall_h(p, walls):
    if not walls:
        return np.inf
    return min(nq.h_wall(p, w["normal"], w["point"], w["d_safe"]) for w in walls)


def main():
    arena = ARENAS["door"]
    obstacles = [{"pos": o["pos"], "radius": o["radius"]} for o in arena["obstacles"]]
    walls, rects, goal = arena["walls"], arena["rects"], arena["goals"]
    va._assert_arena_shape(obstacles)

    lf = LaplacianFormation(FORMATIONS); lf.set_formation("V")
    coord = ac.ADMMCoordinator(dogs=DOGS, edges=EDGES, obstacles=obstacles,
                               walls=walls, formation=lf, w_form=W_FORM)
    planner = AStarPlanner(resolution=0.15, robot_radius=0.35, obstacles=obstacles,
                           x_min=0.0, x_max=10.0, y_min=-5.0, y_max=5.0,
                           rect_obstacles=rects)
    paths, goal_adj, wp_ok = {}, {}, True
    for i in DOGS:
        cand, p = planner.find_reachable_goal(START[i], goal[i])
        goal_adj[i] = cand if cand is not None else goal[i]
        paths[i] = p if p else [list(START[i]), list(goal_adj[i])]
        wp_ok = wp_ok and len(p) > 2
        print("[door] dog%d A* wp=%d goal(%.2f,%.2f)->(%.2f,%.2f)"
              % (i, len(p), goal[i][0], goal[i][1], goal_adj[i][0], goal_adj[i][1]))

    x = {i: np.array([START[i][0], START[i][1], 0.0, 0.0]) for i in DOGS}
    traj = {i: [x[i][:2].copy()] for i in DOGS}
    h_pair = {e: [] for e in EDGES}; node_h = {i: [] for i in DOGS}; wall_h = {i: [] for i in DOGS}
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
            wall_h[i].append(realized_wall_h(x[i][:2], walls))
        f_hist.append(lf.compute([x[i][:2] for i in DOGS])[0])
        for i in DOGS:
            a0 = xi[i][C.a_index(0, C.N):C.a_index(0, C.N) + 2]
            x[i] = C.Ad @ x[i] + C.Bd @ a0
            traj[i].append(x[i][:2].copy())
            if np.linalg.norm(x[i][:2] - np.asarray(goal_adj[i])) < GOAL_TOL:
                reached[i] = True
        if all(reached.values()):
            break

    traj = {i: np.array(traj[i]) for i in DOGS}
    res = dict(cyc=cyc + 1, reached=reached, fail_msg=None, traj=traj,
               h_pair={e: np.array(v) for e, v in h_pair.items()},
               node_h={i: np.array(v) for i, v in node_h.items()},
               f_hist=np.array(f_hist), cyc_hists=cyc_hists,
               min_pair=min(float(np.min(h_pair[e])) for e in EDGES),
               min_node=min(float(np.min(node_h[i])) for i in DOGS),
               rep=int(np.argmax([h["r_prim"][0] for h in cyc_hists])))
    min_wall = min(float(np.min(wall_h[i])) for i in DOGS)

    checks = va.gate(res)
    checks["astar_found_curved_paths(wp>2)"] = wp_ok
    checks["reach_all(no deadlock)"] = all(res["reached"].values())
    checks["wall_h>=0"] = min_wall >= -1e-6

    reach = "/".join("R" if res["reached"][i] else "-" for i in DOGS)
    print("\n[door] cycles=%d reached[%s] min inter-agent h=%+.4f min node h=%+.4f min wall h=%+.4f"
          % (res["cyc"], reach, res["min_pair"], res["min_node"], min_wall))
    for i in DOGS:
        xf = traj[i][-1]
        print("   dog%d final=(%+.2f,%+.2f) dist2goal=%.2f %s"
              % (i, xf[0], xf[1], np.linalg.norm(xf - np.asarray(goal_adj[i])),
                 "REACHED" if res["reached"][i] else "STUCK"))
    for k, v in checks.items():
        print("   [%s] %s" % ("OK " if v else "FAIL", k))

    ok = all(checks.values())
    print("\n[door] %s" % ("ALL GREEN" if ok else "RED -- see FAIL above"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
