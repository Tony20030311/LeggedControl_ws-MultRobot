#!/usr/bin/env python3
"""rung 2 -- three-dog ADMM formation -> OCS2 targets (ONE coupled node).

The ADMM couples the three dogs (one stateful ADMMCoordinator runs node x3 + edge x3
+ dual + warm-start inside a single step()), so this is ONE rospy node that subscribes
to all three dogs and publishes all three targets -- it CANNOT be split into three
processes.

Per 10 Hz cycle (mirrors the single-dog ocs2_target_publisher._rung1_target, x3):
  sub  /dogN/dogN_mpc_observation  (time + estimator px,py + measured yaw seed)
  sub  /dogN/ground_truth/state    (world position + velocity)
    -> X0[i]  = [P0.x, P0.y, vx_ema, vy_ema]  (world; gt pos + EMA/clamped obs vel)
    -> xdes[i]= build_reference(P0[i], [P0[i], slot_goal[i]])   (per-dog line to V slot)
  coord.step(xnow, xdes) -> xi[i] (6N) x3          <- the only coupled point
    per dog: re-express xi[i] positions to dog i's MPC estimator frame, then
             adapter.build_target(yaw_mode='path', seed_yaw = dog i's MEASURED yaw)
  pub  /dogN/dogN_mpc_target x3

Standup is done BEFORE this runs (fleet_bringup + start_fleet.sh stand+trot, then the
three C++ legged_robot_target nodes are killed so this owns the target topics). No
warmup-hold here. Formation / goals / w_form are params (default = stage-3 V).
"""

import os
import sys
import math
import itertools

import numpy as np
import rospy
from ocs2_msgs.msg import (mpc_target_trajectories, mpc_state, mpc_input,
                           mpc_observation)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

# rospy-free ADMM core (admm/) + core.formation (this node IS rospy, so it may import
# core.formation and inject it -- keeping the coordinator itself rospy-free).
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ADMM = os.path.join(_PKG, "legged_upper_control", "admm")
sys.path.insert(0, _ADMM)
sys.path.insert(0, os.path.join(_PKG, "legged_upper_control"))   # core.formation
import motion_adapter as ma            # noqa: E402
import constants as C                  # noqa: E402
import admm_coordinator as ac          # noqa: E402
from reference import build_reference  # noqa: E402
from core.formation import LaplacianFormation  # noqa: E402
from core.planning import AStarPlanner  # noqa: E402  (dense-arena curved refs)

# A1 standing pose (same placeholder as the single-dog publisher).
A1_DEFAULT_JOINTS = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                     0.1, 1.0, -1.5, -0.1, 1.0, -1.5]

# formation shapes (offsets rel. to dog1), from the stage-3 gate (verify_stage3.py).
FORMATIONS = {
    "V":      [(0.0, 0.0), (-0.7, 0.5), (-0.7, -0.5)],
    "column": [(0.0, 0.0), (-0.7, 0.0), (-1.4, 0.0)],
    "V_wide": [(0.0, 0.0), (-0.7, 0.8), (-0.7, -0.8)],
}
# default per-dog V slot goals (stage-3 V scenario).
DEFAULT_GOALS = {1: (3.0, 0.0), 2: (2.3, 0.5), 3: (2.3, -0.5)}


def rot2d(yaw):
    """2D rotation matrix (world frame). Ported from the legacy Formation_manager."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def centroid_slot_targets(goal_c, offsets, yaw):
    """Legacy virtual-centroid command: given a formation-CENTROID goal, the (mean-
    centred) V offsets, and a heading, return the per-slot world targets so the fleet
    CENTROID lands on goal_c and the V faces `yaw`. slot[k] = goal_c + R(yaw)*offset_c[k].
    Mean-centring makes goal_c the true centroid (not dog1). Pure math -- unit-testable."""
    offs_c = [np.asarray(o, dtype=float) - np.mean(offsets, axis=0) for o in offsets]
    R = rot2d(yaw)
    return [np.asarray(goal_c, dtype=float) + R @ o for o in offs_c]


def min_cost_assignment(positions, slots):
    """One-shot nearest-slot assignment: the permutation of slot indices minimising
    Σ‖pos[k]-slot[perm[k]]‖². Prevents the 180°-turn reshuffle without any hysteresis/
    freeze machinery (which ADMM forbids). positions/slots are equal-length lists."""
    best, bestcost = tuple(range(len(positions))), 1e18
    for perm in itertools.permutations(range(len(slots))):
        c = sum(float(np.dot(positions[k] - slots[perm[k]], positions[k] - slots[perm[k]]))
                for k in range(len(positions)))
        if c < bestcost:
            bestcost, best = c, perm
    return best

# obstacle arenas (rung 2 basic arena): circular obstacles fed to the node CBF plus
# matching per-dog goals. Each obstacle "pos" MUST match a cylinder in the Gazebo world
# (legged_gazebo/worlds/three_dogs_obstacles.world). "radius" = physical + clearance
# buffer; the node CBF adds robot_margin (0.30) -> r_eff. Select via ~arena (default ""
# = empty world, backward-compatible with the empty-world rung-2 flow).
ARENAS = {
    "obstacles": {
        "obstacles": [{"pos": (4.0, 0.40), "radius": 0.30},
                      {"pos": (5.5, 0.40), "radius": 0.30}],
        "goals": {1: (7.0, 0.0), 2: (6.3, 0.5), 3: (6.3, -0.5)},
    },
    # 梅花樁 (plum-blossom / quincunx): dense circular pegs the V formation threads.
    # Straight refs DEADLOCK on any in-corridor peg -> this arena REQUIRES ~use_astar
    # (per-dog A* curved refs). Pegs off the exact centre lines; r_eff = 0.30+0.30=0.60.
    # Offline-validated: scratchpad/measure_plum.py threads all 3 clean, barriers>=0.
    "plum": {
        "obstacles": [{"pos": p, "radius": 0.30} for p in
                      [(3.4, 1.1), (3.4, -1.1),
                       (4.7, 0.0), (4.7, 2.1), (4.7, -2.1),
                       (6.0, 1.1), (6.0, -1.1),
                       (7.3, 0.0), (7.3, 2.1), (7.3, -2.1)]],
        "goals": {1: (9.0, 0.0), 2: (8.3, 0.5), 3: (8.3, -0.5)},
    },
    # door arena = the real target map (legged_gazebo/worlds/obstacle_world.world, the
    # five_dogs.launch arena). A 2.5m DOOR at x=6 + 5 field cylinders + boundary walls.
    # A half-plane can't make a gap (stage-3 lesson), so the door is: A* rect obstacles
    # (full wall boxes -> routes through the gap) + circular door-POSTS at the corners
    # (local node CBF). Boundary walls (right/top/bottom, no gap) are node-CBF half-planes.
    # Requires ~use_astar. Offline-validated: scratchpad/measure_door.py threads all 3.
    "door": {
        "obstacles": ([{"pos": p, "radius": 0.30} for p in    # 5 field cylinders (r_eff 0.60)
                       [(7.5, 2.5), (8.5, -1.5), (9.0, 3.5), (7.0, -3.0), (8.0, 0.5)]]
                      + [{"pos": p, "radius": 0.15} for p in   # door posts (r_eff 0.45)
                         [(6.0, 1.25), (6.0, 1.75), (6.0, -1.25), (6.0, -1.75)]]),
        "walls": [   # boundary half-planes: normal points INTO free space
            {"normal": (-1.0, 0.0), "point": (9.925, 0.0), "d_safe": 0.30},   # right x=10
            {"normal": (0.0, -1.0), "point": (0.0, 4.925), "d_safe": 0.30},   # top y=5
            {"normal": (0.0, 1.0), "point": (0.0, -4.925), "d_safe": 0.30},   # bottom y=-5
        ],
        "rects": [   # A* rect obstacles = the 5 physical wall boxes (center, size)
            {"center": (6.0, 3.125), "size": (0.15, 3.75)},   # wall_left_upper
            {"center": (6.0, -3.125), "size": (0.15, 3.75)},  # wall_left_lower
            {"center": (10.0, 0.0), "size": (0.15, 10.0)},    # wall_right
            {"center": (8.0, 5.0), "size": (4.0, 0.15)},      # wall_top
            {"center": (8.0, -5.0), "size": (4.0, 0.15)},     # wall_bottom
        ],
        "goals": {1: (9.3, 0.5), 2: (8.6, 1.0), 3: (8.6, 0.0)},   # V slots clear of (8,0.5)
    },
}


class FleetPublisher:
    def __init__(self):
        self.dogs = [1, 2, 3]
        self.edges = [(1, 2), (1, 3), (2, 3)]
        self.N, self.ts = C.N, C.TS
        self.v = float(rospy.get_param("~v", 0.15))
        self.com_height = float(rospy.get_param("~com_height", 0.30))
        self.formation_name = rospy.get_param("~formation", "V")
        self.w_form = float(rospy.get_param("~w_form", 10.0))
        # arena selection: "" = empty world (default, backward-compatible with the
        # empty-world rung-2 flow); a key in ARENAS -> its circular obstacles feed the
        # node CBF and its goals become the per-dog defaults.
        self.arena_name = rospy.get_param("~arena", "")
        arena = ARENAS.get(self.arena_name, {"obstacles": [], "goals": DEFAULT_GOALS})
        arena_obstacles, arena_goals = arena["obstacles"], arena["goals"]
        # walls = node-CBF half-planes (boundaries); rects = A*-only box obstacles (walls
        # the CBF can't represent as a gap, e.g. the door). Both optional (open arenas omit).
        arena_walls = arena.get("walls", [])
        self._arena_rects = arena.get("rects", [])
        # per-dog world-frame goals (arena defaults; override via ~goalN_x / ~goalN_y).
        self.goal = {i: np.array([float(rospy.get_param("~goal%d_x" % i, arena_goals[i][0])),
                                  float(rospy.get_param("~goal%d_y" % i, arena_goals[i][1]))])
                     for i in self.dogs}

        # ONE coupled coordinator: leaderless complete graph. obstacles from the arena
        # (node-local circular CBF); no walls in the open-field basic arena.
        lf = LaplacianFormation(FORMATIONS)
        lf.set_formation(self.formation_name)
        self.formation = lf                              # kept for /formation/goal slotting
        self._last_formation_yaw = 0.0                   # centroid->goal bearing memory
        self.coord = ac.ADMMCoordinator(dogs=tuple(self.dogs), edges=tuple(self.edges),
                                        obstacles=arena_obstacles, walls=arena_walls,
                                        formation=lf, w_form=self.w_form)
        # shared adapter (per-dog path yaw; seed re-anchored to each dog's MEASURED yaw
        # every cycle -> DON'T use adapter.adapt()'s internal seed, call build_target).
        self.adapter = ma.MotionAdapter(com_height=self.com_height,
                                        default_joints=A1_DEFAULT_JOINTS,
                                        n=self.N, ts=self.ts,
                                        yaw_mode="path", ema_alpha=0.2,
                                        lookahead_M=int(rospy.get_param("~lookahead_m", 5)),
                                        pos_eps=float(rospy.get_param("~pos_eps", 0.02)),
                                        max_yaw_rate=float(rospy.get_param("~yaw_rate_max", 1.2)))

        # goal-proximity yaw latch: near its slot a dog isn't really travelling, so the
        # path-lookahead direction is noise -> the measured-yaw-seeded feedback winds the
        # yaw up unboundedly (dog spins, trot topples). Latch the arrival heading within
        # r_latch, release past r_latch+margin (hysteresis). Breaks the loop at the goal.
        self.r_latch = float(rospy.get_param("~yaw_latch_r", 0.25))
        self.latch_margin = float(rospy.get_param("~yaw_latch_margin", 0.10))
        self._yaw_latch = {i: None for i in self.dogs}   # held heading, or None

        # dynamic safe-prefix truncation (edge handoff fix): each cycle send OCS2 only
        # the leading steps of the plan that clear D_MIN(+margin) from every other dog;
        # OCS2 clamps to the last (safe) sent point past that. Only k=0 is a HARD CBF
        # step, so the ADMM plan itself dives unsafe at a close approach -- we hand over
        # only the part we can vouch for. send_cap caps it at OCS2's ~1s horizon.
        self.send_cap = int(rospy.get_param("~k_send", C.K_SEND))
        self.d_safe = C.D_MIN + float(rospy.get_param("~send_margin", 0.0))

        # dense-arena curved references (~use_astar): straight per-dog refs DEADLOCK on
        # any peg in the formation corridor (the reactive single-linearization CBF can't
        # find the global detour). A* over the SAME obstacles gives each dog a curved
        # waypoint polyline that build_reference already samples -> no ADMM-core change.
        # Planned ONCE per dog on the first ready tick (static field); off by default so
        # the open-field / basic-obstacle flows stay byte-identical. r_astar (0.35) +
        # peg radius (0.30) = 0.65 keeps the reference centre >= node-CBF r_eff (0.60).
        self.use_astar = bool(rospy.get_param("~use_astar", False))
        self._path = {}                                  # per-dog A* polyline (lazy)
        self._planner = None
        if self.use_astar and self.coord.obstacles:
            r_astar = float(rospy.get_param("~astar_robot_radius", 0.35))
            self._planner = AStarPlanner(
                resolution=float(rospy.get_param("~astar_res", 0.15)),
                robot_radius=r_astar,
                obstacles=[{"pos": o["pos"], "radius": o["radius"]}
                           for o in self.coord.obstacles],
                x_min=float(rospy.get_param("~astar_x_min", 0.0)),
                x_max=float(rospy.get_param("~astar_x_max", 10.0)),
                y_min=float(rospy.get_param("~astar_y_min", -5.0)),
                y_max=float(rospy.get_param("~astar_y_max", 5.0)),
                rect_obstacles=self._arena_rects)   # A* respects wall boxes (door etc.)

        self.obs = {i: None for i in self.dogs}
        self.gt = {i: None for i in self.dogs}
        self._v_ema = {i: None for i in self.dogs}   # per-dog EMA of X0 velocity
        self.pub = {}
        self._logged_first = False

        for i in self.dogs:
            ns = "dog%d" % i
            self.pub[i] = rospy.Publisher("/%s/%s_mpc_target" % (ns, ns),
                                          mpc_target_trajectories, queue_size=1)
            rospy.Subscriber("/%s/%s_mpc_observation" % (ns, ns), mpc_observation,
                             self._obs_cb, callback_args=i)
            rospy.Subscriber("/%s/ground_truth/state" % ns, Odometry,
                             self._gt_cb, callback_args=i)
            # live re-targeting (demo tours): publish PoseStamped to /dogN/goal to send the
            # fleet to a new waypoint mid-run; forces an A* re-plan from the current pose.
            rospy.Subscriber("/%s/goal" % ns, PoseStamped, self._goal_cb, callback_args=i)
        # virtual-centroid command (legacy behaviour): publish ONE PoseStamped to
        # /formation/goal = the destination of the V CENTROID; it fans out to per-dog slots.
        rospy.Subscriber("/formation/goal", PoseStamped, self._formation_goal_cb)

        # RViz debug: per-dog ADMM ROLLOUT (predicted trajectory) + slot goals + obstacle
        # circles, all in one MarkerArray. Lets you watch the upper-layer plan, not just the
        # body.
        self.viz = bool(rospy.get_param("~viz_markers", True))
        self.viz_frame = rospy.get_param("~viz_frame", "world")
        self._marker_pub = (rospy.Publisher("/formation/admm_markers", MarkerArray,
                                            queue_size=1) if self.viz else None)
        # let the ADMM publisher OWN the OCS2 target: kill the C++ legged_robot_target nodes
        # that otherwise fight it on /dogN/..._mpc_target (was a manual `rosnode kill` step).
        if bool(rospy.get_param("~kill_cpp_target", False)):
            import rosnode
            targets = ["/dog%d/legged_robot_target" % i for i in self.dogs]
            killed = set()
            for _ in range(20):                     # wait up to ~10s: the C++ targets may
                try:                                # register slightly AFTER us (launch race)
                    live = set(rosnode.get_node_names())
                except Exception:
                    live = set()
                for t in [t for t in targets if t in live and t not in killed]:
                    try:
                        rosnode.kill_nodes([t]); killed.add(t)
                    except Exception as e:
                        rospy.logwarn("[fleet_pub] kill %s failed: %s", t, e)
                if killed == set(targets):
                    break
                rospy.sleep(0.5)
            rospy.loginfo("[fleet_pub] killed C++ target nodes: %s", sorted(killed))

        # combined per-tick log (host reads via the mount) -- per-dog gt + commanded yaw.
        self._csv_path = rospy.get_param("~log_csv", os.path.join(
            _PKG, "docs", "progress", "fleet_track.csv"))
        try:
            self._csv = open(self._csv_path, "w")
            self._csv.write("# fleet rung2 formation=%s w_form=%.1f v=%.3f goals=%s\n"
                            % (self.formation_name, self.w_form, self.v,
                               {i: self.goal[i].tolist() for i in self.dogs}))
            self._csv.write("t," + ",".join("%s%d" % (c, i) for i in self.dogs
                                            for c in self._DBG_COLS) + "\n")
            rospy.on_shutdown(lambda: self._csv and self._csv.close())
        except IOError as e:
            rospy.logwarn("[fleet_pub] cannot open log %s: %s", self._csv_path, e)
            self._csv = None

        rospy.Timer(rospy.Duration(self.ts), self._tick)
        rospy.loginfo("[fleet_pub] dogs=%s formation=%s w_form=%.1f v=%.2f arena=%r "
                      "obstacles=%s goals=%s", self.dogs, self.formation_name,
                      self.w_form, self.v, self.arena_name or "(empty)",
                      [o["pos"] for o in self.coord.obstacles],
                      {i: self.goal[i].tolist() for i in self.dogs})

    def _obs_cb(self, msg, i):
        self.obs[i] = msg

    def _gt_cb(self, msg, i):
        self.gt[i] = msg

    def _goal_cb(self, msg, i):
        """Live re-target dog i (demo tours): update the goal + drop the cached A* path so
        _waypoints re-plans from the current pose next tick; clear the yaw latch to steer."""
        self.goal[i] = np.array([msg.pose.position.x, msg.pose.position.y])
        self._path.pop(i, None)
        if hasattr(self, "_yaw_latch"):
            self._yaw_latch[i] = None
        rospy.loginfo("[fleet_pub] dog%d new goal (%.2f,%.2f)",
                      i, self.goal[i][0], self.goal[i][1])

    def _formation_goal_cb(self, msg):
        """Virtual-centroid command (ported from legacy _goal_slot_targets): ONE
        PoseStamped = where the V CENTROID should go. Fan out to per-dog slots
        (V rotated to face travel dir), assign nearest slot per dog (one-shot, no
        reshuffle on turn), then each dog A*'s to its slot -- find_reachable_goal
        auto-snaps a slot that lands in/near an obstacle to the nearest safe point."""
        if any(self.gt[i] is None for i in self.dogs):
            rospy.logwarn("[fleet_pub] /formation/goal ignored: no state yet")
            return
        goal_c = np.array([msg.pose.position.x, msg.pose.position.y])
        pos = [np.array([self.gt[i].pose.pose.position.x,
                         self.gt[i].pose.pose.position.y]) for i in self.dogs]
        centroid = np.mean(pos, axis=0)
        d = goal_c - centroid
        yaw = (math.atan2(d[1], d[0]) if float(np.linalg.norm(d)) > 0.25
               else self._last_formation_yaw)
        self._last_formation_yaw = yaw
        offsets = self.formation.current_offsets
        if offsets is None or len(offsets) != len(self.dogs):
            rospy.logwarn("[fleet_pub] /formation/goal ignored: no formation offsets")
            return
        slots = centroid_slot_targets(goal_c, offsets, yaw)
        assign = min_cost_assignment(pos, slots)
        for k, i in enumerate(self.dogs):
            self.goal[i] = slots[assign[k]]
            self._path.pop(i, None)                  # force A* re-plan (auto-snaps unsafe slot)
            if hasattr(self, "_yaw_latch"):
                self._yaw_latch[i] = None
        rospy.loginfo("[fleet_pub] /formation/goal centroid (%.2f,%.2f) yaw=%.2f -> %s",
                      goal_c[0], goal_c[1], yaw,
                      {i: self.goal[i].round(2).tolist() for i in self.dogs})

    # --- RViz debug markers (rollout + slots + obstacles) -----------------------------
    _DOG_RGB = {1: (0.90, 0.20, 0.20), 2: (0.20, 0.80, 0.30), 3: (0.20, 0.45, 0.95)}

    def _line_marker(self, ns, mid, pts, rgb, width):
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.lifetime = rospy.Duration(3 * self.ts)    # auto-clear if the publisher stops (no ghost)
        m.ns = ns; m.id = mid; m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = width; m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b = rgb; m.color.a = 1.0
        m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
        return m

    def _sphere_marker(self, ns, mid, xy, rgb, d=0.14):
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.lifetime = rospy.Duration(3 * self.ts)
        m.ns = ns; m.id = mid; m.type = Marker.SPHERE; m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = d; m.pose.orientation.w = 1.0
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), 0.1
        m.color.r, m.color.g, m.color.b = rgb; m.color.a = 1.0
        return m

    @staticmethod
    def _circle_pts(cx, cy, r, n=24):
        return [(cx + r * math.cos(2 * math.pi * k / n),
                 cy + r * math.sin(2 * math.pi * k / n), 0.05) for k in range(n + 1)]

    def _publish_markers(self, wpx, wpy):
        arr = MarkerArray()
        for j, o in enumerate(self.coord.obstacles):          # peg circles (grey)
            cx, cy = o["pos"]
            arr.markers.append(self._line_marker("obstacles", j,
                                                 self._circle_pts(cx, cy, o["radius"]),
                                                 (0.55, 0.55, 0.55), 0.03))
        pos = {i: (self.gt[i].pose.pose.position.x, self.gt[i].pose.pose.position.y)
               for i in self.dogs}
        for i in self.dogs:
            rgb = self._DOG_RGB.get(i, (1.0, 1.0, 1.0))
            pts = [(wpx[i][k], wpy[i][k], 0.1) for k in range(len(wpx[i]))]
            arr.markers.append(self._line_marker("rollout", i, pts, rgb, 0.05))     # predicted path
            arr.markers.append(self._sphere_marker("dog", i, pos[i], rgb, d=0.22))  # current body
            arr.markers.append(self._sphere_marker("slot", i, self.goal[i], rgb, d=0.12))  # V slot
        # formation shape: connect every pair of current dog positions (complete graph =
        # the leaderless formation; N-agnostic, no hardcoded 3-dog indexing)
        edge_pts = []
        for a, b in itertools.combinations(self.dogs, 2):
            edge_pts += [(pos[a][0], pos[a][1], 0.1), (pos[b][0], pos[b][1], 0.1)]
        fm = self._line_marker("formation", 0, edge_pts, (1.0, 0.85, 0.1), 0.03)
        fm.type = Marker.LINE_LIST                            # independent segments, not a strip
        arr.markers.append(fm)
        self._marker_pub.publish(arr)

    def _ready(self):
        return all(self.obs[i] is not None and self.obs[i].time != 0.0
                   and self.gt[i] is not None for i in self.dogs)

    def _x0(self, i):
        """world X0 for dog i: ground_truth position + EMA/clamped obs velocity.
        Returns (X0 (4,), P0 (2,)). Same recipe as the single-dog publisher."""
        s = self.obs[i].state.value
        gp = self.gt[i].pose.pose.position
        P0 = np.array([gp.x, gp.y])
        vraw = np.array([s[ma.MOM_LIN_X], s[ma.MOM_LIN_Y]])        # obs idx0,1 (world)
        self._v_ema[i] = (vraw if self._v_ema[i] is None
                          else 0.25 * vraw + 0.75 * self._v_ema[i])
        vx0 = float(np.clip(self._v_ema[i][0], -C.MAX_VX, C.MAX_VX))
        vy0 = float(np.clip(self._v_ema[i][1], -C.MAX_VY, C.MAX_VY))
        return np.array([P0[0], P0[1], vx0, vy0]), P0

    def _waypoints(self, i, P0i):
        """Reference polyline for dog i: A* curved path (planned once, dense arenas)
        or the straight [current, goal] segment. build_reference samples either."""
        if self._planner is None:
            return [P0i.tolist(), self.goal[i].tolist()]
        if i not in self._path:                          # plan once from the real start
            # find_reachable_goal snaps an occupied/near-obstacle goal to the nearest free
            # cell and returns the path (legacy-stack pattern; robust in cluttered fields).
            cand, p = self._planner.find_reachable_goal(tuple(P0i), tuple(self.goal[i]))
            self._path[i] = p if p else [P0i.tolist(), self.goal[i].tolist()]
            rospy.loginfo("[fleet_pub] dog%d A* waypoints=%d goal->%s%s", i, len(self._path[i]),
                          cand, " (EMPTY->straight fallback)" if not p else "")
        return self._path[i]

    def _tick(self, _evt):
        if not self._ready():
            rospy.logwarn_throttle(2.0, "[fleet_pub] waiting for all 3 obs+gt ...")
            return
        xnow, P0 = {}, {}
        for i in self.dogs:
            xnow[i], P0[i] = self._x0(i)
        xdes = {i: build_reference(P0[i], self._waypoints(i, P0[i]),
                                   v_cruise=self.v) for i in self.dogs}
        try:
            xi, _hist = self.coord.step(xnow, xdes)
        except Exception as e:                       # ponytail: one QP hiccup shouldn't kill the node
            rospy.logwarn_throttle(2.0, "[fleet_pub] coord.step failed: %s", e)
            return

        # NaN guard (the door-arena fall root cause): OSQP returns a non-finite xi when the
        # node/edge QP goes infeasible (e.g. a dog squeezed between an obstacle-CBF and the
        # edge-CBF at a tight spot). A NaN is NOT an exception, so it slips past the try above
        # and a NaN target crashes OCS2's SqpSolver -> the dog falls. Never publish it: hold
        # the last good target this cycle (OCS2 zero-order-holds -> the dog decelerates toward
        # the last safe point instead of toppling). Coordinator couples all 3, so hold all.
        if not all(np.all(np.isfinite(np.asarray(xi[i], float))) for i in self.dogs):
            rospy.logwarn_throttle(1.0, "[fleet_pub] ADMM xi non-finite (QP infeasible) "
                                        "-> HOLD last targets this cycle")
            return

        # world-frame predicted positions per dog (for inter-agent safe-prefix truncation)
        wpx = {i: np.array([xi[i][C.px_index(k, self.N)] for k in range(1, self.N + 1)])
               for i in self.dogs}
        wpy = {i: np.array([xi[i][C.py_index(k, self.N)] for k in range(1, self.N + 1)])
               for i in self.dogs}
        if self._marker_pub is not None:
            self._publish_markers(wpx, wpy)

        dbg = {}
        for i in self.dogs:
            s = self.obs[i].state.value
            xi_i = np.asarray(xi[i], float)
            # world -> dog i's MPC estimate frame: shift every position by (obs - world)
            dx = s[ma.BASE_PX] - P0[i][0]
            dy = s[ma.BASE_PY] - P0[i][1]
            xi_mpc = xi_i.copy()
            for k in range(1, self.N + 1):
                xi_mpc[C.px_index(k, self.N)] += dx
                xi_mpc[C.py_index(k, self.N)] += dy
            # how many leading steps clear D_MIN from the other dogs (world frame)
            others = [(wpx[j], wpy[j]) for j in self.dogs if j != i]
            k_send = ma.safe_prefix_length(wpx[i], wpy[i], others,
                                           self.d_safe, self.send_cap)
            # path yaw, seed = THIS dog's measured yaw (re-anchor each cycle, per rung 1)
            out = self.adapter.build_target(xi_mpc, t0=self.obs[i].time,
                                            seed_yaw=s[ma.BASE_YAW], yaw_mode="path",
                                            k_send=k_send)
            states = [st.tolist() for st in out["states"]]
            # goal-proximity yaw latch (hysteresis) -- freeze heading near the slot so the
            # near-goal path-yaw feedback can't wind up.
            d_goal = float(np.hypot(P0[i][0] - self.goal[i][0], P0[i][1] - self.goal[i][1]))
            if self._yaw_latch[i] is None:
                if d_goal < self.r_latch:
                    self._yaw_latch[i] = s[ma.BASE_YAW]          # freeze the arrival heading
            elif d_goal > self.r_latch + self.latch_margin:
                self._yaw_latch[i] = None                        # left the slot -> steer again
            if self._yaw_latch[i] is not None:
                for st in states:
                    st[ma.BASE_YAW] = self._yaw_latch[i]
            self.pub[i].publish(self._to_msg(out["times"].tolist(), states))
            # boundary instrumentation (which quantity diverges first):
            dbg[i] = dict(estpx=s[ma.BASE_PX], estpy=s[ma.BASE_PY], dx=dx, dy=dy,
                          obsvx=s[ma.MOM_LIN_X], obsvy=s[ma.MOM_LIN_Y],
                          x0vx=xnow[i][2], x0vy=xnow[i][3],
                          tgt1x=states[0][ma.BASE_PX], tgt1y=states[0][ma.BASE_PY],
                          seed=s[ma.BASE_YAW], cmd=states[0][ma.BASE_YAW], ksend=k_send)

        self._log(dbg)

    def _to_msg(self, times, states):
        assert len(times) == len(states), "target time/state length mismatch"
        m = mpc_target_trajectories()
        m.timeTrajectory = times
        m.stateTrajectory = [mpc_state(value=st) for st in states]
        m.inputTrajectory = [mpc_input(value=[0.0] * 24) for _ in states]
        return m

    _DBG_COLS = ["gt_x", "gt_y", "estpx", "estpy", "dx", "dy",
                 "obsvx", "obsvy", "x0vx", "x0vy", "tgt1x", "tgt1y", "seed", "cmd", "ksend"]

    def _log(self, dbg):
        if self._csv is None or self._csv.closed:   # Timer can fire after on_shutdown closes it
            return
        row = ["%.4f" % self.obs[self.dogs[0]].time]
        for i in self.dogs:
            gp = self.gt[i].pose.pose.position
            d = dbg[i]
            vals = [gp.x, gp.y, d["estpx"], d["estpy"], d["dx"], d["dy"],
                    d["obsvx"], d["obsvy"], d["x0vx"], d["x0vy"],
                    d["tgt1x"], d["tgt1y"], d["seed"], d["cmd"], d["ksend"]]
            row += ["%.5f" % v for v in vals]
        self._csv.write(",".join(row) + "\n")
        self._csv.flush()


if __name__ == "__main__":
    rospy.init_node("ocs2_fleet_publisher")
    FleetPublisher()
    rospy.spin()
