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

import numpy as np
import rospy
from ocs2_msgs.msg import (mpc_target_trajectories, mpc_state, mpc_input,
                           mpc_observation)
from nav_msgs.msg import Odometry

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
        # per-dog world-frame goals (arena defaults; override via ~goalN_x / ~goalN_y).
        self.goal = {i: np.array([float(rospy.get_param("~goal%d_x" % i, arena_goals[i][0])),
                                  float(rospy.get_param("~goal%d_y" % i, arena_goals[i][1]))])
                     for i in self.dogs}

        # ONE coupled coordinator: leaderless complete graph. obstacles from the arena
        # (node-local circular CBF); no walls in the open-field basic arena.
        lf = LaplacianFormation(FORMATIONS)
        lf.set_formation(self.formation_name)
        self.coord = ac.ADMMCoordinator(dogs=tuple(self.dogs), edges=tuple(self.edges),
                                        obstacles=arena_obstacles, walls=[],
                                        formation=lf, w_form=self.w_form)
        # shared adapter (per-dog path yaw; seed re-anchored to each dog's MEASURED yaw
        # every cycle -> DON'T use adapter.adapt()'s internal seed, call build_target).
        self.adapter = ma.MotionAdapter(com_height=self.com_height,
                                        default_joints=A1_DEFAULT_JOINTS,
                                        n=self.N, ts=self.ts,
                                        yaw_mode="path", ema_alpha=0.2,
                                        lookahead_M=int(rospy.get_param("~lookahead_m", 5)),
                                        pos_eps=float(rospy.get_param("~pos_eps", 0.02)))

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

    def _tick(self, _evt):
        if not self._ready():
            rospy.logwarn_throttle(2.0, "[fleet_pub] waiting for all 3 obs+gt ...")
            return
        xnow, P0 = {}, {}
        for i in self.dogs:
            xnow[i], P0[i] = self._x0(i)
        xdes = {i: build_reference(P0[i], [P0[i].tolist(), self.goal[i].tolist()],
                                   v_cruise=self.v) for i in self.dogs}
        try:
            xi, _hist = self.coord.step(xnow, xdes)
        except Exception as e:                       # ponytail: one QP hiccup shouldn't kill the node
            rospy.logwarn_throttle(2.0, "[fleet_pub] coord.step failed: %s", e)
            return

        # world-frame predicted positions per dog (for inter-agent safe-prefix truncation)
        wpx = {i: np.array([xi[i][C.px_index(k, self.N)] for k in range(1, self.N + 1)])
               for i in self.dogs}
        wpy = {i: np.array([xi[i][C.py_index(k, self.N)] for k in range(1, self.N + 1)])
               for i in self.dogs}

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
