#!/usr/bin/env python3
"""Stage 4 wiring -- thin ROS shell that publishes an OCS2 target trajectory.

This is the ONLY rospy layer of the ADMM->OCS2 handoff; the mapping/yaw math lives
in the rospy-free, C++-translatable admm/motion_adapter.py and is NOT touched here.

Chain:  observation (time + measured base yaw) -> target trajectory -> OCS2 NMPC.
  sub  /<dog>/legged_robot_mpc_observation  (ocs2_msgs/mpc_observation)
  pub  /<dog>/legged_robot_mpc_target       (ocs2_msgs/mpc_target_trajectories)

Modes:
  rung0 (default) -- a hand-made TRIVIAL target: walk forward along the dog's own
        measured heading at a constant speed. NO ADMM, NO formation -> if the dog
        does not move, it is 100% a chain problem, not the controller logic.
  arc   (rung 0.5) -- a hand-made CONSTANT-CURVATURE arc ξ, routed through the SAME
        adapter path rung 1 uses. NO ADMM: isolates OCS2 shape-tracking.
  rung1 -- single-dog ADMM node QP to a WORLD goal. WORLD position from ground_truth
        (the Kalman estimator base position is unobservable -> ~0); the ξ is re-expressed
        in the MPC estimate frame (anchor + displacement) before publishing.

Run:  rosrun legged_upper_control ocs2_target_publisher.py _dog:=dog1
      rosrun legged_upper_control ocs2_target_publisher.py _dog:=dog1 _mode:=arc
      rosrun legged_upper_control ocs2_target_publisher.py _dog:=dog1 _mode:=rung1 \
             _goal_x:=4.0 _goal_y:=1.5
"""

import os
import sys
import math

import numpy as np
import rospy
from ocs2_msgs.msg import (mpc_target_trajectories, mpc_state, mpc_input,
                           mpc_observation)
from nav_msgs.msg import Odometry     # /<dog>/ground_truth/state (world localisation)

# rospy-free mapping core (admm/motion_adapter.py). It self-inserts its own paths.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ADMM = os.path.join(_PKG, "legged_upper_control", "admm")
sys.path.insert(0, _ADMM)
import motion_adapter as ma   # noqa: E402
import constants as C          # noqa: E402

# A1 standing pose (placeholder; load reference.info defaultJointState at rung 1+).
A1_DEFAULT_JOINTS = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                     0.1, 1.0, -1.5, -0.1, 1.0, -1.5]


class OCS2TargetPublisher:
    def __init__(self):
        self.dog = rospy.get_param("~dog", "dog1")
        # this workspace's load_controller_multi.launch sets the OCS2 robot_name to
        # the namespace (ns=dog1), so topics are /dog1/dog1_mpc_* -- NOT legged_robot.
        # Overridable via _robot_name if a setup uses the OCS2 default.
        self.robot_name = rospy.get_param("~robot_name", self.dog)
        self.mode = rospy.get_param("~mode", "rung0")
        self.v = float(rospy.get_param("~v", 0.15))            # forward speed [m/s]
        self.com_height = float(rospy.get_param("~com_height", 0.30))
        self.arc_radius = float(rospy.get_param("~arc_radius", 4.0))  # canned-arc R [m]
        self.arc_sign = float(rospy.get_param("~arc_sign", 1.0))      # +1 CCW / -1 CW
        # rung1 diagnostics (A/B/C matrix): X0 velocity source + yaw generation
        self.x0_vel = rospy.get_param("~x0_vel", "obs")          # obs | gt | plan
        # yaw_mode: path (Route B, position-lookahead -- default, kills the orbit) |
        #           tangent (legacy velocity direction, the buggy one) | frozen (isolation)
        self.yaw_mode = rospy.get_param("~yaw_mode", "path")
        self.lookahead_m = int(rospy.get_param("~lookahead_m", 5))    # path yaw lookahead steps
        self.pos_eps = float(rospy.get_param("~pos_eps", 0.02))       # near-goal freeze threshold [m]
        self._yaw0 = None                                        # frozen-yaw capture (rung1)
        # goal-proximity yaw latch (same fix as the fleet publisher): near the goal the
        # path-lookahead direction is noise, so the measured-yaw-seeded yaw winds up
        # unboundedly (dog spins). Latch the arrival heading within r, release past r+margin.
        self.yaw_latch_r = float(rospy.get_param("~yaw_latch_r", 0.25))
        self.yaw_latch_margin = float(rospy.get_param("~yaw_latch_margin", 0.10))
        self._yaw_latch = None                                   # held heading, or None
        self.N, self.ts = C.N, C.TS
        self.obs = None
        self._logged_first = False
        self._arc_anchor = None       # (t0, px0, py0, yaw0) frozen on the first arc tick
        # rung1: WORLD goal; position comes from ground_truth (estimator idx6,7 is
        # unobservable -> stays ~0), only velocity (idx0,1) + yaw (idx9) from the obs.
        self.goal = np.array([float(rospy.get_param("~goal_x", 4.0)),
                              float(rospy.get_param("~goal_y", 1.5))])
        self.gt = None                # latest /<dog>/ground_truth/state
        self._node = None             # lazy ADMM node QP (rung1)
        self._prev_xi = None          # receding-horizon warm start
        self._v_ema = None            # EMA-filtered X0 velocity (trot COM velocity swings
                                      # ±0.2 within a stride; raw instantaneous velocity fed
                                      # to the QP flips the plan/yaw -> filter it, per the
                                      # open-source reference which also filters vx,vy)

        # shared downstream: hand-made (arc) OR ADMM ξ -> same adapter -> same target.
        self.adapter = ma.MotionAdapter(com_height=self.com_height,
                                        default_joints=A1_DEFAULT_JOINTS,
                                        n=self.N, ts=self.ts,
                                        v_freeze=0.10,   # freeze yaw below 0.1 m/s (tangent
                                                         # mode: decel near goal -> no atan2 noise)
                                        ema_alpha=0.2,   # smoother yaw
                                        yaw_mode="path", # Route B default (per-call overridable)
                                        lookahead_M=self.lookahead_m,
                                        pos_eps=self.pos_eps)
        # tracking-fidelity log: observed base pose per tick (host reads via the mount).
        self._csv_path = rospy.get_param("~log_csv", os.path.join(
            _PKG, "docs", "progress", "rung_track.csv"))
        try:
            self._csv = open(self._csv_path, "w")
            self._csv.write("# mode=%s v=%.3f arc_radius=%.3f arc_sign=%+.0f goal=%.2f,%.2f "
                            "x0_vel=%s yaw_mode=%s\n"
                            % (self.mode, self.v, self.arc_radius, self.arc_sign,
                               self.goal[0], self.goal[1], self.x0_vel, self.yaw_mode))
            self._csv.write("t,px,py,yaw,gt_x,gt_y,obs_vx,obs_vy,cmd_vx,cmd_vy,cmd_yaw\n")
            rospy.on_shutdown(lambda: self._csv and self._csv.close())
        except IOError as e:
            rospy.logwarn("[ocs2_target_pub] cannot open log %s: %s", self._csv_path, e)
            self._csv = None

        prefix = "/%s/%s" % (self.dog, self.robot_name)
        self.pub = rospy.Publisher(prefix + "_mpc_target",
                                   mpc_target_trajectories, queue_size=1)
        rospy.Subscriber(prefix + "_mpc_observation", mpc_observation, self._obs_cb)
        rospy.Subscriber("/%s/ground_truth/state" % self.dog, Odometry, self._gt_cb)
        rospy.Timer(rospy.Duration(self.ts), self._tick)
        rospy.loginfo("[ocs2_target_pub] %s mode=%s v=%.2f x0_vel=%s yaw_mode=%s pub=%s_mpc_target",
                      self.dog, self.mode, self.v, self.x0_vel, self.yaw_mode, prefix)

    def _obs_cb(self, msg):
        self.obs = msg
        if not self._logged_first:
            self._logged_first = True
            s = list(msg.state.value)
            rospy.loginfo("[ocs2_target_pub] first obs: time=%.3f state_dim=%d "
                          "px=%.3f py=%.3f yaw=%.4f", msg.time, len(s),
                          s[ma.BASE_PX], s[ma.BASE_PY], s[ma.BASE_YAW])

    def _gt_cb(self, msg):
        self.gt = msg

    def _tick(self, _evt):
        obs = self.obs
        if obs is None or obs.time == 0.0:          # wait for a real observation
            return
        if self.mode == "rung0":
            times, states = self._straight_target(obs)
        elif self.mode == "arc":
            times, states = self._arc_target(obs)
        elif self.mode == "rung1":
            res = self._rung1_target(obs)
            if res is None:
                return
            times, states = res
        else:
            rospy.logwarn_once("[ocs2_target_pub] mode %s not implemented", self.mode)
            return
        self.pub.publish(self._to_msg(times, states))
        self._log(obs, states)

    def _straight_target(self, obs):
        """Walk forward along the dog's OWN measured heading -> zero turn command."""
        s = obs.state.value
        px0, py0, yaw = s[ma.BASE_PX], s[ma.BASE_PY], s[ma.BASE_YAW]
        c, sn = math.cos(yaw), math.sin(yaw)
        times, states = [], []
        for k in range(1, self.N + 1):
            d = self.v * k * self.ts
            st = ma.pad_ocs2_state(px0 + d * c, py0 + d * sn,   # position along heading
                                   self.v * c, self.v * sn,     # world-frame velocity
                                   yaw, self.com_height, A1_DEFAULT_JOINTS)
            times.append(obs.time + k * self.ts)                # future (times[0] > obs.time)
            states.append(st.tolist())
        return times, states

    def _arc_xi(self, obs):
        """Canned constant-curvature arc packed as an ADMM ξ (states + velocities
        filled from the arc; accel slots left 0). Same layout the ADMM solver emits
        -> the downstream (build_target) is byte-identical to rung 1.

        FIXED-FRAME, TIME-PARAMETRIZED (anchored ONCE on the first arc tick). The
        earlier "re-anchor to the measured pose every cycle" version spun in place:
        a rotating-yaw reference re-drawn from the dog's own (rotating) heading each
        cycle is a positive feedback loop -> the dog chases its tail. Here the arc is
        frozen in the world at anchor (t0,px0,py0,yaw0) and only marched forward by
        elapsed sim time -- no measured-pose feedback, so the shape is a clean target.

          s(t)  = v*(obs.time - t0)           arc length the reference has advanced
          s_k   = s(t) + v*k*ts               arc length at horizon step k
          th_k  = yaw0 + kappa*s_k            heading (tangent) along the arc
          px_k  = px0 + (sin th_k - sin yaw0)/kappa
          py_k  = py0 - (cos th_k - cos yaw0)/kappa   (kappa->0 falls back to straight)
        """
        if self._arc_anchor is None:                     # freeze the anchor once
            s = obs.state.value
            self._arc_anchor = (obs.time, s[ma.BASE_PX], s[ma.BASE_PY], s[ma.BASE_YAW])
        t0, px0, py0, yaw0 = self._arc_anchor
        kappa = self.arc_sign / self.arc_radius
        s_now = self.v * max(0.0, obs.time - t0)         # advance the reference in time
        xi = np.zeros(C.xi_dim(self.N))
        for k in range(1, self.N + 1):
            arc = s_now + self.v * k * self.ts           # arc length at horizon step k
            th = yaw0 + kappa * arc                      # heading (tangent) along the arc
            if abs(kappa) > 1e-9:
                px = px0 + (math.sin(th) - math.sin(yaw0)) / kappa
                py = py0 - (math.cos(th) - math.cos(yaw0)) / kappa
            else:                                        # kappa -> 0 : straight line
                px = px0 + arc * math.cos(yaw0)
                py = py0 + arc * math.sin(yaw0)
            xi[C.px_index(k, self.N)] = px
            xi[C.py_index(k, self.N)] = py
            xi[C.vx_index(k, self.N)] = self.v * math.cos(th)
            xi[C.vy_index(k, self.N)] = self.v * math.sin(th)
        return xi

    def _arc_target(self, obs):
        """rung 0.5: hand-made arc ξ -> the SAME adapter path rung 1 (ADMM ξ) uses.
        seed_yaw is the arc tangent at the current window start (fixed schedule, not
        the measured heading) so the yaw bypass tracks the arc without feedback."""
        xi = self._arc_xi(obs)
        t0, px0, py0, yaw0 = self._arc_anchor
        kappa = self.arc_sign / self.arc_radius
        seed_yaw = yaw0 + kappa * (self.v * max(0.0, obs.time - t0))
        # arc yaw stays the clean velocity tangent (its verified behavior); Route B path
        # mode is for the ADMM rung-1 trajectory, not this hand-made schedule.
        out = self.adapter.build_target(xi, t0=obs.time, seed_yaw=seed_yaw,
                                        yaw_mode="tangent")
        return out["times"].tolist(), [st.tolist() for st in out["states"]]

    def _ensure_admm(self):
        """Lazy-load the rospy-free ADMM node QP + reference builder (rung1 only)."""
        if self._node is not None:
            return
        import node_subproblem as nq          # noqa: E402  (admm dir on sys.path)
        from reference import build_reference  # noqa: E402
        self._build_reference = build_reference
        self._node = nq.NodeSubproblem(obstacles=[], walls=[], robot_margin=0.30,
                                       q_pos=10.0, q_v=1.0, r_accel=0.5, w_pred=20.0)

    def _shift_xi(self, xi):
        """Receding-horizon warm start: previous solution shifted one step."""
        n = self.N
        out = np.zeros(C.xi_dim(n))
        for k in range(1, n + 1):
            src = min(k + 1, n)
            out[C.x_index(k, n):C.x_index(k, n) + 4] = \
                xi[C.x_index(src, n):C.x_index(src, n) + 4]
        for k in range(n):
            src = min(k + 1, n - 1)
            out[C.a_index(k, n):C.a_index(k, n) + 2] = \
                xi[C.a_index(src, n):C.a_index(src, n) + 2]
        return out

    def _rung1_target(self, obs):
        """rung 1: single-dog ADMM node QP to a WORLD goal.

        The Kalman estimator base position (obs idx6,7) is UNOBSERVABLE (stays ~0), so
        the WORLD position comes from ground_truth; only velocity (idx0,1) and yaw
        (idx9) are taken from the observation. The node QP plans a world trajectory that
        decelerates into the goal; the ξ positions are then re-expressed in the MPC's
        estimate frame (anchor at obs + displacement-from-current-world) so the target
        reads as 'go this far from where you are'. As the dog nears the goal the
        displacement -> 0 -> target velocity -> 0 -> it STOPS (no perpetual drift)."""
        if self.gt is None:
            rospy.logwarn_throttle(2.0, "[rung1] waiting for ground_truth ...")
            return None
        self._ensure_admm()
        s = obs.state.value
        gp = self.gt.pose.pose.position
        P0 = np.array([gp.x, gp.y])                          # WORLD position (ground_truth)
        # velocity from the obs (idx0,1, world) -- but the trot COM velocity swings ±0.2
        # within a stride, so EMA-filter it (raw instantaneous velocity flips the plan/yaw,
        # observed as a -319deg yaw blow-up during decel). Then CLAMP to the QP box so a
        # fast X0 can still decelerate within one step (else the QP is infeasible).
        # X0 velocity source (diagnostic switch; ALL world-frame, no rotation):
        #   obs  -> obs.state[0,1] EMA (estimator velocity: trot sway + estimator noise)
        #   gt   -> ground_truth twist EMA (world; frameName=world -> no sensor noise, still swings)
        #   plan -> previous ξ planned velocity at k=1 (smoothest, open-loop; cold-starts on obs)
        if self.x0_vel == "plan" and self._prev_xi is not None:
            vx0 = float(self._prev_xi[C.vx_index(1, self.N)])
            vy0 = float(self._prev_xi[C.vy_index(1, self.N)])
        else:
            if self.x0_vel == "gt":
                gv = self.gt.twist.twist.linear
                vraw = np.array([gv.x, gv.y])
            else:                                    # obs (default) or plan cold-start
                vraw = np.array([s[ma.MOM_LIN_X], s[ma.MOM_LIN_Y]])
            self._v_ema = vraw if self._v_ema is None else 0.25 * vraw + 0.75 * self._v_ema
            vx0, vy0 = self._v_ema[0], self._v_ema[1]
        vx0 = float(np.clip(vx0, -C.MAX_VX, C.MAX_VX))
        vy0 = float(np.clip(vy0, -C.MAX_VY, C.MAX_VY))
        X0 = np.array([P0[0], P0[1], vx0, vy0])
        x_des = self._build_reference(P0, [P0.tolist(), self.goal.tolist()],
                                      v_cruise=self.v)
        res = self._node.solve(X0, x_des, xbar=self._prev_xi)
        if not res["status"].startswith("solved") or res.get("xi") is None:
            rospy.logwarn_throttle(2.0, "[rung1] QP status=%s", res["status"])
            return None
        xi = res["xi"]
        self._prev_xi = self._shift_xi(xi)
        # world -> MPC estimate frame: shift every position by (obs_anchor - world_start)
        dx = s[ma.BASE_PX] - P0[0]
        dy = s[ma.BASE_PY] - P0[1]
        xi_mpc = xi.copy()
        for k in range(1, self.N + 1):
            xi_mpc[C.px_index(k, self.N)] += dx
            xi_mpc[C.py_index(k, self.N)] += dy
        # Route B: path-lookahead yaw (default). 'tangent' = legacy velocity yaw for
        # A/B; 'frozen' overwrites the states below so its underlying mode is moot.
        underlying = "tangent" if self.yaw_mode == "tangent" else "path"
        out = self.adapter.build_target(xi_mpc, t0=obs.time, seed_yaw=s[ma.BASE_YAW],
                                        yaw_mode=underlying)
        states = [st.tolist() for st in out["states"]]
        if self.yaw_mode == "frozen":
            # holonomic isolation test: hold the heading captured on the first rung1 tick.
            # velocity idx0,1 stays as planned (world) -> the base STRAFES to follow the
            # position curve without turning. If the orbit dies here, yaw coupling is the cause.
            if self._yaw0 is None:
                self._yaw0 = s[ma.BASE_YAW]
            for st in states:
                st[ma.BASE_YAW] = self._yaw0
        elif self.yaw_mode == "path":
            # goal-proximity yaw latch (hysteresis) -- freeze heading near the goal so the
            # near-goal path-yaw feedback can't wind up (see fleet publisher for the trace).
            d_goal = float(np.hypot(P0[0] - self.goal[0], P0[1] - self.goal[1]))
            if self._yaw_latch is None:
                if d_goal < self.yaw_latch_r:
                    self._yaw_latch = s[ma.BASE_YAW]
            elif d_goal > self.yaw_latch_r + self.yaw_latch_margin:
                self._yaw_latch = None
            if self._yaw_latch is not None:
                for st in states:
                    st[ma.BASE_YAW] = self._yaw_latch
        return out["times"].tolist(), states

    def _log(self, obs, states):
        """Per-tick diagnostic: observed pose (obs px,py is the unobservable estimator
        position ~0), ground_truth world pos (what matters for the goal), the measured X0
        velocity (idx0,1), and the COMMANDED first-point velocity + yaw -- so we SEE the
        commanded direction/jerk instead of inferring it."""
        if self._csv is None:
            return
        s = obs.state.value
        gx = gy = float("nan")
        if self.gt is not None:
            gx, gy = self.gt.pose.pose.position.x, self.gt.pose.pose.position.y
        c = states[0]                      # commanded first target point (24-dim)
        self._csv.write("%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n"
                        % (obs.time, s[ma.BASE_PX], s[ma.BASE_PY], s[ma.BASE_YAW], gx, gy,
                           s[ma.MOM_LIN_X], s[ma.MOM_LIN_Y],           # measured X0 velocity
                           c[ma.MOM_LIN_X], c[ma.MOM_LIN_Y], c[ma.BASE_YAW]))  # commanded
        self._csv.flush()

    def _to_msg(self, times, states):
        # OCS2 target contract (getDesiredState = linear interp + boundary CLAMP):
        #  - time/state/input EQUAL length; - times ABSOLUTE (first > obs.time);
        #  - span >= mpc.timeHorizon (1.0s; ours N*ts = 2.0s so the whole horizon is
        #    covered, no clamp-to-endpoint contradiction). The legged cost ignores the
        #    target input (uNominal = weightCompensatingInput) but we send equal-length
        #    zeros to honour the contract instead of an empty array.
        assert len(times) == len(states), "target time/state length mismatch"
        m = mpc_target_trajectories()
        m.timeTrajectory = times
        m.stateTrajectory = [mpc_state(value=st) for st in states]
        m.inputTrajectory = [mpc_input(value=[0.0] * 24) for _ in states]  # legged input dim
        return m


if __name__ == "__main__":
    rospy.init_node("ocs2_target_publisher")
    OCS2TargetPublisher()
    rospy.spin()
