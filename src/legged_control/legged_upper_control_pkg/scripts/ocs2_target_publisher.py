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
  (rung1+ ADMM/formation modes are added once the chain is proven.)

Run:  rosrun legged_upper_control ocs2_target_publisher.py _dog:=dog1
"""

import os
import sys
import math

import rospy
from ocs2_msgs.msg import (mpc_target_trajectories, mpc_state, mpc_input,
                           mpc_observation)

# rospy-free mapping core (admm/motion_adapter.py). It self-inserts its own paths.
_ADMM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "legged_upper_control", "admm")
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
        self.N, self.ts = C.N, C.TS
        self.obs = None
        self._logged_first = False

        prefix = "/%s/%s" % (self.dog, self.robot_name)
        self.pub = rospy.Publisher(prefix + "_mpc_target",
                                   mpc_target_trajectories, queue_size=1)
        rospy.Subscriber(prefix + "_mpc_observation", mpc_observation, self._obs_cb)
        rospy.Timer(rospy.Duration(self.ts), self._tick)
        rospy.loginfo("[ocs2_target_pub] %s mode=%s v=%.2f pub=%s_mpc_target",
                      self.dog, self.mode, self.v, prefix)

    def _obs_cb(self, msg):
        self.obs = msg
        if not self._logged_first:
            self._logged_first = True
            s = list(msg.state.value)
            rospy.loginfo("[ocs2_target_pub] first obs: time=%.3f state_dim=%d "
                          "px=%.3f py=%.3f yaw=%.4f", msg.time, len(s),
                          s[ma.BASE_PX], s[ma.BASE_PY], s[ma.BASE_YAW])

    def _tick(self, _evt):
        obs = self.obs
        if obs is None or obs.time == 0.0:          # wait for a real observation
            return
        if self.mode == "rung0":
            times, states = self._straight_target(obs)
        else:
            rospy.logwarn_once("[ocs2_target_pub] mode %s not implemented", self.mode)
            return
        self.pub.publish(self._to_msg(times, states))

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

    def _to_msg(self, times, states):
        m = mpc_target_trajectories()
        m.timeTrajectory = times
        m.stateTrajectory = [mpc_state(value=st) for st in states]
        m.inputTrajectory = []          # control not sent (OCS2 recomputes GRF, C6.2)
        return m


if __name__ == "__main__":
    rospy.init_node("ocs2_target_publisher")
    OCS2TargetPublisher()
    rospy.spin()
