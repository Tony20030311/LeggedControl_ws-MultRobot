"""
機器人狀態收集 — 訂閱每隻狗的 ground truth，
將 (position, yaw, velocity) 存入 RobotState。
"""

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from .geometry import quaternion_to_yaw, rot2d


class RobotState:
    __slots__ = ("x", "y", "yaw", "vx_world", "vy_world", "received")

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx_world = 0.0
        self.vy_world = 0.0
        self.received = False

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def vel_world(self):
        return np.array([self.vx_world, self.vy_world])


class StateCollector:
    def __init__(self, dog_names):
        self.states = {name: RobotState() for name in dog_names}
        self._subs = []
        for name in dog_names:
            sub = rospy.Subscriber(
                f"/{name}/ground_truth/state",
                Odometry,
                self._odom_cb,
                callback_args=name,
                queue_size=1,
            )
            self._subs.append(sub)

    def _odom_cb(self, msg, name):
        s = self.states[name]
        s.x = msg.pose.pose.position.x
        s.y = msg.pose.pose.position.y
        s.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        R = rot2d(s.yaw)
        v_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        v_world = R @ v_body
        s.vx_world = v_world[0]
        s.vy_world = v_world[1]
        s.received = True

    def all_received(self, names):
        return all(self.states[n].received for n in names)


# ═══════════════════════════════════════════════════════════════
# Module A: AStarPlanner（不動）
# ═══════════════════════════════════════════════════════════════
