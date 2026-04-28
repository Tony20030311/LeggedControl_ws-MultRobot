#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DogState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.received = False


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FormationManager:
    def __init__(self):
        rospy.init_node("formation_manager")

        self.leader_name = rospy.get_param("~leader_name", "dog1")
        self.follower_names = rospy.get_param("~follower_names", ["dog2", "dog3"])
        self.all_dogs = [self.leader_name] + self.follower_names

        self.kp_x = rospy.get_param("~kp_x", 0.8)
        self.kp_y = rospy.get_param("~kp_y", 0.8)
        self.kp_yaw = rospy.get_param("~kp_yaw", 1.0)

        self.max_vx = rospy.get_param("~max_vx", 0.5)
        self.max_vy = rospy.get_param("~max_vy", 0.3)
        self.max_wz = rospy.get_param("~max_wz", 0.8)

        self.pos_tolerance = rospy.get_param("~pos_tolerance", 0.15)
        self.yaw_tolerance = rospy.get_param("~yaw_tolerance", 0.15)
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.stop_without_leader = rospy.get_param("~stop_without_leader", True)

        default_offsets = {
            "dog2": [-1.0, 1.0],
            "dog3": [-1.0, -1.0],
        }
        self.offsets = rospy.get_param("~offsets", default_offsets)

        self.states = {}
        self.cmd_pubs = {}

        for name in self.all_dogs:
            self.states[name] = DogState()

        for name in self.follower_names:
            self.cmd_pubs[name] = rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=1)

        for name in self.all_dogs:
            rospy.Subscriber(
                f"/{name}/ground_truth/state",
                Odometry,
                self._odom_callback,
                callback_args=name,
                queue_size=1,
            )

        rospy.sleep(1.0)
        rospy.loginfo("[FormationManager] leader=%s, followers=%s", self.leader_name, self.follower_names)
        rospy.loginfo("[FormationManager] offsets=%s", self.offsets)

    def _odom_callback(self, msg, dog_name):
        s = self.states[dog_name]
        s.x = msg.pose.pose.position.x
        s.y = msg.pose.pose.position.y
        s.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        s.received = True

    def _leader_target_world(self, follower_name):
        leader = self.states[self.leader_name]
        offset = self.offsets.get(follower_name, [-1.0, 0.0])
        dx, dy = float(offset[0]), float(offset[1])

        x_ref = leader.x + math.cos(leader.yaw) * dx - math.sin(leader.yaw) * dy
        y_ref = leader.y + math.sin(leader.yaw) * dx + math.cos(leader.yaw) * dy
        yaw_ref = leader.yaw
        return x_ref, y_ref, yaw_ref

    def _compute_cmd(self, follower_name):
        cmd = Twist()

        leader = self.states[self.leader_name]
        follower = self.states[follower_name]

        if not follower.received:
            return cmd

        if not leader.received:
            if self.stop_without_leader:
                return cmd
            return cmd

        x_ref, y_ref, yaw_ref = self._leader_target_world(follower_name)

        ex_world = x_ref - follower.x
        ey_world = y_ref - follower.y
        yaw_error = wrap_to_pi(yaw_ref - follower.yaw)

        # world -> follower body frame
        ex_body = math.cos(follower.yaw) * ex_world + math.sin(follower.yaw) * ey_world
        ey_body = -math.sin(follower.yaw) * ex_world + math.cos(follower.yaw) * ey_world

        pos_error = math.hypot(ex_world, ey_world)
        if pos_error < self.pos_tolerance and abs(yaw_error) < self.yaw_tolerance:
            return cmd

        cmd.linear.x = max(-self.max_vx, min(self.kp_x * ex_body, self.max_vx))
        cmd.linear.y = max(-self.max_vy, min(self.kp_y * ey_body, self.max_vy))
        cmd.angular.z = max(-self.max_wz, min(self.kp_yaw * yaw_error, self.max_wz))

        return cmd

    def _publish_zero_to_all_followers(self):
        zero = Twist()
        for name in self.follower_names:
            self.cmd_pubs[name].publish(zero)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.stop_without_leader and not self.states[self.leader_name].received:
                self._publish_zero_to_all_followers()
                rate.sleep()
                continue

            for name in self.follower_names:
                cmd = self._compute_cmd(name)
                self.cmd_pubs[name].publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    try:
        manager = FormationManager()
        manager.spin()
    except rospy.ROSInterruptException:
        pass