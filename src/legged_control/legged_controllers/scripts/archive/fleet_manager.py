#!/usr/bin/env python3
"""
fleet_manager.py
================
讀每隻狗的位置,為各自計算速度指令 → 分開控制。

資料流:
  訂閱 /dogN/ground_truth/state  (每隻狗的位置)
    ↓
  為每隻狗計算到目標的速度指令
    ↓
  發布 /dogN/cmd_vel
"""
import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DogState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.received = False
        self.arrived = False  # 新增:標記是否已抵達,避免反覆 log


def quaternion_to_yaw(q):
    """把 quaternion 轉成 yaw 角"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FleetManager:
    def __init__(self):
        rospy.init_node("fleet_manager")

        self.dog_names = ["dog1", "dog2", "dog3", "dog4", "dog5"]

        # 控制參數
        self.kp_linear = 0.5
        self.kp_angular = 1.0
        self.goal_tolerance = 0.3
        self.max_linear = 0.5
        self.max_angular = 1.0

        self.states = {}
        self.cmd_pubs = {}
        self.goals = {}

        # ===== 修正 1:分兩段初始化,避免 race condition =====
        # 先建立所有 state dict 和 publisher
        for name in self.dog_names:
            self.states[name] = DogState()
            self.goals[name] = None
            self.cmd_pubs[name] = rospy.Publisher(
                f"/{name}/cmd_vel", Twist, queue_size=1
            )

        # 等全部 dict 建好後再註冊 subscriber(此時不會有 KeyError 風險)
        for name in self.dog_names:
            rospy.Subscriber(
                f"/{name}/ground_truth/state",
                Odometry,
                self._odom_callback,
                callback_args=name,
            )

        rospy.sleep(1.0)
        rospy.loginfo(f"[FleetManager] Managing {len(self.dog_names)} dogs")

    def _odom_callback(self, msg, dog_name):
        s = self.states[dog_name]
        s.x = msg.pose.pose.position.x
        s.y = msg.pose.pose.position.y
        s.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        s.received = True

    def set_goal(self, dog_name, goal_x, goal_y):
        self.goals[dog_name] = (goal_x, goal_y)
        self.states[dog_name].arrived = False  # 重設抵達狀態
        rospy.loginfo(f"[FleetManager] {dog_name} → ({goal_x}, {goal_y})")

    def set_formation_goal(self, center_x, center_y):
        """設定整個編隊的目標中心點,各隻狗保持 V 字型偏移"""
        offsets = {
            "dog1": (-2.0,  3.0),
            "dog2": (-1.0,  1.5),
            "dog3": ( 0.0,  0.0),   # 領頭
            "dog4": (-1.0, -1.5),
            "dog5": (-2.0, -3.0),
        }
        for name in self.dog_names:
            if name in offsets:
                dx, dy = offsets[name]
                self.set_goal(name, center_x + dx, center_y + dy)

    def _compute_cmd(self, dog_name):
        """計算某隻狗的速度指令"""
        s = self.states[dog_name]
        goal = self.goals.get(dog_name)
        cmd = Twist()

        if goal is None or not s.received:
            return cmd

        dx = goal[0] - s.x
        dy = goal[1] - s.y
        distance = math.sqrt(dx * dx + dy * dy)

        # ===== 修正 2:抵達後只 log 一次 =====
        if distance < self.goal_tolerance:
            if not s.arrived:
                rospy.loginfo(f"[FleetManager] {dog_name} arrived!")
                s.arrived = True
            return cmd  # 停止

        # 若重新偏離,清除 arrived flag
        s.arrived = False

        # ===== 修正 3:角度歸一化用數學運算 =====
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - s.yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        # 比例控制
        cmd.linear.x = min(self.kp_linear * distance, self.max_linear)
        cmd.angular.z = max(-self.max_angular,
                            min(self.kp_angular * yaw_error, self.max_angular))

        # 角度差太大時先原地轉,避免繞大圈
        if abs(yaw_error) > 0.5:
            cmd.linear.x = 0.0

        return cmd

    def all_arrived(self):
        """全部狗都抵達了嗎?(為未來 CBBA 串接用)"""
        return all(s.arrived for s in self.states.values())

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            for name in self.dog_names:
                cmd = self._compute_cmd(name)
                self.cmd_pubs[name].publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    try:
        fm = FleetManager()
        # 讓五隻狗 V 字型整體往 (10, 0) 移動
        fm.set_formation_goal(center_x=10.0, center_y=0.0)
        fm.spin()
    except rospy.ROSInterruptException:
        pass