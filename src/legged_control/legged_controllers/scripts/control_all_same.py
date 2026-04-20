#!/usr/bin/env python3
"""
control_all_same.py
===================
把同一個 cmd_vel 複製五份發給五隻狗。
相當於「一起控制」,五隻狗同方向同速度。
"""
import rospy
from geometry_msgs.msg import Twist

rospy.init_node("control_all_same")

DOGS = ["dog1", "dog2", "dog3", "dog4", "dog5"]

pubs = []
for name in DOGS:
    topic = f"/{name}/cmd_vel"  # 絕對路徑,直接寫 /dog1/cmd_vel
    pub = rospy.Publisher(topic, Twist, queue_size=1)
    pubs.append(pub)
    rospy.loginfo(f"Publisher ready: {topic}")

rospy.sleep(1.0)

cmd = Twist()
cmd.linear.x = 0.0
cmd.angular.z = 0.0

rate = rospy.Rate(10)
rospy.loginfo("Publishing same cmd to all 5 dogs. Press Ctrl+C to stop.")

while not rospy.is_shutdown():
    for pub in pubs:
        pub.publish(cmd)
    rate.sleep()