#!/usr/bin/env python3
"""
gait_broadcaster.py
===================
繞過 GaitKeyboardPublisher 的鍵盤輸入,直接發 mode_schedule 給多隻狗。
在單一 terminal 一次切換所有狗的步態(鍵盤的 stdin 只會給其中一隻狗)。

用法:
  python3 gait_broadcaster.py trot          # 切到 trot
  python3 gait_broadcaster.py stance        # 切到 stance
  python3 gait_broadcaster.py trot dog1     # 只切 dog1

執行前提:
  1. Gazebo + fleet_bringup 已啟動
  2. 每隻狗的 legged_controller 已 switch_controller start
  3. OCS2 的 GaitReceiver.cpp 已移除 TransportHints().udp() 並重編
"""
import sys
import time
import rospy
from ocs2_msgs.msg import mode_schedule


# gait.info 對應表(值來自 legged_controllers/config/a1/gait.info)
GAITS = {
    "stance": {"times": [0.0, 0.5],      "modes": [15]},
    "trot":   {"times": [0.0, 0.3, 0.6], "modes": [9, 6]},
}


def main():
    args = sys.argv[1:]
    gait_name = "trot"
    dogs = ["dog1", "dog2", "dog3"]

    if args and args[0] in GAITS:
        gait_name = args[0]
        if len(args) > 1:
            dogs = args[1:]
    elif args:
        print(f"[ERROR] Unknown gait '{args[0]}'. Available: {list(GAITS.keys())}")
        sys.exit(1)

    print(f"[gait_broadcaster] gait={gait_name}, dogs={dogs}")

    rospy.init_node("gait_broadcaster", anonymous=True)

    # latch=True 與 GaitKeyboardPublisher 行為一致
    pubs = {}
    for ns in dogs:
        topic = f"/{ns}/{ns}_mpc_mode_schedule"
        pubs[ns] = rospy.Publisher(topic, mode_schedule, queue_size=1, latch=True)
        print(f"[gait_broadcaster] Publisher created: {topic}")

    # 等 ROS discovery + TCP 連線建立
    print("[gait_broadcaster] Waiting 3s for connection...")
    time.sleep(3.0)

    for ns, pub in pubs.items():
        print(f"  {ns}: connections={pub.get_num_connections()}")

    gait = GAITS[gait_name]
    msg = mode_schedule()
    msg.eventTimes = gait["times"]
    msg.modeSequence = gait["modes"]

    print(f"[gait_broadcaster] Publishing mode_schedule:")
    print(f"  eventTimes:   {msg.eventTimes}")
    print(f"  modeSequence: {msg.modeSequence}")

    # 發 10 秒,確保 GaitReceiver 收到
    print("[gait_broadcaster] Publishing for 10 seconds...")
    start = time.time()
    count = 0
    while time.time() - start < 10.0:
        for ns, pub in pubs.items():
            pub.publish(msg)
        count += 1
        time.sleep(0.2)

    print(f"[gait_broadcaster] Total {count} rounds. Done.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n[gait_broadcaster] Interrupted.")