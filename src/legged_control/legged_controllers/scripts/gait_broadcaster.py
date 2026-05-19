#!/usr/bin/env python3
"""
gait_broadcaster.py
===================
繞過 GaitKeyboardPublisher 的鍵盤輸入,直接發 mode_schedule 給多隻狗。
在單一 terminal 一次切換所有狗的步態(鍵盤的 stdin 只會給其中一隻狗)。

用法:
  python3 gait_broadcaster.py trot           # 切到 trot
  python3 gait_broadcaster.py trot dog1      # 只切 dog1
  python3 gait_broadcaster.py stance         # 切到 stance
  python3 gait_broadcaster.py stance --repeat --duration 1.0
  python3 gait_broadcaster.py standing_trot  # 診斷用,會交替對角接觸

執行前提:
  1. Gazebo + fleet_bringup 已啟動
  2. 每隻狗的 legged_controller 已 switch_controller start
  3. OCS2 的 GaitReceiver.cpp 已移除 TransportHints().udp() 並重編
"""
import argparse
import math
import time
import rospy
from ocs2_msgs.msg import mode_schedule


# gait.info 對應表(值來自 legged_controllers/config/a1/gait.info)
GAITS = {
    "stance":        {"times": [0.0, 0.5],                         "modes": [15]},
    "standing_trot": {"times": [0.0, 0.25, 0.30, 0.55, 0.60],      "modes": [9, 15, 6, 15]},
    "trot":          {"times": [0.0, 0.3, 0.6],                    "modes": [9, 6]},
}


def main():
    parser = argparse.ArgumentParser(
        description="Broadcast one OCS2 mode_schedule to one or more dogs.")
    parser.add_argument(
        "gait", nargs="?", default="trot", choices=sorted(GAITS.keys()))
    parser.add_argument(
        "dogs", nargs="*", help="dog namespaces, default: dog1 dog2 dog3")
    parser.add_argument(
        "--connect-timeout", type=float, default=1.0,
        help="seconds to wait for ROS subscribers before publishing")
    parser.add_argument(
        "--repeat", action="store_true",
        help="repeat the same schedule; diagnostic only because it resets gait phase")
    parser.add_argument(
        "--duration", type=float, default=1.0,
        help="seconds to publish when --repeat is set")
    parser.add_argument(
        "--rate", type=float, default=10.0,
        help="publish rate while duration is active")
    args = parser.parse_args()

    gait_name = args.gait
    dogs = args.dogs or ["dog1", "dog2", "dog3"]

    print(f"[gait_broadcaster] gait={gait_name}, dogs={dogs}")
    if gait_name == "standing_trot":
        print("[gait_broadcaster] WARNING: standing_trot alternates diagonal contact modes 9/6.")
    if args.repeat:
        print("[gait_broadcaster] WARNING: repeated schedules reset gait phase on every publish.")

    rospy.init_node("gait_broadcaster", anonymous=True)

    # latch=True 與 GaitKeyboardPublisher 行為一致
    pubs = {}
    for ns in dogs:
        topic = f"/{ns}/{ns}_mpc_mode_schedule"
        pubs[ns] = rospy.Publisher(topic, mode_schedule, queue_size=1, latch=True)
        print(f"[gait_broadcaster] Publisher created: {topic}")

    # 等 ROS discovery + TCP 連線建立；latched publisher 會保留最後一筆訊息。
    if args.connect_timeout > 0.0:
        print(f"[gait_broadcaster] Waiting up to {args.connect_timeout:.2f}s for connection...")
        deadline = time.time() + args.connect_timeout
        while time.time() < deadline and not rospy.is_shutdown():
            if all(pub.get_num_connections() > 0 for pub in pubs.values()):
                break
            time.sleep(0.05)

    for ns, pub in pubs.items():
        print(f"  {ns}: connections={pub.get_num_connections()}")

    gait = GAITS[gait_name]
    msg = mode_schedule()
    msg.eventTimes = gait["times"]
    msg.modeSequence = gait["modes"]

    print(f"[gait_broadcaster] Publishing mode_schedule:")
    print(f"  eventTimes:   {msg.eventTimes}")
    print(f"  modeSequence: {msg.modeSequence}")

    rate_hz = max(args.rate, 0.1)
    rounds = 1
    if args.repeat:
        rounds = max(1, int(math.ceil(max(args.duration, 0.0) * rate_hz)))
    rate = rospy.Rate(rate_hz)

    print(f"[gait_broadcaster] Publishing {rounds} round(s) at {rate_hz:.1f} Hz...")
    count = 0
    while count < rounds and not rospy.is_shutdown():
        for ns, pub in pubs.items():
            pub.publish(msg)
        count += 1
        if count < rounds:
            rate.sleep()

    time.sleep(0.2)
    print(f"[gait_broadcaster] Total {count} rounds. Done.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n[gait_broadcaster] Interrupted.")
