#!/bin/bash
#
# start_fleet.sh
# ==============
# 啟動三隻狗並讓它們進入可接收 cmd_vel 的狀態。
#
# 執行順序:
#   1. switch_controller start(每隻狗 controller 從 initialized → running)
#   2. 發 cmd_vel=0 觸發站立(原版 starting() 把當下狀態當 target,
#      狗平躺 spawn 時需要一個新 target 來啟動「站立規劃」)
#   3. 跑 gait_broadcaster 切 trot
#
# 前置條件:
#   - Gazebo + fleet_bringup 已啟動且 controller 已 load
#
# 執行完後狗會 trot 原地踏步,再發 cmd_vel 就會走。

set -e

DOGS=(dog1 dog2 dog3)

echo "[start_fleet] Starting controllers for: ${DOGS[*]}"
echo ""

for ns in "${DOGS[@]}"; do
    echo "=== [$ns] Start controller ==="
    rosservice call /${ns}/controller_manager/switch_controller \
        "{start_controllers: ['controllers/legged_controller'], stop_controllers: [], strictness: 0, start_asap: false, timeout: 0.0}"
    sleep 3
done

echo ""
echo "=== Send zero cmd_vel to trigger stand-up ==="
for ns in "${DOGS[@]}"; do
    echo "  $ns"
    rostopic pub -1 /${ns}/cmd_vel geometry_msgs/Twist \
        "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" > /dev/null
    sleep 1
done

echo ""
echo "=== Wait 5s for stand-up stabilization ==="
sleep 5

echo ""
echo "=== Verify all controllers running ==="
for ns in "${DOGS[@]}"; do
    state=$(rosservice call /${ns}/controller_manager/list_controllers "{}" 2>/dev/null \
            | grep -A1 "legged_controller" | grep state | tr -d ' "')
    echo "  $ns: $state"
done

echo ""
echo "=== Switch all dogs to trot ==="
rosrun legged_controllers gait_broadcaster.py trot

echo ""
echo "======================================================================"
echo "[start_fleet] Ready! All dogs should be in trot mode."
echo ""
echo "To walk, send cmd_vel in another terminal:"
echo "  rostopic pub -r 10 /dog1/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.3}}\""
echo "  rostopic pub -r 10 /dog2/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.3}}\""
echo "  rostopic pub -r 10 /dog3/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.3}}\""
echo "======================================================================"