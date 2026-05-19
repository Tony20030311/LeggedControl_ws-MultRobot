#!/bin/bash
#
# start_fleet.sh
# ==============
# 啟動三隻狗並讓它們進入可接收 cmd_vel 的狀態。
#
# 執行順序:
#   1. switch_controller start(每隻狗 controller 從 initialized → running)
#   2. 短 zero cmd_vel pulse 觸發站起來 target trajectory
#   3. 直接切到 trot，不再經過 standing_trot warm-up
#
# 前置條件:
#   - Gazebo + fleet_bringup 已啟動且 controller 已 load
#
# 預設執行完狗會站起來並進入 trot 待命。

set -e

DOGS=(dog1 dog2 dog3)
START_GAIT="${1:-${START_GAIT:-trot}}"   # trot | stance | none
CONTROLLER_SETTLE="${CONTROLLER_SETTLE:-1.0}"
ZERO_ROUNDS="${ZERO_ROUNDS:-1}"
ZERO_SLEEP="${ZERO_SLEEP:-0.2}"
STANDUP_WAIT="${STANDUP_WAIT:-2.0}"
GAIT_CONNECT_TIMEOUT="${GAIT_CONNECT_TIMEOUT:-1.0}"

case "$START_GAIT" in
    stance|trot|none) ;;
    *)
        echo "[start_fleet] ERROR: gait must be stance, trot, or none; got '$START_GAIT'"
        exit 2
        ;;
esac

echo "[start_fleet] Starting controllers for: ${DOGS[*]}"
echo "[start_fleet] START_GAIT=$START_GAIT, ZERO_ROUNDS=$ZERO_ROUNDS, STANDUP_WAIT=${STANDUP_WAIT}s"
echo ""

for ns in "${DOGS[@]}"; do
    echo "=== [$ns] Start controller ==="
    rosservice call /${ns}/controller_manager/switch_controller \
        "{start_controllers: ['controllers/legged_controller'], stop_controllers: [], strictness: 1, start_asap: false, timeout: 0.0}"
    sleep "$CONTROLLER_SETTLE"
done

if [[ "$ZERO_ROUNDS" -gt 0 ]]; then
    echo ""
    echo "=== Short zero cmd_vel pulse to trigger stand-up target ==="
    for ((round=1; round<=ZERO_ROUNDS; round++)); do
        echo "  zero round $round/$ZERO_ROUNDS"
        for ns in "${DOGS[@]}"; do
            rostopic pub -1 /${ns}/cmd_vel geometry_msgs/Twist \
                "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" > /dev/null
        done
        sleep "$ZERO_SLEEP"
    done

    echo ""
    echo "=== Wait ${STANDUP_WAIT}s for stand-up ==="
    sleep "$STANDUP_WAIT"
fi

echo ""
echo "=== Verify all controllers running ==="
for ns in "${DOGS[@]}"; do
    state=$(rosservice call /${ns}/controller_manager/list_controllers "{}" 2>/dev/null \
            | grep -A1 "legged_controller" | grep state | tr -d ' "' || true)
    echo "  $ns: $state"
done

if [[ "$START_GAIT" != "none" ]]; then
    echo ""
    echo "=== Broadcast $START_GAIT schedule ==="
    rosrun legged_controllers gait_broadcaster.py "$START_GAIT" \
        --connect-timeout "$GAIT_CONNECT_TIMEOUT"
fi

echo ""
echo "======================================================================"
echo "[start_fleet] Ready. Current gait request: $START_GAIT"
echo ""
echo "Default startup sends one zero cmd_vel pulse to stand up, then switches directly to trot."
echo "No standing_trot warm-up is used."
echo ""
echo "To switch trot again manually:"
echo "  rosrun legged_controllers gait_broadcaster.py trot"
echo ""
echo "Then send cmd_vel in another terminal:"
echo "  rostopic pub -r 10 /dog1/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.3}}\""
echo "  rostopic pub -r 10 /dog2/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.3}}\""
echo "  rostopic pub -r 10 /dog3/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.3}}\""
echo "======================================================================"
