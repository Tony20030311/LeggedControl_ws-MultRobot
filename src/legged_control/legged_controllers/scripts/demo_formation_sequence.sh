#!/bin/bash
#
# demo_formation_sequence.sh
# =========================
# 用 dog1 當 leader，執行一套較明顯的 formation demo：
#   1) 快速直走
#   2) 停住，讓 followers 補位
#   3) 稍快一點的轉彎前進
#   4) 停住，讓 followers 補位
#   5) 小幅後退
#   6) 最後停住
#
# 使用前提：
#   - Gazebo 已啟動
#   - fleet_bringup.launch 已啟動
#   - start_fleet.sh 已完成
#   - formation_manager.py 已啟動

set -e

LEADER_TOPIC="/dog1/cmd_vel"
RATE=10

publish_for() {
    local duration="$1"
    local linear_x="$2"
    local linear_y="$3"
    local linear_z="$4"
    local angular_z="$5"

    timeout "$duration" rostopic pub -r "$RATE" "$LEADER_TOPIC" geometry_msgs/Twist \
        "{linear: {x: $linear_x, y: $linear_y, z: $linear_z}, angular: {x: 0.0, y: 0.0, z: $angular_z}}"
}

publish_zero() {
    rostopic pub -1 "$LEADER_TOPIC" geometry_msgs/Twist \
        "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" > /dev/null
}

echo "============================================================"
echo "[formation_demo] Start scripted formation demo with leader dog1"
echo "[formation_demo] Only /dog1/cmd_vel will be commanded"
echo "============================================================"

echo "[1/6] Fast straight forward for 4s"
publish_for 4 0.45 0.0 0.0 0.0 || true

echo "[2/6] Stop and settle for 2s"
publish_zero
sleep 2

echo "[3/6] Forward + quicker gentle left turn for 4s"
publish_for 4 0.35 0.0 0.0 0.18 || true

echo "[4/6] Stop and settle for 2s"
publish_zero
sleep 2

echo "[5/6] Gentle backward for 3s"
publish_for 3 -0.20 0.0 0.0 0.0 || true

echo "[6/6] Final stop"
publish_zero

echo "============================================================"
echo "[formation_demo] Done. dog1 stopped. Followers should settle."
echo "============================================================"