#!/usr/bin/env sh
mkdir -p /tmp/legged_control/
if [ -z "$3" ]; then
    # 原始用法:不帶 namespace
    rosrun xacro xacro $1 robot_type:=$2 \
        > /tmp/legged_control/$2.urdf
else
    # 新增用法:帶 namespace,輸出檔名加上 namespace 前綴
    rosrun xacro xacro $1 robot_type:=$2 namespace:=$3 \
        > /tmp/legged_control/${3}_$2.urdf
fi