#!/bin/bash

IMAGE_NAME="my_ros_generic:noetic"
CONTAINER_NAME="LeggedControl_SIL"

# 本地掛載路徑：先改這裡
HOST_DIR="/home/tony/LeggedControl_ws"

# 容器內掛載路徑：目前先用 /root
CONTAINER_DIR="/root/LeggedControl_ws"

xhost +local:root

docker rm -f ${CONTAINER_NAME} 2>/dev/null

docker run -it \
  --network host \
  --cap-add=IPC_LOCK \
  --cap-add=SYS_NICE \
  --ulimit memlock=-1 \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --gpus all \
  --privileged \
  --name ${CONTAINER_NAME} \
  -v ${HOST_DIR}:${CONTAINER_DIR} \
  ${IMAGE_NAME}
