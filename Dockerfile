FROM osrf/ros:noetic-desktop-full

# ============================================
# 第 1 部分：基本環境設定
# ============================================
ENV TZ=Asia/Taipei
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    tzdata \
    locales \
    sudo \
    curl \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8

# ============================================
# 第 2 部分：安裝基本工具
# ============================================
ENV DEBIAN_FRONTEND=noninteractive
# 解釋：避免安裝過程中出現交互式提示

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    git \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*
# 解釋：
# - apt-get update: 更新軟體列表
# - apt-get install -y: 自動安裝（-y 表示自動回答 yes）
# - rm -rf /var/lib/apt/lists/*: 清理緩存，減小映像大小

# ============================================
# 第 3 部分：安裝 ROS 控制相關套件
# ============================================
RUN apt-get update && apt-get install -y \
    ros-noetic-controller-interface \
    ros-noetic-controller-manager \
    ros-noetic-ros-control \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-controller \
    ros-noetic-effort-controllers \
    ros-noetic-position-controllers \
    ros-noetic-robot-state-publisher \
    ros-noetic-xacro \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 第 4 部分：常用 shell 設定
# ============================================
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

WORKDIR /root

CMD ["/bin/bash"]
