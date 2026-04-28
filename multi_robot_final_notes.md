# 多機器狗編隊控制 — 最終完整筆記

###### tags: `legged_control` `multi-robot` `ROS` `MPC` `OCS2`

> **研究主題**:機器狗多機編隊上層任務分配(CBBA & CBF → SLAM → VLA)
> **框架**:[legged_control](https://github.com/qiayuanl/legged_control) + OCS2 + Gazebo
> **機器狗**:Unitree A1 × 3(架構支援擴展至 5)
> **環境**:Docker Ubuntu 20.04 + ROS Noetic
> **狀態**:✅ 三隻狗同時站立、切 trot、共同走動已驗證
> **最後更新**:2026-04-20

---

## 目錄

[TOC]

---

## 第一部分 多機隔離的整體架構

### 1.1 四層隔離原則

legged_control 原版只設計給單隻狗。要變成多機,必須在**四個層面**同時做 namespace 隔離,任何一層斷裂整個系統崩。

| Layer | 隔離機制 | 關鍵檔案 |
|---|---|---|
| **L1 Gazebo** | `<robotNamespace>` 參數化 | `gazebo.xacro`, `robot.xacro` |
| **L2 Launch** | `<group ns="dogN">` + 參數路徑 | `single_dog.launch`, `load_controller_multi.launch`, `fleet_bringup.launch` |
| **L3 C++ Controller** | NodeHandle 使用對的 namespace | `LeggedController.cpp`, `TargetTrajectoriesPublisher.cpp` |
| **L4 C++ State & MPC** | 訂閱/發布去掉 `/`,讀 robot_name | `StateEstimateBase.cpp`, `FromTopicEstimate.cpp`, `GaitReceiver.cpp`, OCS2 `LeggedRobotGaitCommandNode.cpp` |

### 1.2 最關鍵的踩坑:OCS2 GaitReceiver 用 UDP

這是非常隱晦的 bug:

> `GaitReceiver::GaitReceiver()` 在 subscribe 時強制指定 `ros::TransportHints().udp()`,只接受 UDPROS。
> 
> Python (rospy) 的 Publisher 只支援 TCPROS,所以發的 mode_schedule msg **永遠送不到** GaitReceiver callback。
> 
> 導致「鍵盤按 trot 可以,寫 Python script 發 mode_schedule 沒反應」這個一看不懂的現象。

解法是把這個 UDP 約束拿掉。詳見第三部分 L4。

---

## 第二部分 檔案修改清單(四層)

以下每個檔案的完整修改與解釋。你可以當 reference 用,復製任一段直接貼。

### L1 Gazebo 層

#### L1-1 `gazebo.xacro`

**路徑**:`legged_examples/legged_unitree/legged_unitree_description/urdf/common/gazebo.xacro`

**為什麼改**:Gazebo `libgazebo_ros_control` plugin 的 `<robotNamespace>` 決定 controller_manager service 掛在哪。原版寫死 `/`,五隻狗全擠一起。

**完整內容**:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

    <xacro:arg name="namespace" default="/"/>

    <gazebo>
        <plugin name="gazebo_ros_control" filename="liblegged_hw_sim.so">
            <robotNamespace>$(arg namespace)</robotNamespace>
            <robotParam>$(arg namespace)/legged_robot_description</robotParam>
            <robotSimType>legged_gazebo/LeggedHWSim</robotSimType>
        </plugin>
    </gazebo>

    <gazebo>
        <plugin name="p3d_base_controller" filename="libgazebo_ros_p3d.so">
            <alwaysOn>true</alwaysOn>
            <updateRate>1000.0</updateRate>
            <bodyName>base</bodyName>
            <topicName>$(arg namespace)/ground_truth/state</topicName>
            <gaussianNoise>0</gaussianNoise>
            <frameName>world</frameName>
            <xyzOffsets>0 0 0</xyzOffsets>
            <rpyOffsets>0 0 0</rpyOffsets>
        </plugin>
    </gazebo>
</robot>
```

**三個關鍵參數化**:
- `<robotNamespace>` → `/dogN` 下建 controller_manager
- `<robotParam>` → 從 `/dogN/legged_robot_description` 讀 URDF
- `<topicName>` → ground truth 發布到 `/dogN/ground_truth/state`

#### L1-2 `robot.xacro`

**路徑**:`legged_examples/legged_unitree/legged_unitree_description/urdf/robot.xacro`

**為什麼改**:`gazebo.xacro` 用了 `$(arg namespace)`,但 xacro 變數作用域限制 include 它的檔案也要宣告一次。

**需要新增這一行**(在 `<robot>` 開頭附近,跟 `robot_type` 一起):

```xml
<xacro:arg name="robot_type" default="a1"/>
<xacro:arg name="namespace" default="/"/>    <!-- 新增 -->
```

其他保持原樣。

---

### L2 Launch 層

#### L2-1 `generate_urdf.sh`

**路徑**:`legged_common/scripts/generate_urdf.sh`

**為什麼改**:每隻狗的 URDF 內容不同(`robotNamespace` 不同),所以要生成獨立的 URDF 檔。

**完整內容**:

```bash
#!/usr/bin/env sh
mkdir -p /tmp/legged_control/
if [ -z "$3" ]; then
    # 單機:沿用舊行為,不帶 namespace
    rosrun xacro xacro $1 robot_type:=$2 \
        > /tmp/legged_control/$2.urdf
else
    # 多機:加 namespace,檔名前綴加 ns 避免衝突
    rosrun xacro xacro $1 robot_type:=$2 namespace:=$3 \
        > /tmp/legged_control/${3}_$2.urdf
fi
```

#### L2-2 `single_dog.launch`

**路徑**:`legged_examples/legged_unitree/legged_unitree_description/launch/single_dog.launch`

**用途**:可重用的單狗模板,處理一隻狗的 Gazebo spawn + URDF 參數。

**完整內容**:

```xml
<?xml version="1.0"?>
<launch>
    <arg name="robot_type" default="a1"/>
    <arg name="ns"/>
    <arg name="init_x"/>
    <arg name="init_y" default="0.0"/>
    <arg name="init_z" default="0.5"/>

    <group ns="$(arg ns)">
        <rosparam file="$(find legged_gazebo)/config/default.yaml" command="load"/>

        <param name="legged_robot_description"
               command="$(find xacro)/xacro
                        $(find legged_unitree_description)/urdf/robot.xacro
                        robot_type:=$(arg robot_type)
                        namespace:=/$(arg ns)"/>

        <node name="generate_urdf"
              pkg="legged_common"
              type="generate_urdf.sh"
              output="screen"
              args="$(find legged_unitree_description)/urdf/robot.xacro
                    $(arg robot_type)
                    $(arg ns)"/>

        <node name="spawn_urdf"
              pkg="gazebo_ros"
              type="spawn_model"
              output="screen"
              args="-x $(arg init_x)
                    -y $(arg init_y)
                    -z $(arg init_z)
                    -param /$(arg ns)/legged_robot_description
                    -urdf
                    -model $(arg ns)"/>
    </group>
</launch>
```

#### L2-3 `five_dogs.launch`(目前測試三隻狗版本)

**路徑**:`legged_examples/legged_unitree/legged_unitree_description/launch/five_dogs.launch`

**完整內容**:

```xml
<?xml version="1.0"?>
<launch>
    <arg name="robot_type" default="a1"/>

    <include file="$(find gazebo_ros)/launch/empty_world.launch">
        <arg name="world_name" value="$(find legged_gazebo)/worlds/empty_world.world"/>
        <arg name="paused" value="false"/>
        <arg name="use_sim_time" value="true"/>
        <arg name="gui" value="true"/>
    </include>

    <include file="$(find legged_unitree_description)/launch/single_dog.launch">
        <arg name="robot_type" value="$(arg robot_type)"/>
        <arg name="ns" value="dog1"/>
        <arg name="init_x" value="0.0"/>
        <arg name="init_y" value="3.0"/>
    </include>

    <include file="$(find legged_unitree_description)/launch/single_dog.launch">
        <arg name="robot_type" value="$(arg robot_type)"/>
        <arg name="ns" value="dog2"/>
        <arg name="init_x" value="1.0"/>
        <arg name="init_y" value="1.5"/>
    </include>

    <include file="$(find legged_unitree_description)/launch/single_dog.launch">
        <arg name="robot_type" value="$(arg robot_type)"/>
        <arg name="ns" value="dog3"/>
        <arg name="init_x" value="2.0"/>
        <arg name="init_y" value="0.0"/>
    </include>

    <!-- 要擴展到 5 隻狗時解開註解 -->
</launch>
```

#### L2-4 `load_controller_multi.launch`

**路徑**:`legged_controllers/launch/load_controller_multi.launch`

**用途**:單隻狗的 controller + OCS2 node 啟動模板。

**完整內容**:

```xml
<?xml version="1.0"?>
<launch>
    <arg name="robot_type" default="a1"/>
    <arg name="ns"/>
    <arg name="cheater" default="false"/>
    <arg name="delay" default="0"/>

    <group ns="$(arg ns)">
        <group ns="controllers/legged_controller">
            <param name="urdfFile"
                   value="/tmp/legged_control/$(arg ns)_$(arg robot_type).urdf"/>
            <param name="taskFile"
                   value="$(find legged_controllers)/config/$(arg robot_type)/task.info"/>
            <param name="referenceFile"
                   value="$(find legged_controllers)/config/$(arg robot_type)/reference.info"/>
            <param name="gaitCommandFile"
                   value="$(find legged_controllers)/config/$(arg robot_type)/gait.info"/>
            <param name="robot_name" value="$(arg ns)"/>
        </group>

        <param name="referenceFile"
               value="$(find legged_controllers)/config/$(arg robot_type)/reference.info"/>
        <param name="taskFile"
               value="$(find legged_controllers)/config/$(arg robot_type)/task.info"/>
        <param name="gaitCommandFile"
               value="$(find legged_controllers)/config/$(arg robot_type)/gait.info"/>

        <rosparam file="$(find legged_controllers)/config/controllers.yaml" command="load"/>
        <rosparam file="$(find legged_gazebo)/config/default.yaml" command="load"/>

        <node unless="$(arg cheater)"
              name="controller_loader"
              pkg="controller_manager" type="controller_manager"
              output="screen"
              launch-prefix="bash -c 'sleep $(arg delay) &amp;&amp; $0 $@'"
              args="load controllers/joint_state_controller controllers/legged_controller"/>

        <node if="$(arg cheater)"
              name="controller_loader"
              pkg="controller_manager" type="controller_manager"
              output="screen"
              launch-prefix="bash -c 'sleep $(arg delay) &amp;&amp; $0 $@'"
              args="load controllers/joint_state_controller controllers/legged_controller controllers/legged_cheater_controller"/>

        <node pkg="ocs2_legged_robot_ros"
              type="legged_robot_gait_command"
              name="legged_robot_gait_command"
              output="screen">
            <param name="robot_name" value="$(arg ns)"/>
        </node>

        <node pkg="legged_controllers"
              type="legged_target_trajectories_publisher"
              name="legged_robot_target"
              output="screen">
            <param name="robot_name" value="$(arg ns)"/>
        </node>
    </group>
</launch>
```

**幾個關鍵設計**:
- `launch-prefix="bash -c 'sleep X && $0 $@'"` — delay 啟動,避免多隻狗同時初始化時 CppAD race
- `<param name="robot_name">` — 傳給 OCS2 的 gait_command 和我們的 target publisher,讓它們知道自己是哪隻狗
- `args="load ..."` 而不是 `args="spawner ..."` — 只 load 不 start,給後續 switch_controller 控制時機

#### L2-5 `fleet_bringup.launch`

**路徑**:`legged_controllers/launch/fleet_bringup.launch`

**完整內容**:

```xml
<?xml version="1.0"?>
<launch>
    <arg name="robot_type" default="a1"/>

    <include file="$(find legged_controllers)/launch/load_controller_multi.launch">
        <arg name="robot_type" value="$(arg robot_type)"/>
        <arg name="ns" value="dog1"/>
        <arg name="delay" value="0"/>
    </include>

    <include file="$(find legged_controllers)/launch/load_controller_multi.launch">
        <arg name="robot_type" value="$(arg robot_type)"/>
        <arg name="ns" value="dog2"/>
        <arg name="delay" value="15"/>
    </include>

    <include file="$(find legged_controllers)/launch/load_controller_multi.launch">
        <arg name="robot_type" value="$(arg robot_type)"/>
        <arg name="ns" value="dog3"/>
        <arg name="delay" value="30"/>
    </include>

    <!-- 要擴展到 5 隻狗時解開註解,delay 繼續 45, 60 -->
</launch>
```

**為什麼 delay 0/15/30**:每隻狗 controller 初始化時要建立 Pinocchio 模型 + 可能編 CppAD code,同時做會:
1. 記憶體峰值過高
2. CppAD 寫同資料夾 race condition → segfault
3. Pinocchio 全域靜態變數衝突

錯開啟動讓每隻狗依序初始化乾淨。

---

### L3 C++ Controller 層

#### L3-1 `LeggedController.h`

**路徑**:`legged_controllers/include/legged_controllers/LeggedController.h`

**修改**:在 private 區塊新增:

```cpp
private:
    ros::NodeHandle controllerNh_;   // 新增:存 init() 拿到的 nh,setupMpc 會用
```

#### L3-2 `LeggedController.cpp` — init() 修改

**路徑**:`legged_controllers/src/LeggedController.cpp`

**關鍵修改點**:

##### A. init() 開頭存 nh,getParam 去掉 `/`

```cpp
bool LeggedController::init(hardware_interface::RobotHW* robot_hw,
                            ros::NodeHandle& controller_nh) {
    controllerNh_ = controller_nh;  // 新增

    std::string urdfFile, taskFile, referenceFile;
    // 原版:controller_nh.getParam("/urdfFile", urdfFile);  ← 帶 "/" 永遠讀全域
    // 修改:去掉 "/" 用 relative path,透過 controller_nh 的 namespace 解析
    controller_nh.getParam("urdfFile", urdfFile);
    controller_nh.getParam("taskFile", taskFile);
    controller_nh.getParam("referenceFile", referenceFile);

    // ... 其餘邏輯保持原版
}
```

##### B. Visualization nh 用 parentNamespace × 2

```cpp
    // 原版:ros::NodeHandle nh(controller_nh.getNamespace());
    //   太深,拿到 /dog1/controllers/legged_controller
    // 修改:拉回 /dog1(parentNamespace x 2)
    ros::NodeHandle nh(ros::names::parentNamespace(
                       ros::names::parentNamespace(
                       controller_nh.getNamespace())));
```

#### L3-3 `LeggedController.cpp` — setupMpc() 完整修改

```cpp
void LeggedController::setupMpc() {
    // 這兩行是原版就有,qiayuan 的版本會執行,一定要保留
    mpc_ = std::make_shared<SqpMpc>(leggedInterface_->mpcSettings(),
                                     leggedInterface_->sqpSettings(),
                                     leggedInterface_->getOptimalControlProblem(),
                                     leggedInterface_->getInitializer());
    rbdConversions_ = std::make_shared<CentroidalModelRbdConversions>(
        leggedInterface_->getPinocchioInterface(),
        leggedInterface_->getCentroidalModelInfo());

    // 關鍵 1:從 launch 傳的參數讀 robot_name
    // 原版硬寫 const std::string robotName = "legged_robot";
    // 五隻狗的 MPC topic 全部叫 legged_robot_mpc_* → publisher 互打架
    std::string robotName;
    if (!controllerNh_.getParam("robot_name", robotName)) {
        robotName = "legged_robot";
        ROS_WARN("[setupMpc] robot_name not found, using default");
    }

    // 關鍵 2:用對的 namespace NodeHandle
    // controllerNh_ = /dog1/controllers/legged_controller
    // parent x 2 → /dog1,然後 advertise/subscribe 出來會是 /dog1/<topic>
    ros::NodeHandle nh(ros::names::parentNamespace(
                       ros::names::parentNamespace(
                       controllerNh_.getNamespace())));

    auto gaitReceiverPtr = std::make_shared<GaitReceiver>(
        nh, 
        leggedInterface_->getSwitchedModelReferenceManagerPtr()->getGaitSchedule(), 
        robotName);

    auto rosReferenceManagerPtr = std::make_shared<RosReferenceManager>(
        robotName, leggedInterface_->getReferenceManagerPtr());
    rosReferenceManagerPtr->subscribe(nh);

    mpc_->getSolverPtr()->addSynchronizedModule(gaitReceiverPtr);
    mpc_->getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);

    observationPublisher_ = nh.advertise<ocs2_msgs::mpc_observation>(
        robotName + "_mpc_observation", 1);
}
```

#### L3-4 `TargetTrajectoriesPublisher.h`

**路徑**:`legged_controllers/include/legged_controllers/TargetTrajectoriesPublisher.h`

**修改**:建構子裡的 subscribe 去掉 `/`:

```cpp
// 原版:
// goalSub_ = nh.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, ...);
// cmdVelSub_ = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 1, ...);

// 修改(去掉前面的 "/"):
goalSub_ = nh.subscribe<geometry_msgs::PoseStamped>("move_base_simple/goal", 1, goalCallback);
cmdVelSub_ = nh.subscribe<geometry_msgs::Twist>("cmd_vel", 1, cmdVelCallback);
```

#### L3-5 `TargetTrajectoriesPublisher.cpp`

**路徑**:`legged_controllers/src/TargetTrajectoriesPublisher.cpp`

**main() 修改**:

```cpp
int main(int argc, char** argv) {
    ::ros::init(argc, argv, "legged_robot_target");

    ::ros::NodeHandle nh;       // /dog1/(從 <group ns="dog1"> 來)
    ::ros::NodeHandle pnh("~"); // /dog1/legged_robot_target/(私有)

    // 從 private 讀 robot_name(launch 裡 <param name="robot_name" value="dog1"/>)
    std::string robotName;
    pnh.param<std::string>("robot_name", robotName, "legged_robot");

    // 相對路徑讀 /dog1/ 下的參數(去掉 "/")
    std::string referenceFile, taskFile;
    nh.getParam("referenceFile", referenceFile);
    nh.getParam("taskFile", taskFile);

    // ... 其餘邏輯原版
}
```

---

### L4 C++ State & OCS2 層

#### L4-1 `StateEstimateBase.cpp`

**路徑**:`legged_estimation/src/StateEstimateBase.cpp`

**問題**:建構子裡 `ros::NodeHandle nh;` 是全域,發 odom/pose 時發到 `/odom`、`/pose`,五隻狗打架。

**修改**:建構子的 nh 改成讀當前 node 的 namespace:

```cpp
StateEstimateBase::StateEstimateBase(PinocchioInterface pinocchioInterface,
                                     CentroidalModelInfo info,
                                     const PinocchioEndEffectorKinematics& eeKinematics)
    : pinocchioInterface_(std::move(pinocchioInterface)),
      info_(std::move(info)),
      eeKinematics_(eeKinematics.clone()),
      rbdState_(vector_t::Zero(2 * info_.generalizedCoordinatesNum)) {
    // 原版:ros::NodeHandle nh;  ← 全域
    // 修改:讀當前 node 的 namespace(多機 /dog1,單機 /)
    ros::NodeHandle nh(ros::this_node::getNamespace());
    odomPub_.reset(new realtime_tools::RealtimePublisher<nav_msgs::Odometry>(nh, "odom", 10));
    posePub_.reset(new realtime_tools::RealtimePublisher<
        geometry_msgs::PoseWithCovarianceStamped>(nh, "pose", 10));
}
```

#### L4-2 `FromTopicEstimate.cpp`

**路徑**:`legged_estimation/src/FromTopiceEstimate.cpp`

**問題**:hardcode 訂閱 `/ground_truth/state`(全域)。但 gazebo.xacro 已經把每隻狗的 ground_truth 改成 `/dogN/ground_truth/state`。

**修改**:

```cpp
FromTopicStateEstimate::FromTopicStateEstimate(...) {
    // 原版:ros::NodeHandle nh;
    //       sub_ = nh.subscribe<nav_msgs::Odometry>("/ground_truth/state", ...);
    // 修改:namespace 化
    ros::NodeHandle nh(ros::this_node::getNamespace());
    sub_ = nh.subscribe<nav_msgs::Odometry>("ground_truth/state", 10,
                                             &FromTopicStateEstimate::callback, this);
}
```

#### L4-3 OCS2 `LeggedRobotGaitCommandNode.cpp`

**路徑**:`ocs2_robotic_examples/ocs2_legged_robot_ros/src/LeggedRobotGaitCommandNode.cpp`

**為什麼改**:原版硬寫 `robotName = "legged_robot"`,忽略 launch 傳的 `~robot_name`。五隻狗的 gait_command 都發到同一個 topic。

**完整內容**:

```cpp
#include <ocs2_core/Types.h>
#include <ocs2_core/misc/CommandLine.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot_ros/gait/GaitKeyboardPublisher.h>
#include <ros/init.h>
#include <ros/package.h>

using namespace ocs2;
using namespace legged_robot;

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "legged_robot_gait_command");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string robotName;
    pnh.param<std::string>("robot_name", robotName, "legged_robot");

    std::string gaitCommandFile;
    nh.getParam("gaitCommandFile", gaitCommandFile);

    GaitKeyboardPublisher gaitCommand(nh, gaitCommandFile, robotName, true);

    while (ros::ok() && ros::master::check()) {
        gaitCommand.getKeyboardCommand();
    }

    return 0;
}
```

#### L4-4 ⭐ OCS2 `GaitReceiver.cpp`(最關鍵修改!)

**路徑**:`ocs2_robotic_examples/ocs2_legged_robot_ros/src/gait/GaitReceiver.cpp`

**為什麼改**:原版用 `ros::TransportHints().udp()` 強制 UDPROS。rospy Publisher 不支援 UDP,Python 發的 mode_schedule msg 完全送不到 GaitReceiver callback。

導致的症狀:**鍵盤觸發 trot 可以(因為 roscpp 自動支援 UDP),但任何 Python / `rostopic pub` 嘗試切 gait 都無效**。

**修改**:第 42-43 行

```cpp
// 原版:
// mpcModeSequenceSubscriber_ = nodeHandle.subscribe(
//     robotName + "_mpc_mode_schedule", 1, 
//     &GaitReceiver::mpcModeSequenceCallback, this,
//     ::ros::TransportHints().udp());     // ← 強制 UDP

// 修改:移除 TransportHints().udp(),讓預設使用 TCP
mpcModeSequenceSubscriber_ = nodeHandle.subscribe(
    robotName + "_mpc_mode_schedule", 1, 
    &GaitReceiver::mpcModeSequenceCallback, this);
```

**影響評估**:
- 延遲:TCP 比 UDP 多 1-2ms,對步態切換完全可忽略
- 可靠性:UDP 可能丟包,TCP 保證送達(更好)

---

## 第三部分 新增的輔助工具

### 3.1 `gait_broadcaster.py` — 繞過鍵盤切步態

這是我們新寫的 Python 工具,取代鍵盤按 trot 的動作。**必須在 L4-4 的 UDP→TCP 修改後才能 work**。

**完整內容**:

```python
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
```

**安裝位置建議**:`legged_controllers/scripts/gait_broadcaster.py`(記得 `chmod +x`)

---

### 3.2 `start_fleet.sh` — 一鍵啟動三隻狗

取代「三個 for 迴圈 + 手動跑 Python」的啟動流程。

**完整內容**:

```bash
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
```

**安裝位置**:`legged_controllers/scripts/start_fleet.sh`

---

## 第四部分 關於 rosservice call 的 YAML

### 為什麼 `rosservice call ... "{start_controllers: [...], ...}"` 要用這種格式

`rosservice call <service> <args>` 的 `<args>` 本質上是 YAML 格式。兩種寫法都合法:

**Flow style(一行內,最穩定)**:
```bash
rosservice call /dog1/controller_manager/switch_controller \
  "{start_controllers: ['controllers/legged_controller'], stop_controllers: [], strictness: 0, start_asap: false, timeout: 0.0}"
```

**Block style(多行,所有 key 必須頂格)**:
```bash
rosservice call /dog1/controller_manager/switch_controller "
start_controllers: ['controllers/legged_controller']
stop_controllers: []
strictness: 0
start_asap: false
timeout: 0.0
"
```

**寫法陷阱**:如果寫成:
```bash
"start_controllers: ['...']
   stop_controllers: ['']    # 這個 3 格縮排會被 YAML parser 當成 start_controllers 的 child
"
```
會炸 `yaml.parser.ParserError`。**為了絕對安全,一律用 flow style({} 包起來的單行寫法)**。

### 需不需要把這些 YAML 存成檔案?

**不需要**。這些是 service 呼叫,一次性指令。正確的抽象是:
- 把多個 service call 包成 shell script(像 `start_fleet.sh`)
- 把 controller/MPC 的**設定**放在 YAML(`controllers.yaml`、`default.yaml`)並用 launch 的 `<rosparam>` load

目前架構已經清楚分離了這兩件事,不需要新增 YAML。

---

## 第五部分 完整啟動流程(最終版)

### 前置作業(只要做一次)

```bash
# 1. 把 gait_broadcaster.py 安裝到 legged_controllers
cp /path/to/gait_broadcaster.py \
   /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/

chmod +x /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/gait_broadcaster.py

# 2. 把 start_fleet.sh 安裝到 legged_controllers
cp /path/to/start_fleet.sh \
   /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/

chmod +x /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/start_fleet.sh

# 3. 編譯(只需要在第一次或修改 C++ 後做)
cd /root/LeggedControl_ws
catkin build
source devel/setup.bash
```

### 每次啟動的流程

**Terminal 1**:Gazebo
```bash
cd /root/LeggedControl_ws
source devel/setup.bash
roslaunch legged_unitree_description five_dogs.launch
# 等到三隻狗都 spawn 在地上(15-30 秒)
```

**Terminal 2**:Controller load + gait_command
```bash
cd /root/LeggedControl_ws
source devel/setup.bash
roslaunch legged_controllers fleet_bringup.launch
# 等到三隻狗都 load 完(60-90 秒,最後 dog3 有 30 秒 delay)
# 判斷 ready:log 滾動變慢或停止
```

**Terminal 3**:一鍵啟動(start + 站立 + trot)
```bash
cd /root/LeggedControl_ws
source devel/setup.bash
rosrun legged_controllers start_fleet.sh
```

**Terminal 4**:發速度走
```bash
cd /root/LeggedControl_ws
source devel/setup.bash

# 三隻狗同時往前走(x = 0.3 m/s)
rostopic pub -r 10 /dog1/cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}" > /dev/null &
rostopic pub -r 10 /dog2/cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}" > /dev/null &
rostopic pub -r 10 /dog3/cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}" > /dev/null &

# 要停
kill %1 %2 %3
```

### 如何單獨控制各隻狗走不同方向

```bash
# dog1 往前
rostopic pub -r 10 /dog1/cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}" > /dev/null &

# dog2 原地轉
rostopic pub -r 10 /dog2/cmd_vel geometry_msgs/Twist "{angular: {z: 0.5}}" > /dev/null &

# dog3 側移
rostopic pub -r 10 /dog3/cmd_vel geometry_msgs/Twist "{linear: {y: 0.2}}" > /dev/null &
```

### 切換步態

```bash
# 切 stance(所有狗停下來立正)
rosrun legged_controllers gait_broadcaster.py stance

# 切回 trot
rosrun legged_controllers gait_broadcaster.py trot
```

---

## 第六部分 topic 對照總表

最終架構下所有 topic 都在對應 namespace 底下。以 dog1 為例:

| 功能 | Topic | 發布者 | 訂閱者 |
|---|---|---|---|
| 速度指令 | `/dog1/cmd_vel` | 使用者 / fleet_manager | `/dog1/legged_robot_target` |
| 導航目標 | `/dog1/move_base_simple/goal` | 使用者 / rviz | `/dog1/legged_robot_target` |
| MPC 觀測 | `/dog1/dog1_mpc_observation` | legged_controller plugin | MPC internal |
| MPC 目標軌跡 | `/dog1/dog1_mpc_target` | `/dog1/legged_robot_target` | legged_controller |
| 步態切換 | `/dog1/dog1_mpc_mode_schedule` | `/dog1/legged_robot_gait_command` 或 gait_broadcaster.py | legged_controller(GaitReceiver) |
| 關節狀態 | `/dog1/joint_states` | joint_state_controller | tf, rviz |
| Ground truth | `/dog1/ground_truth/state` | Gazebo p3d plugin | FromTopicStateEstimate |
| Odometry | `/dog1/odom` | StateEstimateBase | 使用者 / SLAM |
| Pose | `/dog1/pose` | StateEstimateBase | 使用者 / SLAM |
| URDF 描述 | `/dog1/legged_robot_description` | launch | Gazebo spawn, controller |
| Controller manager | `/dog1/controller_manager/*` | controller_manager plugin | 使用者(rosservice) |

---

## 第七部分 關鍵除錯指令速查

### 檢查三隻狗狀態

```bash
for ns in dog1 dog2 dog3; do
  state=$(rosservice call /${ns}/controller_manager/list_controllers "{}" 2>/dev/null \
          | grep -A1 "legged_controller" | grep state | tr -d ' "')
  echo "$ns: $state"
done
```

### 看 MPC 是否活著

```bash
timeout 3 rostopic hz /dog1/dog1_mpc_observation
# 正常:~1000 Hz
# 如果 0Hz / no messages → MPC 卡住
```

### 看某隻狗的 cmd_vel 有沒有被訂閱

```bash
rostopic info /dog1/cmd_vel
# 應該看到 Subscribers: /dog1/legged_robot_target
```

### 看 GaitReceiver 有沒有收到 mode_schedule

```bash
# 看 gazebo launch 的 terminal,每次 gait 切換時應印:
# [GaitReceiver]: Setting new gait after time XX.XXX
# Template switching times: {0, 0.3, 0.6}
# Template mode sequence:   {9, 6}
```

### 如果狗跌倒了怎麼辦

```bash
# reset Gazebo 世界,所有狗回到 spawn 位置
rosservice call /gazebo/reset_world "{}"

# 因為 safety check trip,controller 會自動 stop。重新 start:
rosrun legged_controllers start_fleet.sh
```

### 清乾淨(如果要完全重啟)

```bash
pkill -9 -f roslaunch gzserver gzclient rosmaster roscore legged_
sleep 3
```

---

## 第八部分 下一步研究方向

### 短期(1-2 週)

- [ ] **Leader-Follower 控制**:寫 `leader_follower.py`,dog1 手動控制,dog2/dog3 自動跟隨
- [ ] **V 字編隊**:在前一項基礎上加「保持相對位置」邏輯
- [ ] **優化 fleet_manager.py**:目前的版本可以當 CBBA 上層邏輯的骨架

### 中期(1-2 個月)

- [ ] **讀 CBBA paper**(Choi, Brunet, How 2009),整理筆記
- [ ] **實作 CBBA core**:auction phase + consensus phase
- [ ] **定義任務**:巡邏區域、檢查目標點
- [ ] **整合 CBBA + fleet platform**

### 長期

- [ ] CBF 安全保障(多機避碰)
- [ ] SLAM 整合(地圖建立與共享)
- [ ] VLA 語義理解(語言指令 → 任務分配)

---

## 第九部分 Future Work / 未解議題

### 1. URDF link prefix(Prefix 派做法)

學長使用的方案是 `dog1_LF_hip`, `dog2_LF_hip`... 這種 link 名前綴。

**現狀**:我們走 Namespace 派,URDF link 名都叫 `LF_hip`(沒前綴)。目前 Gazebo 用 model::link 區分,不影響功能。

**未來要補**(當下面任一需求出現時):
- tf 整合:五隻狗都 broadcast `LF_hip` frame,tf tree 衝突 → 要用 `tf_prefix` 或改 URDF link 名
- Self-collision avoidance:OCS2 用 link 名比對,三隻狗可能被當作同一個機器人
- rviz 多機視覺化:frame 衝突

### 2. Visualizer 的全域 topic 污染

原版 `LeggedRobotVisualizer` 發的 topic 目前部分會變成全域:
```
/legged_robot/currentState
/legged_robot/desiredBaseTrajectory
...
```

需要改 OCS2 內的 `LeggedRobotVisualizer.cpp` 或修改 `LeggedController.cpp` 傳給它的 NodeHandle。

**優先級低**:Visualizer 只影響 rviz,不影響控制。

### 3. 效能:五隻狗的 MPC 計算

電腦測試:三隻狗跑得動(CPU ~40% 空閒),五隻狗會卡。

**選項**:
- 降低 MPC 頻率(改 `task.info`)
- 用 `legged_cheater_controller`(跳過 Kalman Filter,省 20-30% CPU)
- headless 模式(`gui:=false`)

---

## 第十部分 編譯與重載提醒

### 修改什麼,要編譯什麼?

| 修改檔案類型 | 需要重編的 package | 指令 |
|---|---|---|
| xacro/launch/yaml | 無需編譯 | roslaunch 會重讀 |
| `legged_controllers/src/*.cpp` | `legged_controllers` | `catkin build legged_controllers` |
| `legged_estimation/src/*.cpp` | `legged_estimation` | `catkin build legged_estimation` |
| OCS2 `src/*.cpp`(如 GaitReceiver) | `ocs2_legged_robot_ros` | `catkin build ocs2_legged_robot_ros` |

### 發現編譯太快(可疑地跳過 link)?

```bash
catkin clean <pkg_name> -y
catkin build <pkg_name>
# 看 Finished 的秒數,正常 OCS2 是 30 秒以上
```

### 重編後必做

1. 關掉所有 ROS(pkill)
2. `source devel/setup.bash`(確保新 .so 被 load)
3. 重啟 Gazebo + fleet_bringup

---

## 附錄 常見問題 FAQ

| 症狀 | 可能原因 | 解法 |
|---|---|---|
| 狗生不出來 | URDF 參數路徑錯 | 檢查 `rostopic list \| grep description` |
| MPC 沒發 observation | controller 不是 running | 跑 `start_fleet.sh` |
| `mpc_` nullptr segfault | setupMpc 漏 `SqpMpc` 建立 | L3-3 |
| Python 發 gait 沒反應 | OCS2 GaitReceiver 用 UDP | L4-4(這次的關鍵 bug) |
| `yaml.parser.ParserError` | rosservice call 的 YAML 縮排亂 | 改用 `{}` flow style |
| Controller state: initialized | 沒 start | `switch_controller` |
| 狗站起來又往前翻倒 | 步態是 stance 但發了速度 | 先切 trot 再發速度 |
| 只有一隻狗動 | gait_command 的 stdin 只給一隻 | 用 `gait_broadcaster.py` |
| 狗跌倒後 controller 自己 stop | safety check trip | reset + 重新 start |
