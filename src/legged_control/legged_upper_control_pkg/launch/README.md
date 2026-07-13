# launch — ADMM 三狗編隊


| launch | 開什麼場景 |
|--------|-----------|
| `three_dogs_empty.launch` | 空世界，無障礙 |
| `three_dogs_obstacles.launch` | 開放場 + 圓柱 |
| `three_dogs_door.launch` | 2.5m 門 + 5 柱 + 邊界牆 |
| `three_dogs_plum.launch` | 梅花樁：10 圓柱樁陣 |

## 其他 launch

| launch | 用途 |
|--------|------|
| `admm_demo.launch` | 起 ADMM publisher（C++ node）+ RViz，停在 home V 等 `/formation/goal`。**這是 ADMM 入口** |
| `formation_hocbf.launch` | ⚠️ legacy Pure-Pursuit stack（**非 ADMM**，舊反應式） |

## 跑一場（手動自己下點）

容器內，每行一個 terminal（`docker exec -it LeggedControl_SIL bash`），或加 `&` 背景。
先 `source /opt/ros/noetic/setup.bash && source /root/LeggedControl_ws/devel/setup.bash`。

```bash
# 1) Gazebo（門；換場景改這行的 launch 名）
roslaunch legged_upper_control three_dogs_door.launch

# 2) 控制器（等 Gazebo 看到三隻狗再跑）
roslaunch legged_controllers fleet_bringup.launch

# 3) 站立 + trot（等控制器載入完再跑）
rosrun legged_controllers start_fleet.sh trot

# 4) ADMM publisher + RViz（等三狗站起來再跑；arena 對應地圖）
roslaunch legged_upper_control admm_demo.launch 
```

起好後，**自己下點**（`/formation/goal` 是質心點，改 x/y 去哪都行）：

```bash
rostopic pub -1 /formation/goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 8.0, y: 0.0, z: 0.5}}}"
```

### 備註
- 每步要等前一步起好：Gazebo ~15s、控制器 ~35s、站立 ~10s。
- `arena:=` 對應場景：`empty`（空，或省略給 `""`）/ `obstacles` / `door` / `plum`。
- publisher 收一個質心點後，自動散成三狗 V slot + 各自 A* 繞障礙。

