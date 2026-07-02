# 階段 4(離線核心)— ADMM ξ → OCS2 centroidal target adapter（完成）

> 施工圖見 `docs/文件二_架構藍圖.md` C6 + `docs/opensource_reference.md`(legged_planner 只當範本讀,
> 不建其 C++)。**通過標準**:`test_motion_adapter.py` **8/8 綠** + `verify_stage4.py` **7/7 整合檢查 + 出圖**。
> 狀態:**離線核心綠**。接線(rung 0→2 上 Gazebo)是**獨立下一步**,不在本階段。

## 範圍(離線核心,刻意先隔離)

只做 **ADMM ξ → OCS2 24 維 centroidal state 序列** 的映射 + yaw 旁路,純 numpy、rospy-free、可翻 C++。
**發佈路徑 A**(直發 `ocs2_msgs/mpc_target_trajectories` 給 OCS2,不夾 legged_planner C++)。ROS 發佈殼與
Gazebo 端到端留給接線階段(見文末)。

## 做了什麼(3 新檔,純加法,未動任何現有檔 → 無回歸)

| 檔案 | 角色 |
|---|---|
| `admm/motion_adapter.py` | 4 function + 具名 index 常數 + 明確賦值 padding;rospy-free |
| `admm/test_motion_adapter.py` | 8 單元(index placement、idx3-5=0、yaw 凍結/±π 連續/seed 連續、組裝) |
| `admm/verify_stage4.py` | 真弧線 ADMM ξ 跑 adapter + 7 整合檢查 + 三 panel 圖 `stage4_adapter.png` |

## OCS2 24 維映射（從 OCS2 C++ 釘死,非 legged_planner 的 loose 註解）

證據:`ocs2_centroidal_model/AccessHelperFunctionsImpl.h:112/130/147`(`getNormalizedMomentum=state[0:6]`、
`getBasePose=state[6:12]`、`getJointAngles=state[12:]`)、`CentroidalModelRbdConversions.cpp:70-71,82,119`
(`basePosition=head<3>`、`baseOrientation=tail<3>` euler-ZYX、`momentum=A·v/mass=[linear;angular]`)。

| idx | 真實語意(OCS2 C++) | 我方填 |
|---|---|---|
| 0,1,2 | 正規化線動量 = COM 線速度(÷mass) | vx, vy, 0 |
| **3,4,5** | 正規化**角動量** [Lx,Ly,Lz]/m(**非** ψ̇) | **0**(拍板:不給角動量參考,MPC 自算) |
| 6,7,8 | base 位置 x,y,z | px, py, COM_HEIGHT(強制蓋) |
| 9,10,11 | base 姿態 euler-ZYX = yaw,pitch,roll | **yaw**(旁路), 0, 0 |
| 12–23 | 12 關節 | DEFAULT_JOINT_STATES |
| control/input | — | 全 0(OCS2 自算 GRF) |

**修正 spec C6.2**:原表 idx3=ψ̇ 是承 legged_planner 註解的錯標,實體是角動量(單位 kg·m²/s vs rad/s,
且軸序 yaw≈L_z≈idx5 非 idx3)→ 改為 **idx3-5=0**,yaw 只走 idx9 姿態。`test_ratified_zero_slots` 釘死。

## 4 個 function
- `pad_ocs2_state(px,py,vx,vy,yaw, com_height, default_joints) -> (24,)`:**明確 `s[6]=px` 式賦值 +
  關節 for-loop**(不用 numpy fancy reshape),一對一可翻 C++。
- `yaw_step(vx,vy, prev, v_freeze, ema_alpha)`:低速凍結(避 `atan2(0,0)`)+ `wrap_to_pi(raw-prev)` 最短弧
  增量 + **連續累加**。
- `yaw_trajectory(...)`:掃 k=1..N 帶 prev;末值當下週期 seed。
- `MotionAdapter.build_target(xi,t0,seed_yaw)` / `.adapt(xi,dog,t0)`:ξ→(times, states(N,24), zero inputs,
  next_seed);`adapt` 帶 per-dog seed。

## yaw 旁路設計定案:wrap 增量、不 wrap 累加器
單元測試最初假設 yaw 輸出留 [−π,π],FAIL 逼出正確設計:**追蹤參考的 yaw 要連續跨 ±π**(否則
OCS2 的 `(yaw−yaw_ref)²` 成本在 ±π 看到 2π 假跳、誤判要倒轉一圈)。故 `wrap_to_pi` 套在**增量**
`(raw−prev)`(最短弧)、不套累加器。這才是「never let EMA cross ±π unwrapped」的正解(不對原始 atan2
做 EMA,對 wrapped 增量做)。`test_yaw_wrap_continuous_crossing` 驗連續+平滑穿越。

## 接口確認(給接線用,從 `LeggedController.cpp`)
- robotName 預設 `"legged_robot"`(`:288-289`)。
- **收 target**:`RosReferenceManager(robotName).subscribe(nh)` 訂 `<robotName>_mpc_target`(`:308-311`)→
  `/dogN/legged_robot_mpc_target`,msg `ocs2_msgs/mpc_target_trajectories`。
- **發 observation**:`/dogN/legged_robot_mpc_observation`(`:316-318`,`ocs2_msgs/mpc_observation`)——接線時
  訂它拿 `observation.time`(對齊時間軸)+ `state[9]`(實測 base yaw 當 yaw seed)。

## verify 結果
`test_motion_adapter` 8/8 綠(host + 容器)。`verify_stage4`:弧線 68 週期,yaw 隨轉彎掃 0→78.7°,7/7 整合
檢查綠(states shape、times 遞增、inputs 全 0、px→idx6、z→COM_HEIGHT、**idx3-5=0**、pitch/roll=0)。
`stage4_adapter.png` 三 panel:(a) 路徑+yaw 箭頭(頭朝移動方向)、(b) yaw(t) EMA vs 原始 atan2、
(c) ±180° 穿越 EMA 連續 vs 原始跳變。

## 未做(接線,獨立下一步)
- **薄 rospy 發佈節點** `ocs2_target_publisher.py`:訂 observation → `MotionAdapter.adapt` → 發 target。
  **adapter 核心不動**。
- **3 rung 上 Gazebo**:rung 0 手搓 trivial 直線 target 驗純鏈路(不接 ADMM)、rung 1 單狗 ADMM、
  rung 2 三狗編隊。
- `COM_HEIGHT`/`DEFAULT_JOINT_STATES` 從 `reference.info` 載入(現用 placeholder 0.30 + A1 站姿)。
- idx9 吃 wrapped 還是 continuous、seed 從實測 base yaw 錨定 —— Gazebo 行為確認。
- input_dim 從 OCS2 info 確認(現預設 24)。

## 環境限制
`verify_stage4.py` 在容器 `5cdd1d8f092e` 跑(osqp 0.6.3 + matplotlib);`test_motion_adapter.py` rospy-free、
host/容器皆可。bind mount:host `==` 容器 `/root/LeggedControl_ws`。
