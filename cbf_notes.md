# CBF 安全濾波器 — 完整筆記

###### tags: `legged_control` `multi-robot` `CBF` `QP` `safety`

> **接續**:`multi_robot_final_notes.md`（底層平台已完成）
> **本篇重點**:在多機編隊上層加入 Control Barrier Function (CBF) 安全濾波器
> **狀態**:✅ inter-dog + obstacle + wall CBF 已驗證
> **最後更新**:2026-04-28

---

## 目錄

[TOC]

---

## 第一部分 CBF 的數學本質

### 1.1 為什麼需要 CBF？

原版 `formation_manager.py` 用 PID 追蹤 leader 的 offset 位置。PID 是 best-effort 的——它「盡力」追目標，但沒有安全保證。以下情境會出事：

- Leader 急轉彎 → follower 的目標點瞬間跳位，兩隻 follower 軌跡交叉 → 碰撞
- Leader 後退 / U-turn → follower 目標點穿過 leader 本身位置
- 隊形切換過渡期 → offset 目標軌跡交叉
- 通訊延遲 / odom 掉包 → 基於過時資料算出錯誤指令

CBF 不是替代 PID，而是在 PID 產生 `u_nominal` 之後加一層**硬約束**：正常情況下 CBF 不介入（`u_safe ≈ u_nominal`），危險時自動修正速度，**數學上保證不碰撞**。

### 1.2 安全集合（Safe Set）

定義一個函數 `h(x)` 描述「離危險有多遠」：

```
h_ij(x) = ‖p_i - p_j‖² - D²_min
```

- `p_i, p_j`：兩隻狗的位置
- `D_min`：最小安全距離（我們用 1.0m）
- `h > 0`：安全（距離夠遠）
- `h = 0`：邊界（即將碰撞）
- `h < 0`：違規（碰撞）

安全集合：
```
C = { x | h(x) ≥ 0 }
```

### 1.3 Forward Invariance（前向不變性）

目標：如果系統起始在安全集合 `C` 裡面，就永遠留在 `C` 裡面。

```
x(0) ∈ C  ⟹  ∀t ≥ 0, x(t) ∈ C
```

### 1.4 與 Lyapunov 的對比

| | Lyapunov（穩定性） | CBF（安全性） |
|---|---|---|
| 函數 | V(x)：能量函數 | h(x)：安全裕度 |
| 目標 | 讓 V 遞減 → 收斂到原點 | 讓 h 不要掉到零以下 → 留在安全集 |
| 條件 | V̇ ≤ −α(V) | ḣ ≥ −α(h) |

方向相反：Lyapunov 要「下降」，CBF 要「不要下降太快」。

### 1.5 Class K 函數

`α : ℝ → ℝ` 是 class K 函數，滿足：`α(0) = 0`，嚴格遞增，連續。

最常用的選擇：**線性** `α(h) = γh`，`γ > 0`。

- `γ` 越大 → 干預越早越保守
- `γ` 越小 → 允許逼近邊界更多但反應更激進

我們目前用 `γ = 2.0`。

### 1.6 ZCBF 核心定理（Ames 2017）

考慮仿射控制系統：
```
ẋ = f(x) + g(x)u
```

我們用 single integrator model `ṗ_i = u_i`（上層 CBF 不碰底層動力學），所以 `f = 0`, `g = I`。

**ZCBF 條件**：
```
sup_u [ L_f h(x) + L_g h(x) · u ] ≥ −α(h(x))
```

其中 `L_f h`, `L_g h` 是 Lie 導數。

**定理**：如果 h 是 ZCBF，則任何滿足上面不等式的控制器 u 都能使 C 前向不變。

### 1.7 具體推導：兩狗碰撞避免

single integrator model：`ṗ_i = u_i`, `ṗ_j = u_j`

```
h_ij = ‖p_i - p_j‖² - D²_min
```

對時間微分：
```
ḣ_ij = 2(p_i - p_j)ᵀ (ṗ_i - ṗ_j)
     = 2(p_i - p_j)ᵀ (u_i - u_j)
```

CBF 條件：
```
ḣ_ij ≥ −γ · h_ij
```

展開：
```
2(p_i - p_j)ᵀ (u_i - u_j) ≥ −γ (‖p_i - p_j‖² - D²_min)
```

**這是關於 u 的線性不等式** → 可以寫成 QP constraint。

### 1.8 為什麼 h 永遠不會穿越零？

`ḣ ≥ −γh` 的解滿足：
```
h(t) ≥ h(0) · e^{−γt}
```

只要 `h(0) > 0`（起始在安全集裡），`h(t) > 0` 對所有 `t` 成立。指數函數趨近零但永遠不會到零——這就是「Zeroing」CBF 的含義。

### 1.9 坐標系問題：body frame vs world frame

CBF 推導在 **world frame**，但 `/dogN/cmd_vel` 送出去的是 **body frame**。

轉換關係：
```
ṗ_i^{world} = R(ψ_i) · u_i^{body}

R(ψ) = [ cos(ψ)  -sin(ψ) ]
        [ sin(ψ)   cos(ψ) ]
```

代入 CBF constraint 後，constraint 對 `u_i^{body}` 仍然是線性的（因為 R(ψ_i) 在當前時刻是常數矩陣），QP 的凸性不被破壞。

### 1.10 wz（yaw rate）為什麼不進 QP？

`angular.z` 不影響位置的變化率（single integrator model `ṗ = u` 只有 `vx, vy`），所以在「碰撞避免」這個目標下，CBF constraint 只約束 `(vx, vy)`，`wz` 直接 pass-through。

---

## 第二部分 CBF-QP 的完整形式

### 2.1 決策變數

所有狗（含 leader）的 body frame 速度：
```
u = [vx₁, vy₁, vx₂, vy₂, vx₃, vy₃] ∈ ℝ⁶
```

### 2.2 目標函數

離 nominal 最近：
```
min ‖u - u_nominal‖²
```

`u_nominal` 來源：
- Leader → 來自 `/dog1/cmd_vel_raw`（你的鍵盤指令）
- Follower → 來自 PID 追蹤 formation target

### 2.3 三類 Constraint

#### A. Pairwise Robot-Robot CBF（3 pairs for 3 dogs）

```
2 Δp_ij^T (R(ψ_i) u_i^{body} - R(ψ_j) u_j^{body}) ≥ −γ_robot · h_ij
```

所有狗都是決策變數（含 leader），所以兩邊都有可控項。

#### B. Obstacle CBF（N_dogs × M_obstacles）

```
h_obs = ‖p_i - p_obs‖² - r²_obs

2(p_i - p_obs)^T R(ψ_i) u_i^{body} ≥ −γ_obs · h_obs
```

每隻狗對每個障礙物一個 constraint。

#### C. Wall CBF（N_dogs × M_walls）

```
h_wall = n^T (p_i - p_wall) - d_safe

n^T R(ψ_i) u_i^{body} ≥ −γ_wall · h_wall
```

`n` = 牆壁法向量（指向安全側），`d_safe` = 安全距離。每面牆只有一個半平面約束，比圓形更簡單。

### 2.4 QP 規模

| 項目 | 數量 |
|---|---|
| 決策變數 | 6（3 dogs × 2） |
| Pairwise CBF | 3（C(3,2) = 3 pairs） |
| Obstacle CBF | 3 × 11 = 33 |
| Wall CBF | 3 × 3 = 9 |
| 總 constraints | 45 |

用 OSQP solver，解一次 < 1ms，20Hz 控制迴圈綽綽有餘。

---

## 第三部分 系統架構

### 3.1 模組化設計（fleet_manager_cbf.py）

```
FleetManagerCBF
├── StateCollector          訂閱 /dogN/ground_truth/state → RobotState
│                           新增: vx_world, vy_world（從 odom twist 轉換）
│
├── FormationPlanner        leader pose + offsets → follower 目標位姿
│                           支援 runtime 切換隊形（update_offsets）
│
├── NominalController       P 控制器，world→body 誤差轉換 → u_nominal
│
├── LeaderCmdRelay          訂閱 /dog1/cmd_vel_raw → leader 的 u_nominal
│                           （你的鍵盤指令先進這裡，再經 CBF 過濾）
│
├── CBFSafetyFilter         組裝 QP (pairwise + obstacle + wall) → u_safe
│                           solver: cvxpy + OSQP
│
├── VelocityLimiter         硬限幅 vx, vy, wz
│
└── CmdVelPublisher         發布 /dogN/cmd_vel（所有狗，含 leader）
```

### 3.2 資料流

```
鍵盤 / teleop
      │
      ▼
/dog1/cmd_vel_raw ──→ LeaderCmdRelay ──→ u_nom_leader ─┐
                                                          │
/dogN/odom ──→ StateCollector ──→ FormationPlanner       │
                                    │                     │
                                    ▼                     │
                              NominalController           │
                                    │                     │
                                    ▼                     │
                              u_nom_followers ────────────┤
                                                          │
                                                          ▼
                                                   CBFSafetyFilter
                                                          │
                                                          ▼
                                                   VelocityLimiter
                                                          │
                                                          ▼
                                                   /dogN/cmd_vel → MPC
```

### 3.3 vs 原版 formation_manager.py 的差異

| 項目 | formation_manager.py | fleet_manager_cbf.py |
|---|---|---|
| 架構 | 全部在一個 class | 六個獨立模組 |
| Leader 保護 | 無 | CBF 保護 |
| Leader 輸入 | 直接 /cmd_vel | /cmd_vel_raw → CBF → /cmd_vel |
| Follower 保護 | 無 | CBF 保護 |
| 障礙物避障 | 無 | obstacle CBF |
| 牆壁避障 | 無 | wall CBF |
| leader velocity | 不記錄 | 從 odom 取得 world velocity |
| 參數管理 | rosparam | YAML + rosparam fallback |
| spin() 邏輯 | 算一隻 publish 一隻 | 收集全部 → QP 一次解 → publish 全部 |

### 3.4 檔案清單

| 檔案 | 位置 | 用途 |
|---|---|---|
| `formation_manager.py` | `scripts/` | 純編隊 baseline（不含 CBF） |
| `formation_managerCBF.py` | `scripts/` | 編隊 + CBF 完整版 |
| `cbf_params.yaml` | `scripts/` | 場地 + CBF 參數 |
| `obstacle_world.world` | `legged_gazebo/worlds/` | 測試場地 |

### 3.5 依賴

```bash
pip install numpy cvxpy
# cvxpy 預設帶 OSQP solver
```

注意：Ubuntu 20.04 + Python 3.8 的 pip 可能太舊，需要先 `pip install --upgrade pip`。

---

## 第四部分 測試場地

### 4.1 場地設計

長方形 4m (depth, x 軸) × 10m (length, y 軸)：
```
牆壁範圍:
  左長牆: x=6, y=-5..5 （有 1.2m 洞口在 y=-0.6..0.6）
  右長牆: x=10, y=-5..5
  上短牆: y=5, x=6..10
  下短牆: y=-5, x=6..10
  
狗 spawn 位置（原版）:
  dog1: (2.0, 0.0)
  dog2: (1.0, 1.0)
  dog3: (1.0, -1.0)

狗面朝 x 正方向，走 ~4m 到達洞口。
```

### 4.2 障礙物配置

左長牆因為有洞口，不能用半平面 wall CBF（會把整面牆都擋住，狗連洞口都進不去）。解法是用離散圓形 obstacles 模擬：

```yaml
# 左牆圓形模擬（洞口 y=-0.6~0.6 之間不放）
- {pos: [6.0,  1.5], radius: 1.2}
- {pos: [6.0,  3.0], radius: 1.2}
- {pos: [6.0,  4.5], radius: 1.2}
- {pos: [6.0, -1.5], radius: 1.2}
- {pos: [6.0, -3.0], radius: 1.2}
- {pos: [6.0, -4.5], radius: 1.2}

# 場地內圓柱（物理半徑 0.2m + 安全裕量 0.4m）
- {pos: [7.5,  2.5], radius: 0.6}
- {pos: [8.5, -1.5], radius: 0.6}
- {pos: [9.0,  3.5], radius: 0.6}
- {pos: [7.0, -3.0], radius: 0.6}
- {pos: [8.0,  0.5], radius: 0.6}

# 三面實體牆（右牆 + 上牆 + 下牆）
walls:
  - {normal: [-1, 0], point: [10.0, 0],  d_safe: 0.5}  # 右牆
  - {normal: [0, -1], point: [0,  5.0],  d_safe: 0.5}  # 上牆
  - {normal: [0,  1], point: [0, -5.0],  d_safe: 0.5}  # 下牆
```

### 4.3 CBF 參數

```yaml
cbf_enabled: true
cbf_d_min: 1.0        # 狗與狗最小安全距離 (m)
cbf_gamma: 2.0        # 狗與狗 CBF 積極程度
cbf_gamma_obs: 2.0    # 障礙物 CBF 積極程度
cbf_gamma_wall: 2.0   # 牆壁 CBF 積極程度
```

### 4.4 Formation Offset

```yaml
offsets:
  dog2: [-0.6, 0.4]   # 原版 [-1.0, 1.0] 太寬（2m > 洞口 1.2m）
  dog3: [-0.6, -0.4]  # 縮小後 V 字寬 0.8m，可通過洞口
```

---

## 第五部分 遇到的問題與解法

### 問題 1: CBF 參數沒讀進去（obstacles=0, walls=0）

**症狀**：啟動 log 顯示 `obstacles = 0`, `walls = 0`。

**原因**：用 command line 的 rosparam 語法傳 list of dict 時，shell 解析失敗。

**解法**：改用 YAML 檔 + `rosparam load`，確認 namespace 跟 `rospy.init_node()` 的名稱一致。最終改成程式直接讀 YAML（`yaml.safe_load`），不再需要 `rosparam load`。

```bash
# 錯誤：command line 傳 list of dict
rosrun ... "_obstacles:=[{pos: [1,2], radius: 0.5}]"  # shell 可能吃掉

# 正確：YAML load
rosparam load cbf_params.yaml /fleet_manager_cbf

# 最終：程式直接讀 YAML
_CFG = yaml.safe_load(open("cbf_params.yaml"))
```

**關鍵教訓**：`rospy.get_param("~xxx")` 的 `~` 是 private namespace，對應 node name（`rospy.init_node("fleet_manager_cbf")` → `/fleet_manager_cbf/xxx`），不是檔案名。

### 問題 2: Follower 直接撞牆（左牆有洞口）

**症狀**：follower 衝向左牆，完全沒有避開的跡象。

**原因**：半平面 wall CBF 是「無限長」的約束——它不知道牆上有洞口。如果加了左牆的 wall CBF，狗連洞口都進不去。所以左牆沒有加 CBF，follower 不知道牆的存在。

**解法**：用離散圓形 obstacles 模擬左牆（洞口 y=-0.6~0.6 之間不放），這樣 CBF 知道左牆在哪，但洞口處沒有 obstacle，狗可以通過。

**半平面 wall CBF 的局限**：只能描述「整面牆」，不能描述「牆上有洞」。要描述有洞的牆，需要更複雜的 CBF（e.g. 用兩個半平面的交集），或者像我們這樣用離散圓形近似。

### 問題 3: Follower 卡在牆內不動

**症狀**：leader 走出洞口，follower 留在牆內一動不動。

**原因**：PID 和 CBF 互相打架：
- PID：「leader 在牆外，目標在牆外，往牆方向走！」
- Wall CBF：「離牆太近，不准往牆方向走！」
- QP 的最優解 = 兩力抵消 = 站在原地。

**根本原因**：CBF 是 reactive 的（只看當下），不會規劃「先走到洞口再出去」。follower 不知道洞口在哪。

**解法**：縮小 formation offset（從 `[-1.0, ±1.0]` 改成 `[-0.6, ±0.4]`），讓 follower 不會跟 leader 差太遠，follower 的目標點不會跑到牆壁另一邊。

**長期解法**：加入路徑規劃（Phase 4），讓 follower 知道怎麼繞到洞口。

### 問題 4: CBF 來不及介入（碰撞已發生）

**症狀**：偶爾兩隻狗還是會碰到。

**原因**：`cbf_d_min` 和 obstacle radius 太小，CBF 到邊界才開始介入，但 20Hz 的控制頻率下，一個 cycle（50ms）的距離可能就穿越了邊界。

**解法**：加大 `cbf_d_min`（0.6 → 1.0）和 obstacle radius（0.8 → 1.2），讓 CBF 更早介入。代價是空間利用率降低（狗不能靠太近）。

---

## 第六部分 啟動流程

### 前置安裝（只做一次）

```bash
# 安裝 Python 依賴
pip install --upgrade pip
pip install numpy cvxpy

# 安裝鍵盤遙控
sudo apt install ros-noetic-teleop-twist-keyboard

# 放檔案
cp formation_managerCBF.py \
   /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/
cp cbf_params.yaml \
   /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/
cp obstacle_world.world \
   /root/LeggedControl_ws/src/legged_control/legged_gazebo/worlds/

chmod +x /root/LeggedControl_ws/src/legged_control/legged_controllers/scripts/formation_managerCBF.py

# 改 five_dogs.launch 指向 obstacle_world.world
```

### 每次啟動

```
Terminal 1: roslaunch legged_unitree_description five_dogs.launch
Terminal 2: roslaunch legged_controllers fleet_bringup.launch
Terminal 3: rosrun legged_controllers start_fleet.sh
Terminal 4: rosrun legged_controllers formation_managerCBF.py
Terminal 5: rosrun teleop_twist_keyboard teleop_twist_keyboard.py \
              cmd_vel:=/dog1/cmd_vel_raw
```

### 鍵盤操控

| 按鍵 | 效果 |
|---|---|
| `i` | 前進 (+x) |
| `,` | 後退 (-x) |
| `j` | 原地左轉 |
| `l` | 原地右轉 |
| `Shift+J` | 純側移左 (+y) |
| `Shift+L` | 純側移右 (-y) |
| `k` | 停止 |
| `q` / `z` | 加速 / 減速 |

注意：指令發到 `/dog1/cmd_vel_raw`（不是 `/dog1/cmd_vel`），CBF 修正後再發到 `/dog1/cmd_vel`。

---

## 第七部分 學長的 Swarm-Formation 分析

### 7.1 架構比較

學長 fork 了浙大 FAST-Lab 的 Swarm-Formation（ICRA 2022），適配到機器狗上。

| 項目 | 我的系統 | 學長的系統 |
|---|---|---|
| 編隊方法 | Leader-Follower + PID offset | Graph Laplacian（去中心化） |
| 安全性 | CBF-QP（有數學保證） | 軌跡優化 soft cost（沒有保證） |
| 路徑規劃 | 無（手動操控） | A* global + EGO Planner local |
| 環境感知 | 手動 YAML 座標 | map_server 靜態地圖 |
| 語言 | Python | C++ |

### 7.2 學長的避障方式

學長也是用已知地圖，不是即時感知。他的 `leader_vel.cpp` 訂閱 `/map`（OccupancyGrid），把佔據格子轉成障礙物座標，餵給 CBF。地圖由 `map_server` 載入 `test_map.pgm`。

### 7.3 SwarmGraph 核心

用 Graph Laplacian 描述編隊：
- 鄰接矩陣 A：邊權 = 兩狗距離平方
- 度矩陣 D：每個節點的邊權總和
- 對稱正規化拉普拉斯 L̂：`L̂(i,j) = δ_ij - A(i,j)/√(D(i)D(j))`
- 編隊差異 = `‖L̂ - L̂_des‖²_F`（Frobenius 範數）

優點：去中心化，沒有 leader/follower 區分，所有狗平等。

### 7.4 可以參考的部分

- `map_server` 靜態地圖載入方式（Phase 3 直接可用）
- `leader_vel.cpp` 的 `mapCb`：把 OccupancyGrid 轉成障礙物座標
- 多機 namespace launch 架構

---

## 第八部分 目前的完成狀態與限制

### 8.1 已完成

- [x] 三隻 A1 多機平台（Gazebo + ROS + OCS2 MPC/WBC）
- [x] Leader-Follower V 字編隊控制
- [x] 模組化 fleet_manager_cbf.py（六層架構）
- [x] Inter-dog CBF 碰撞迴避（所有狗含 leader）
- [x] 障礙物 CBF 避障（圓柱體）
- [x] 牆壁 CBF 避障（半平面 + 圓形模擬洞口）
- [x] Leader 也受 CBF 保護（cmd_vel_raw → CBF → cmd_vel）
- [x] 編隊自動縮隊通過窄洞口
- [x] 場地內左右移動全程安全
- [x] YAML config 統一管理
- [x] 鍵盤遙控（teleop_twist_keyboard）

### 8.2 目前的限制

1. **障礙物位置靠手動輸入**：CBF 不知道場地長什麼樣，需要人類事先把座標寫在 YAML 裡。
2. **CBF 是 reactive 的**：只看當下，不會規劃路徑。follower 不知道洞口在哪，遇到「目標在牆另一邊」就會卡住。
3. **半平面 wall CBF 不支援洞口**：只能描述整面牆，有洞口的牆要用離散圓形近似。
4. **Leader 的路徑完全靠人操控**：沒有自主導航能力。

---

## 第九部分 下一步研究方向

### Phase 3 — 靜態地圖（map_server）

**目標**：用 `map_server` 載入靜態地圖發布 `/map`，CBF 訂閱 `/map` 自動讀取障礙物。

**改動**：`CBFSafetyFilter` 新增 `mapCallback`，把 OccupancyGrid 的佔據點轉成 obstacle list。不再需要手動寫 YAML 座標。

**可參考**：學長的 `leader_vel.cpp` 的 `mapCb`。

### Phase 4 — 路徑規劃（move_base / A*）

**目標**：leader 給目標點自動導航，不再需要鍵盤操控。

**做法**：用 ROS 的 `move_base`（內建 A* + DWA local planner），或參考學長的 EGO Planner。

**架構改動**：planner 發 nominal → CBF 仍在最後一層保護。

### Phase 5 — SLAM（gmapping + LiDAR）

**目標**：狗在未知環境裡邊走邊建圖。

**做法**：Gazebo 加 2D LiDAR sensor + `gmapping`。SLAM 發 `/map`，CBF 和 planner 都自動用即時地圖。

### Phase 6 — CBBA 任務分配（核心研究）

**目標**：多個目標點，三隻狗用 CBBA 拍賣機制分配「誰去哪」。

**這是研究的創新點**，學長的 code 裡沒有。

### Phase 7 — VLA

**目標**：自然語言指令 → 任務目標 → CBBA 分配。

---

## 第十部分 Future Work Prompt

以下是可以直接貼給 Claude 用的 prompt，延續本專案的上下文：

---

### Prompt: Phase 3 — 接 map_server

```
我的 fleet_manager_cbf.py 目前的障礙物和牆壁座標是手動寫在 cbf_params.yaml 裡的。
我想改成用 map_server 載入靜態地圖（.pgm + .yaml），
CBF 訂閱 /map (nav_msgs/OccupancyGrid) 自動讀取所有佔據格子，
不再需要手動算座標。

需要你幫我：
1. 把我的 obstacle_world.world 場地轉成一張 .pgm 地圖
2. 在 fleet_manager_cbf.py 的 CBFSafetyFilter 加入 mapCallback，
   把 OccupancyGrid 轉成 obstacle list
3. 更新 launch file 加入 map_server node

學長的 leader_vel.cpp 的 mapCb 可以參考，但要改成 Python 版。
我的平台 spec 在 multi_robot_final_notes.md 裡。
```

### Prompt: Phase 4 — 路徑規劃

```
我的 fleet_manager_cbf.py 目前 leader 靠鍵盤 (teleop_twist_keyboard) 手動操控。
我想讓 leader 可以自動導航到指定目標點。

需要你幫我：
1. 整合 ROS move_base 到我的系統裡（或用更簡單的 A* + pure pursuit）
2. Planner 發 nominal cmd → CBF 仍然在最後一層保護安全
3. 支援 RViz 點選目標點 (2D Nav Goal)
4. Follower 仍然用現有的 formation PID 跟隨

前提：Phase 3 的靜態地圖已經接好（/map topic 可用）。
我的平台 spec 在 multi_robot_final_notes.md 和 cbf_notes.md 裡。
```

### Prompt: Phase 5 — SLAM

```
我想讓機器狗在未知環境裡邊走邊建圖。

需要你幫我：
1. 在 Gazebo 的 Unitree A1 URDF 裡加入 2D LiDAR sensor
2. 配置 gmapping（或 cartographer）做 SLAM
3. SLAM 發布的 /map 自動被 CBF 和 path planner 使用
4. 確認 multi-robot SLAM 的 tf frame 不衝突

我的底層平台 spec 在 multi_robot_final_notes.md，
CBF 架構在 cbf_notes.md。
```

### Prompt: Phase 6 — CBBA

```
我想在我的三隻機器狗平台上實作 CBBA (Consensus-Based Bundle Algorithm) 任務分配。

場景：場地裡有多個目標點（巡邏點 / 物資運送點），
三隻狗用分散式拍賣決定「誰去哪個目標」。

需要你幫我：
1. 讀 CBBA 原始論文 (Choi, Brunet, How 2009) 的核心數學
2. 實作 CBBA core：auction phase + consensus phase
3. 整合到我的 fleet_manager_cbf.py 架構裡
4. CBBA 分配完後，每隻狗用 path planner 導航 + CBF 保護安全

我的平台 spec、CBF 架構、path planner 在之前的筆記裡。
```

---

## 附錄 CBF 除錯速查

| 症狀 | 可能原因 | 解法 |
|---|---|---|
| obstacles=0, walls=0 | YAML 沒讀進去 | 確認 cbf_params.yaml 在同目錄 |
| follower 撞牆 | 左牆沒有 CBF | 用離散圓形 obstacles 模擬 |
| follower 卡住不動 | PID 和 CBF 打架 | 縮小 formation offset |
| 狗還是會碰撞 | d_min / radius 太小 | 加大安全距離 |
| QP solver failed | Constraint 矛盾（無解） | 檢查參數，放寬 d_min |
| CBF 沒介入 | cbf_enabled=False | 確認 YAML 裡 cbf_enabled: true |
| leader 不受保護 | 用舊版程式 | 確認用的是最新版（含 LeaderCmdRelay） |
| leader cmd 無反應 | 發到 /cmd_vel 而不是 /cmd_vel_raw | 改發 /dog1/cmd_vel_raw |
