# 開源參考導航 — ADMM-CBF-DMPC 實作

> **這份文件是「地圖」不是「規格」。** 它告訴你每個實作步驟去哪份開源找參考、該借什麼、
> 該避什麼坑。規格(必須遵守的成品要求)看 `文件二_架構藍圖.md` 和 `spec_updates_v2.md`。
>
> **黃金原則:開源是參考實作,不是照抄對象。** 每份開源都有「跟我們需求不符」的部分,
> 照抄會踩雷。這份文件的核心價值就是標出每份「借什麼、避什麼」。

---

## 開源分三類用途(先分清楚,別搞混)

| 類型 | 開源 | 怎麼用 |
|---|---|---|
| **數學對拍型** | `NMPC-DCLF-DCBF`, `MPC-CBF` (MATLAB) | **只讀不跑**。寫完 CBF code 回去比對係數對不對 |
| **結構骨架型** | `consensus_ADMM` (Python) | **讀結構、仿寫、但要改**。ADMM 骨架可仿,consensus 那步要換 |
| **介面範本型** | `legged_planner` (ROS C++), 本 repo | **讀介面、對接時照著接** |

**規格書 vs 參考實作**:
- `文件一` = 論文數學真相;`文件二` + `spec_updates_v2` = 你架構的施工規格
- 開源 = 參考實作。規格說要什麼,開源示範怎麼寫。**衝突時以規格為準。**

---

## references/ 目錄下的開源清單

```
references/
├── consensus_ADMM/          # ADMM 三步驟骨架 (Python)
├── MPC-CBF/                 # 單機 double-integrator + 一階 CBF (MATLAB)
├── NMPC-DCLF-DCBF/          # 離散二階 HOCBF + 每步線性化 (MATLAB) ⭐
└── legged_planner/          # OCS2 對接 adapter (ROS C++)
```

clone 指令:
```bash
mkdir -p references && cd references
git clone --depth 1 https://github.com/RandyChen233/consensus_ADMM.git
git clone --depth 1 https://github.com/HybridRobotics/MPC-CBF.git
git clone --depth 1 https://github.com/HybridRobotics/NMPC-DCLF-DCBF.git
git clone --depth 1 https://github.com/AndrewZheng-1011/legged_planner.git
```

---

## 逐份開源:借什麼 / 避什麼(最重要)

### ⭐ NMPC-DCLF-DCBF — 離散二階 HOCBF 的黃金對拍

- **關鍵檔**:`references/NMPC-DCLF-DCBF/matlab/acc2023/closedloop_performance/NMPCDCBF2.m`
- **對照檔**(一階版,看差異):同目錄 `NMPCDCBF1.m`
- **借什麼**:離散二階 CBF 的**三點差分結構**。核心是它的第 99-103 行:
  ```matlab
  b        = ‖x_i   - obs‖² - r²        % h(t)
  b_next   = ‖x_i+1 - obs‖² - r²        % h(t+1)
  b_next_next = ‖x_i+2 - obs‖² - r²     % h(t+2) ← 二階要用到 t+2
  b1      = (b_next - b)/dt + (γ1/dt)·b
  b1_next = (b_next_next - b_next)/dt + (γ1/dt)·b_next
  約束: b1_next ≥ (1-γ2)·b1  且  b_next ≥ (1-γ1)·b
  ```
- **⚠️ 避什麼坑**:
  1. **它的 γ 是 adaptive**(第 103 行的 `u(3),u(4)` 是把 decay rate 當可變決策變數)。
     **我們第一版用固定 γ**,把 `u(3),u(4)` 換成常數 1。別照抄 adaptive,多一層複雜度。
  2. **它只處理單一障礙**(`pos=obs.pos1`)。我們有多障礙 + 牆 + 鄰居,每個各一條 CBF row。
  3. **它是 obstacle CBF(node-local)**。我們的 inter-agent 要把同結構套到「兩狗之間」並搬進 edge。
- **用途定位**:寫完 C3 二階 CBF 後,回來比對係數。**這是數學對拍,不是移植。**
- **注意**:我們用**前向差分展開**(見 `文件二 C3` 和 `spec_updates_v2 細節1`),
  最終約束係數是 `3Ts²`,跟這份 MATLAB 的 `(b_next-b)/dt` 形式等價但寫法不同。以我們文件為準。

### consensus_ADMM — ADMM 三步驟骨架

- **關鍵檔**:`references/consensus_ADMM/admm_mpc.py`(主迴圈)、`dynamics.py`(double integrator)、`util.py`(graph)
- **借什麼**:
  1. ADMM 三步驟迴圈的**組織方式**:gather → update → scatter
  2. **primal/dual residual 量測**的寫法(收斂判據)
  3. `dynamics.py` 的 double integrator 離散化思路
  4. `util.py` 的 neighbor graph threshold(含連通性保護)——我們是固定 complete graph 可簡化
- **⚠️ 避什麼坑(最重要)**:
  1. **它是 global consensus,不是 node-edge splitting!** 它的核心是 `xbar = sum(x_all)/N`
     (所有 agent 對同一個全局平均取共識)。**我們要的是 node-edge**(每條 edge 各有 local copy,
     只對鄰居求和 `Σ_{j∈N(i)}`)。**consensus 那步絕對不能照抄**,要照 `文件一 4.3` 和 `文件二 C5` 改寫。
  2. **它用 multiprocessing.Pipe** 做單機多進程通訊。我們偽分散式階段可用,真分散式要換 ROS topic。
  3. **它 rho=0.5**(第 217 行)。我們用 `rho=20`(論文值)。別照抄 0.5,差 40 倍。
  4. **它收斂判據是「目標值變化<0.05」**。我們用**固定 15~20 iterations**(即時性),不等收斂。
  5. **它用 CasADi opti,每 iteration 重建符號圖**(慢)。我們用 **OSQP 直接 binding + warm-start**,
     預組 sparsity pattern,每輪只 update 數值。**別學它每次重建。**
  6. 第 299 行的 send/recv 是 logging 用(它註解明說「not part of ADMM」),別把 debug 通訊也抄進去。
- **用途定位**:仿它的迴圈骨架,但 consensus 邏輯換成 node-edge。這是「仿寫 + 大改」。

### MPC-CBF — 單機 baseline 對拍

- **關鍵檔**:`references/MPC-CBF/examples/MPCCBF.m`
- **借什麼**:最簡單的「4-state double integrator + 一階離散 CBF」baseline。
  `x=sdpvar(4,N+1), u=sdpvar(2,N)`,CBF 是 `b_next - b + γ·b ≥ 0`。跟我們單狗 node 結構最接近。
- **⚠️ 避什麼坑**:
  1. 它是**一階** CBF(位置型 $h$ 對加速度的 relative degree 在單步 MPC 被吸收)。
     我們要**二階**(見 NMPC-DCLF-DCBF)。一階在 multi-step multi-agent 會退化。
  2. 它是集中式單機,沒有 ADMM、沒有 edge。
- **用途定位**:階段 1(單狗 node QP)的最簡對拍。先驗證單機 CBF 對,再往上加二階、加 ADMM。

### legged_planner — OCS2 對接 adapter 範本

- **關鍵檔**:
  - `references/legged_planner/legged_body_planner/src/robot_command/RobotPlanCommand.cpp`
  - `references/legged_planner/legged_body_planner/src/motion_adapters/LeggedRobotAdapter.cpp`
- **借什麼**:「外部軌跡 → ocs2::TargetTrajectories」的完整轉換流程、維度 padding 寫法、
  COM_HEIGHT 補法、control trajectory 回零的做法。它有 `RobotPlanCommand`(送整條軌跡)
  正是我們要的模式(不是 RobotGoalCommand / RobotVelocityCommand)。
- **⚠️ 避什麼坑(關鍵)**:
  1. **它的預設 `toLeggedState` 是「直接平鋪」**(`for i<numStates: leggedState[i]=state[i]`)。
     它假設高層 state 順序跟 OCS2 一樣(速度在前)。**我們的 double integrator 是位置在前
     `[px,py,vx,vy]`,順序跟 OCS2 相反!** 直接用預設會全部錯位。
     **必須自訂 adapter**,照 `文件二 C6.2` 的映射表填(vx,vy→idx 0,1;px,py→idx 6,7)。
  2. **它有個 bug-like 寫法**:`adaptMotion` 的 `checkEqualSize` 檢查寫了兩次同樣的(複製貼上沒改)。
     自己寫時要確保 times/state/control 三者長度一致。
  3. **它的 z 寫死 COM_HEIGHT**——平地 OK,rough terrain 底層要自己補地形適應。
- **用途定位**:階段 4(接 OCS2)的介面範本。照它的結構寫,但 padding 順序要自訂。
- **待挖**:OCS2 訂閱端(RosReferenceManager)、Issue #25(接法)、#49(維度不匹配會炸)——
  做階段 4 時再深挖。

### 本 repo (legged_upper_control) — 你已有的資產

- **關鍵檔**:
  - `core/formation.py` 的 `LaplacianFormation.grad_p`(編隊解析梯度,C2 直接用)
  - `controllers/` 的二階 HOCBF 實作(你已有正確的 chain-rule 處理,C3 可移植)
  - `core/planning.py` 的 A*(C1 reference 生成)
  - `fleet/manager.py` 的 `_goal_slot_targets`(slot 機制)
- **借什麼**:編隊梯度、二階 CBF row 組裝、A*、slot,都已經有。
- **⚠️ 避什麼坑**:
  - 大量 **reactive 補丁**(speed threshold、slot freeze、stuck recovery、door mode…)
    在 ADMM 後多數多餘。見 `spec_updates_v2` 的「reactive 補丁拆除清單」。
    **別把這些補丁帶進 ADMM 版**,會跟軌跡優化打架。
  - `scripts/archive/` 是舊版,別誤啟動。

### grampc-d — 之後 C++ 化才用(現在不碰)

- **repo**:`github.com/grampc/grampc-d`(C++ ADMM distributed MPC)
- **借什麼**:C++ ADMM 框架、**neighbor approximation**(加速收斂:7 vs 89 iterations)、通訊協定。
- **時機**:Python 驗證完、要 C++ 化時。**現在不用 clone,記著它存在。**

---

## 完整施工圖(階段 0-5)

每階段標:產出物、參考哪份開源、驗證標準。**紅色(node-edge)要建在綠色地基上,別跳。**

### 階段 0:數學守門層(地基)
- **產出**:`constants.py`(Ad, Bd, 係數, index 公式)+ `test_constants.py`
- **參考**:`文件二 全局設定` + `spec_updates_v2 細節1,2`(無開源,純規格)
- **驗證**:單元測試全綠。γ1γ2 正號(0.49)、Bd 係數(0.015)、約束係數(0.03)
- **為什麼先做**:四個敲定細節裡三個是常數,焊死後組矩陣不用再擔心係數

### 階段 1:單狗 node QP(不碰 ADMM)
- **產出**:`node_subproblem.py` + 單狗測試
- **參考**:
  - 數學對拍:`MPC-CBF/MPCCBF.m`(結構)、`NMPC-DCLF-DCBF/NMPCDCBF2.m`(二階係數)
  - QP 組裝:OSQP Python docs(**用 OSQP 不用 cvxpy**)
  - 移植:本 repo 的二階 CBF row 組裝
  - 文件:`文件二 C1, C4.1-C4.4`
- **驗證**:單狗追 A* 到終點、繞開障礙、h≥0 全程不破
- **關鍵**:用 OSQP 直接 binding,一開始就用對的寫法

### 階段 2:兩狗 ADMM(node-edge splitting 核心,唯一原創)
- **產出**:`edge_subproblem.py` + `admm_coordinator.py` + `rti_linearizer.py`
- **參考**:
  - 骨架:`consensus_ADMM/admm_mpc.py`(仿三步驟,但 consensus 換 node-edge)
  - 數學真相:`文件一 Part 4`(4.3 增廣拉格朗日、4.6 初始化)——唯一來源
  - z 資料結構:`文件二 C5.3`(按 edge 存不按 node 存)
  - 文件:`文件二 C3, C4.5, C5`
- **驗證**:
  - 兩狗對向走互相避開不撞
  - **r_prim 隨 ADMM 輪數下降**(node-edge 正確的鐵證)
  - h2_viol 收斂後接近 0
- **這階段最易錯**(沒範本)。**先兩狗跑通(一條 edge)再加第三狗。**

### 階段 3:三狗 + 編隊
- **產出**:coordinator 擴展到 3 node 3 edge + `formation_gradient.py`
- **參考**:本 repo `formation.py` 的 grad_p、`文件二 C2`
- **驗證**:三狗保持 V 形、過障礙柔性變形、r_prim 仍收斂
- **注意**:第一週期 w_form=0(spec_updates_v2 細節3)

### 階段 4:接 OCS2
- **產出**:`motion_adapter.py`(自訂 padding + yaw 旁路)
- **參考**:`legged_planner` 的 RobotPlanCommand + LeggedRobotAdapter、`文件二 C6`
- **驗證**:Gazebo 三狗照 ADMM 軌跡走、底層追得上、不抖
- **待挖**:OCS2 Issue #25, #49(做這步時挖)

### 階段 5:調參 + 清理
- **產出**:調 ρ/N/Q/iterations、拆 reactive 補丁
- **參考**:`文件一 Part 6`(ρ 調參)、`spec_updates_v2` reactive 拆除清單
- **驗證**:三個 Gazebo 實驗(cluttered / disturbance / rough terrain)

---

## 一頁速查表

| 階段 | 產出 | 主要開源 | 用法 | 主文件 |
|---|---|---|---|---|
| 0 守門 | constants+test | (無) | 純規格 | 文二全局, spec細節1,2 |
| 1 單狗node | node QP | MPC-CBF, NMPCDCBF2, 本repo | 對拍+移植 | 文二C1,C4 |
| 2 兩狗ADMM | edge+coord | consensus_ADMM(仿改), 文一Part4 | 骨架+數學 | 文一4.3, 文二C3,C5 |
| 3 三狗編隊 | +formation | 本repo formation.py | 直接用 | 文二C2, spec細節3 |
| 4 接OCS2 | adapter | legged_planner | 範本 | 文二C6 |
| 5 調參 | 全系統 | 文一Part6, ACLM | 指南 | 文一6, 文二附錄C |

---

## 貫穿全程的規則

1. **驗證階梯**:每階段驗證是下階段的門檻。階段 N 不過別進 N+1。
   紅色 node-edge(階段2)一定建在綠色地基(0,1)上,出 bug 才知範圍。
2. **Python 先行,C++ 風格寫**:全程 Python,但矩陣組裝用明確 index 迴圈(像 MATLAB 參考),
   不用 Python 動態魔法。之後翻 C++(參考 grampc-d)幾乎一對一。
3. **數學對拍隨時做**:每寫涉及係數的 code,回去比對 NMPCDCBF2.m 或跑守門測試。
   係數錯不會崩,只會行為微妙不對,很難事後 debug,當場擋掉最省事。
4. **OSQP 不用 cvxpy**:即時迴圈裡 cvxpy 每次 canonicalize 太慢。直接 osqp binding,
   預組 sparsity,每輪 update 數值 + warm_start=True。
5. **衝突以規格為準**:開源跟文件一/二/spec 衝突時,聽文件的,開源只是參考。
