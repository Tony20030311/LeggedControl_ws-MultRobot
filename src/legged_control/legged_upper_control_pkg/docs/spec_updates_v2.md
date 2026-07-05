# 規格更新 v2 — 文件二補丁

> **這份是「規格」不是「參考」。** 這裡的每一條都是**必須遵守**的成品要求,不是可選建議。
> 內容 = 對話中敲定、但還沒進文件二的決定,加上討論出的隱藏坑(不在論文、不在任何開源裡)。
>
> **與文件二的關係**:文件二是主藍圖,這份是補丁。衝突時**這份 v2 優先**(它更新)。
> 涉及係數的規格,`test_constants.py` 必須守門。

---

## A. 四個敲定細節(取代文件二對應項)

### 細節 1:Bd 統一精確版(廢除前向歐拉簡化)

**規格**:
- 全 code **只用一個精確 Bd**,C3(CBF 線性化)和 C4.3(動態約束)共用,**不留兩套離散化**
- ```
  Ad = [[I2, Ts·I2], [0, I2]]        (4×4)
  Bd = [[½Ts²·I2], [Ts·I2]]           (4×2)
  ```
- **a_k 對 p_{k+2} 的精確係數 = 3/2·Ts²**,推導:
  - 路徑1:Bd 直接進 p_{k+1} → ½Ts²
  - 路徑2:Bd 進 v_{k+1},再被 Ad 傳一步進 p_{k+2} → Ts·Ts = Ts²
  - 合計 ½Ts² + Ts² = **3/2·Ts²**
- **CBF 約束係數**:col_a = **-3Ts²·ē_{k+2}**, col_b = **+3Ts²·ē_{k+2}**
  (來自 h 的線性化 ∇‖e‖²=2e,乘上 3/2Ts² → 2·3/2Ts² = 3Ts²)
- **Ts=0.1 驗算**:3/2·Ts² = 0.015, 3Ts² = 0.03

**作廢**:文件二 C3 原本寫的 `p_{k+2}=p_k+2v_kTs+a_kTs²`(前向歐拉簡化,係數 Ts²)整段作廢。
全部重算成精確版上述係數。

**最終 edge 約束**(更新文件二 C3 的框):
```
-3Ts²·ē_{k+2}ᵀ·a^i_k + 3Ts²·ē_{k+2}ᵀ·a^j_k ≤ b_k + s^{ij}_k
```

### 細節 2:γ1γ2 正號 + 單元測試守門

**規格**:
- b_k 常數項的 h_k 係數 = **(1 - γ1 - γ2 + γ1γ2)**,γ1γ2 **正號**
- ```
  b_k = c_{k+2} - (2-γ1-γ2)·h_{k+1} + (1-γ1-γ2+γ1γ2)·h_k
  ```
- γ1=γ2=0.3 時:h_k 係數 = 0.49(**負號誤植會變 0.31,差 37%**)
- **寫 code 第一個 unit test 守此**:

```python
def test_hocbf_coefficients():
    g1 = g2 = 0.3
    coef_hk  = 1 - g1 - g2 + g1*g2      # 必須 = 0.49
    coef_hk1 = -(2 - g1 - g2)           # 必須 = -1.4
    assert abs(coef_hk  - 0.49) < 1e-9, f"h_k 係數錯! {coef_hk} 應為 0.49(正號)"
    assert abs(coef_hk1 - (-1.4)) < 1e-9

def test_bd_coefficients():
    Ts = 0.1
    assert abs(1.5*Ts**2 - 0.015) < 1e-9   # a_k → p_{k+2}
    assert abs(3.0*Ts**2 - 0.03)  < 1e-9   # 約束係數
```

### 細節 3:編隊第一週期關閉(選 A)

**規格**:
- **第一週期 w_form = 0**。原因:無耦合 MPC 當 z0 本就不含編隊,編隊梯度的操作點無來源
  (梯度要「未來每步鄰居位置」= 上週期解,第一週期沒有)
- **第二週期起 w_form 正常**(此時有上週期解可當編隊梯度操作點)
- 實作:`if cycle_count == 0: w_form = 0`

### 細節 4:N=20, Ts=0.1(涵蓋 2 秒)

**規格**:
- **N = 20**(過二階前瞻下界 19,不照抄論文 50)
- Ts = 0.10 s(10 Hz 高層)
- **dim ξ^i = 6N = 120**
- **index 速查(N=20 版,取代文件二 N=10 版)**:
  ```
  x_k 起始 index = 4(k-1),  k=1..20
  x_1 @ 0-3,  x_20 @ 76-79
  a_k 起始 index = 4N+2k = 80+2k,  k=0..19
  a_0 @ 80-81,  a_19 @ 118-119
  ```
- **⚠️ index 用 4*N 變數,不要 hardcode 數字**(N 從 10 改 20,寫死的 40 會全錯)
- edge QP 維度 ≈ 12N + n_s = **240 + n_s**
- **連帶影響**:QP 變大 → warm-start + OSQP 直接 binding(非 cvxpy)從「建議」升級成「必須」

---

## B. 隱藏坑清單(不在論文、不在開源,只存在於討論)

這些是討論中發現、任何開源和論文都沒有的坑。**必須遵守。**

### B1. z 資料結構:按 edge 存,不按 node 存

```python
# ✅ 正確:z[edge][endpoint]
z = {(1,2):{1:vec, 2:vec}, (1,3):{1:vec, 3:vec}, (2,3):{2:vec, 3:vec}}
# 狗1 有 z[(1,2)][1] 和 z[(1,3)][1] 兩份分身 → 對應 |N(1)|=2

# ❌ 錯誤:z[node] 會混淆不同邊的分身
# node update 的 Σ_{j∈N(i)} 會求和錯
```
**原因**:一隻狗屬於多條邊,每條邊有獨立分身。按 node 存會把不同邊的分身混在一起,
node update (19) 的鄰居求和就錯了。

### B2. OCS2 adapter 順序與 OCS2 相反(必須自訂)

- 你的 double integrator:`[px, py, vx, vy]`(**位置在前**)
- OCS2 centroidal state:`[vx, vy, vz, ...動量..., px, py, pz, ...]`(**速度在前**)
- **兩者順序相反!** legged_planner 預設「直接平鋪」會全部錯位
- **必須自訂 adapter**,映射:
  ```
  vx → OCS2 idx 0,  vy → idx 1
  px → OCS2 idx 6,  py → idx 7
  z  → idx 8 = COM_HEIGHT(強制蓋,從 reference file 載入)
  yaw → idx 9(yaw 旁路),yaw_dot → idx 3
  其餘 → 0,  後 12 維 → DEFAULT_JOINT_STATES
  控制軌跡不送(toLeggedControl 回零)
  ```
- 詳見文件二 C6.2 映射表

### B3. formation 進 cost 不進 constraint

- formation 是 J^i 的一個加項(跟 tracking/control/terminal 並列),**進 node cost**
- **絕不進 Ξ^i(約束集)**。軟目標當硬約束會 infeasible(編隊想靠近、CBF 想分開,打架)
- 用 grad_p 線性化成 node 的線性項(all-to-all 耦合被凍進常數)
- 鄰居位置用 ADMM 上週期估計當常數 → 對 ξ^i 而言是 node-local 線性項

### B4. formation all-to-all 不可 pairwise 拆

- 正規化 Laplacian 的分母 √(D_ii·D_jj) 含第三隻狗 → ∂f/∂p_i 依賴全部狗
- **f 不能拆成 f^12+f^13+f^23 → 裝不進 edge**(edge 只吃兩端)
- 唯一正解:線性化 → 塌成 node 常數梯度 → 進 node(見 B3)
- **別試圖把 formation 放進 edge**,數學上不可分

### B5. 減速放參考層,不放 limiter

- v_ref(d) = v_cruise · min(1, d/d_brake),d = **沿 A* 路徑到終點的剩餘弧長**(不是直線距離)
- 直線距離繞牆時會誤判提早停
- **放參考層(生成 x_des 時),不放 QP 出口的 limiter**
- 原因:planner 自己知道要慢 → 解出的軌跡本身平滑煞停,不需出口拔河
  (避免 near-wall 橫向振盪)

### B6. v_des 是切向速率,不是回授律

- v_des_k = v_ref · t̂_k(路徑切向 × 巡航速率)
- **不是** K·(p_des - p_now)(那是「想停在 p_des」的比例控制律,把回授混進參考)
- 中途 waypoint 應「通過」不是「停」,只有**終點** waypoint 的 v_des 設 0
- **但注意**:我們決定 reference 只追位置(Q 速度維度=0),所以 v_des 可能根本不填。
  先只追位置,demo 有問題再考慮填 v_des。

### B7. reference 只追位置(Q 速度維度 = 0)

- Q = diag{q_px, q_py, **0, 0**}(速度維度不 penalize)
- 對齊論文(論文只追位置/朝向,不追速度)
- 速度由「追位置 + 最小化加速度 + 動力學約束」自然產生,平滑、終點歸零
- terminal P 可以給速度一點權重讓終點停穩:P = diag{10q_px, 10q_py, q_v, q_v}
- **好處**:不用算 v_des,速度疑慮從根本消失

### B8. 線性化在 ADMM 迴圈外做一次

- RTI/successive linearization:每控制週期**迴圈外**線性化一次,凍結係數
- 凍結:CBF 係數(3Ts²ē)、編隊梯度(g_{i,k})、Ad/Bd、b_k
- ADMM 15~20 輪內共用這組凍結係數(迭代間狀態幾乎不動,且線性化貴)
- 操作點 = 上週期解平移一格

### B9. 不做兜底/watchdog,改埋 h2_viol 量測

- slack 開著(cbf_slack_enabled=True)→ QP 不會 infeasible → 舊的 emergency brake 是 dead code
- **不做** watchdog/emergency brake
- 改成量測 h2_viol = max(-h_2^{ij}(真身 ξ))^+,上報 /admm/h2_viol topic
- 用收斂後**真身** ξ 代回 h_2≥0 量負值 = 要不要兜底的誠實依據
  (誠實框架:slack>0 該步無嚴格保證,量化上報)

---

## C. reactive 補丁拆除清單(ADMM 後多餘)

現在 repo 的這些 reactive 補丁,是「集中式 reactive QP」時代堆的。ADMM 軌跡優化(proactive)後多餘,
**別帶進 ADMM 版,會跟軌跡優化打架**。

**建議做法**:ADMM 先跑通(保留補丁當驗證期保險)→ 驗證 ADMM 行為正常 →
逐個關掉補丁,每關一個測一次 → 全拆完 config 乾淨化。

### 🔴 ADMM 後應拆除
| 補丁 | 為何多餘 |
|---|---|
| accel_injection(scale=0) | 已棄,對接改 TargetTrajectories 後整機制刪 |
| cbf_gamma_vel, cbf_gamma_vel_pair | 已關速度層 CBF,純二階 |
| cbf_lookahead_tau | 二階不用 lookahead |
| emergency_brake_time | reactive 急煞,ADMM 提前規劃安全軌跡(但驗證期可留當保險) |
| stuck_*(4個) | reactive 卡住補救,ADMM 軌跡優化本身找可行路 |
| slot_freeze_projection_threshold | reactive slot 凍結,ADMM 軌跡連續優化不需要 |
| target_projection_* | reactive 推 target 離障礙,node CBF 直接保證 |
| formation_guard_*(3個) | reactive 隊形散了減速,formation cost 直接優化 |
| kp_pos_follower, kp_formation_soft, formation_soft_max_speed | reactive P-control 編隊,被 formation cost 取代 |
| nominal_smooth_alpha, pp_*(Pure Pursuit) | 已決定丟 pure pursuit |
| door_*(6個), door_mode_enabled | DoorPassageManager + FormationSwitcher 已棄 |
| forbidden_zones_*, astar_block_narrow_wall_gaps | 手寫禁區,ADMM+牆CBF 處理窄縫 |

### 🟡 保留但重調
| 參數 | 怎麼改 |
|---|---|
| cbf_gamma1, gamma2 | 核心保留,horizon 版重調 |
| cbf_d_min | 保留,約束從 node **搬到 edge** |
| slack_lambda | 保留且更重要(edge 的 slack 變數,paper Eq.13) |
| w_track, w_formation, w_accel | 保留,重設:w_goal(terminal)大、w_formation 小、w_track 中 |
| prediction_*, w_pred, w_smooth | ADMM 後變真正的 horizon 參數,啟用並重設 |
| rate: 50 | 上層降到 10Hz |
| max_v*/a* | 保留(QP box constraint),但 formation_soft_max_speed 保守值放開 |

### 🟢 保留不動
leader_name/follower_names, formations(V/line offset), default_formation,
robot_footprint_*, obstacles/rect_obstacles/walls, map_*/astar_*,
kp_yaw, **goal_yaw_freeze_dist/goal_yaw_ema_alpha**(yaw 旁路搬 per-dog),
*_topic, debug_*, per_dog_astar_*, dynamic_slot_assignment

---

## D. yaw 旁路(方案 C:速度方向 + 低速凍結 + EMA)

- **不進 ADMM**(yaw 不是決策變數),用 P control + 旁路
- ```
  ψ_k = EMA(atan2(v_{y,k}, v_{x,k}))       # 頭朝移動方向
    低速凍結: |v_k| < 閾值 → 沿用前值
    EMA/slew: 防瞬跳
  ψ̇_k = wrap_to_pi(ψ_{k+1} - ψ_k) / Ts     # 差分,過 wrap_to_pi
  ```
- 塞 OCS2 idx 9(ψ)、idx 3(ψ̇)
- **⚠️ EMA 必須用 wrap_to_pi 包差值**,否則 ±π 邊界(179°→-179°)會暴衝
- repo 已有雛形:goal_yaw_freeze_dist, goal_yaw_ema_alpha(從 goal 層搬到 per-dog)

---

## E. 效能要求(N=20 連帶)

1. **用 OSQP 直接 binding,不用 cvxpy**(cvxpy 每次 canonicalize 太慢,10Hz 跑不動)
2. **預組 sparsity pattern**,每輪只 `osqp.update(q=, l=, u=)` 更新數值,不重建
3. **edge QP 一定 warm-start**(240 維,比 node 大,warm_start=True + 上輪 z,s 當起點)
4. Python 階段用「C++ 風格」寫(明確 index 迴圈),之後翻 C++ 幾乎一對一

---

## F. 待驗證清單(Gazebo,不預先優化)

- N/Ts 是否夠(前瞻距離)
- d_brake 值
- Q 速度權重(先 0,demo 有問題再調)
- ρ(殘差壓不下加大)
- ADMM 迭代輪數(看 r_prim 曲線)
- 轉角是否需 Bézier(先不加,抖再加)
- h2_viol 幅度(決定是否需要兜底)
- 精確 Bd 係數 3Ts² 在真機的表現

## G. edge CBF handoff 安全(已驗,2026-07)

**背景**:整條 2s ξ 當 OCS2 x_ref(軟成本、無 inter-agent CBF、無 staleness guard),對衝時 OCS2
會追軟約束違反的未來步 → 撞。

**關鍵發現(verify_edge_handoff.py + 7 次 Gazebo 實測)**:
- **不能全硬(推翻「回到 paper 全硬」)**。node-edge 拆分下,硬 inter-agent CBF **只撐 ~2 個 active
  binding 步**;硬約束遠端步要的相對加速度 `~u_k/g_k` 隨 `g_k=0.03·e→0`(逼近預測交會)爆掉,超過
  `a_max=1.0`(A1 物理),node 追不上 z → dual 捲 → edge QP primal infeasible → plant 撞。已排除
  C++/Python、a_max(×50)、bounds/迭代(×20/×5)——是 active 步數,非宣告步數。論文 50 全硬能跑,只因
  其場景 active 步 ~1-2。**所以藍圖的「k=0 硬 / k≥1 軟」是對的,不是要修回硬。**
- **修法 = 動態安全前綴截斷**(不是全硬、不是固定截斷)。每 cycle 只送「還 ≥ D_MIN(+`send_margin`)
  的前幾步」給 OCS2(`motion_adapter.safe_prefix_length` → `build_target(k_send=)`,fleet publisher 用三狗
  ξ 算),其餘 OCS2 zero-order-hold clamp 到最後安全點 → 狗放慢蠕行過。固定截斷 K_SEND=10 只把 ref 從
  −0.40 改到 −0.30(仍破);動態截斷 → sent ref 恆 ≥ D_MIN(離線 +0.001),最壞前綴縮到 4 步。
- **Gazebo 驗**:2/3/6.7m 對衝、180° 轉彎折返、三狗同時交叉(3 edge active)、V 編隊全過,交叉 min
  pairwise 一致落 0.58–0.60(D_MIN 0.6,離散 CBF 固有 ~1-2cm 餘裕,物理接觸 0.35→安全),無撞無死鎖。
  OCS2 追速度準(實際≈指令)、巡航 0.3;「慢」是交叉蠕行(設計)+ 到 goal 待命的平均假象,非缺陷。
- `send_margin`(δ)是「reference 安全」旋鈕,**不改 realized 餘裕**(realized 由 CBF D_MIN 決定);δ=0.1
  只把 min 0.591→0.599、卻更慢 → **保持 δ=0**。要嚴格 realized ≥ 0.6 得抬 D_MIN,但會撞 0.7 編隊 offset。
- **未解**:密集場景(梅花樁,多 edge 同時 active)會壓到 2-active 上限,需靠參考規劃壓低 active edge 數。
