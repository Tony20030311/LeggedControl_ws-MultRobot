# 階段 3 — 三狗 complete graph + Laplacian 編隊 + 重啟 node CBF（完成）

> 施工圖見 `docs/opensource_reference.md` 階段 3 + 文件二 C4（編隊塌 node 線性項）+
> `spec_updates_v2.md` B3/B4/B8。建在階段 0/1/2 上:世界座標 double integrator、Xiong
> 離散二階 HOCBF、node-edge splitting、OSQP 直接 binding。
> **通過標準**:`verify_stage3.py` 的 **V / column 兩個核心 gate 全綠** + 四圖目視;
> squeeze 場景是**非-gating 診斷 probe**(死鎖是已知極限,不 gate 階段 3,理由見下)。
> 狀態:**綠**。

## 範圍

2 狗 → 3 狗 + 編隊 + 重啟 node 障礙/牆 CBF。complete graph K3:3 node、3 edge
`(1,2)(1,3)(2,3)`、每 node 2 鄰居。**edge 子問題完全不變**(編隊不進 edge,spec B4)。

## 做了什麼（4 改動,依賴序 A→B→C→D,各 apply anchor patch + 備份 + py_compile + 回歸）

| 檔案 | 改動 |
|---|---|
| `admm/test_admm.py`（改·A） | `+test_three_dog_wiring_generic`:釘死 coordinator 是 edge-generic,`dogs=(1,2,3)` 直接跑,neighbors/z/λ/殘差全自動 |
| `admm/node_subproblem.py`（改·B） | `_q` 加編隊線性項 `+w_form·g`(位置維,k=1..N);`w_form` 建構參數**預設 0 → guard 擋掉 → 位元不變** |
| `admm/admm_coordinator.py`（改·C） | formation **依賴注入** + `w_form`;`_formation_grad(xibar)` 在 ADMM 迴圈**外**凍一次;傳 `formation_grad` 進 `node.solve` |
| `admm/admm_coordinator.py`（改·D） | `__init__` 收 `obstacles/walls` 餵每個 `NodeSubproblem`(階段 2 傳空 list 使 CBF dormant,這裡餵資料重啟) |
| `admm/verify_stage3.py`（新） | 三場景閉迴路 + 2×2 四圖(V/column gating、squeeze probe) |

## 四改動的關鍵定案

1. **改動 A 是純設定,零結構改**:階段 2 的把關 1 審計已確認 coordinator edge-generic
   (`n_neighbors=len(neighbors[i])` 不寫死、`consensus_target` 對含 i 的 edge 求和、`z[edge][endpoint]`)。
   三狗實例化 neighbors 自動 `{1:[2,3],2:[1,3],3:[1,2]}`、z/λ 六份、殘差 `Σ_e Σ_{i∈e}` 六項全自動。
   `__init__` 預設仍 `dogs=(1,2)`,保階段 2 回歸。

2. **編隊是 BARE `+w_form·g` 線性項,不是 tracking 的 `−2Q`**:`−2` 是 `‖x−x_des‖²` 平方展開來的;
   `g=∂f/∂p_i` 已是線性化一階泰勒係數,直接 `q += w_form·g`。**不碰 P/Hessian**(純線性,spec:編隊
   當軟目標塌 node 線性,不當硬約束否則與 CBF 打架 infeasible)。位置維 k=1..N。

3. **formation 依賴注入,保 coordinator/測試 rospy-free**:`core/formation.py` 頂層 `import rospy`。
   coordinator 收注入的 `formation` 物件(只呼叫 `.compute()`),自己不 import → `admm_coordinator`、
   `test_admm` 維持純 numpy。只有 `verify_stage3`(Docker、ROS sourced)建 `LaplacianFormation` 注入。

4. **`formation_grad` 迴圈外凍一次(B8)**:`_formation_grad(xibar)` 對每步 k 取三狗操作點位置 →
   `formation.compute` 拿 per-dog 梯度,回 `{i:(N,2)}` **raw**(w_form 由 node `_q` 疊)。跟 CBF 係數、
   Ad/Bd 一起在 ADMM 15~20 輪外凍。**cycle 0 傳 None**(操作點=無耦合 z0 無編隊資訊,spec 細節 3);
   `w_form=0` 亦全程被 node 的 `self.w_form>0` guard 擋掉。編隊是 all-to-all(第三狗經 normalized
   Laplacian 分母耦合),**不可 pairwise 拆進 edge**,故凍成 node-local 梯度。

## 編隊項有作用 + 符號正確（w_form=0 vs 10 對照）

baseline V(abreast 起 → V,同一起點),疊 `w_form=10` 與 `w_form=0`:

| | 過門後 f_end |
|---|---|
| **w_form=10** | **0.0000** |
| w_form=0（tracking only） | 0.0066 |

tracking 追目標槽已把隊形拉近 V(f→0.0066);編隊項再把穩態誤差壓 ~100× 到 ~0,而且是**降**不是升
→ **確認 `+` 號對**(反了號 f 會升)。over-fitting 檢查:同 abreast 起點,column 單列(不同 `L̂_des`)
f 0.933→0 也綠,證明編隊項不是為單一幾何過擬合。

## 核心 gate 結果（V / column）

| 場景 | 週期 | 到終點 | min inter-agent h（三對） | min node h | r_prim(rep) | r_dual(rep) | f |
|---|---|---|---|---|---|---|---|
| **V** | 200 | R/R/R | +0.0743 | +0.0076 | 3.9e-4→4.4e-15 | 14.6→0.38 | 0.185→0.000 |
| **column** | 222 | R/R/R | +0.0662 | +0.3971 | 1.09→1.9e-3 | 37.1→1.06 | 0.933→0.000 |

- V 隊形天生撐開(最近 0.659 vs D_MIN 0.6),edge 輕度綁定 → r_prim 起始小;**強耦合收斂靠 column**
  (狗擠向中線,r_prim 起 1.09、三數量級乾淨降)。兩者合起來涵蓋強弱耦合。
- V 場景門柱把 **node h 壓到 +0.0076**(真騎障礙邊界);牆全程鬆(如實報:牆只證明 `n_feat=4` 多特徵
  組裝仍 feasible,非壓力測試)。
- 三對 realized h12/h13/h23 全程 ≥0(用 raw `‖pⁱ−pʲ‖²−d²`,非低估 ~11× 的 h2_viol);node CBF h≥0。

## ⚠️ 診斷 probe:過門柔性變形 squeeze → DEADLOCK（有價值發現,非失敗）

**場景**:寬 V(翼 y=±0.8)過窄門(柱 @(0,±1.0) 淨縫 |y|<0.40 < 隊形自然寬),逼隊形必須變形才能過。
三狗起於寬-V、目標寬-V,直線 per-dog 參考。

**現象(w_form=10,看 `stage3_squeeze.png`)**:三狗都到不了終點(跑滿 450 週期)。
- **leader 走中線**(y=0,參考清縫)**順利穿門**到 (2.34, 0),差 0.66 到終點。
- **兩翼狗直線參考指向寬-V 槽 y=±0.8,那正落在門柱後方** → head-on 撞柱,卡在 (−0.66, ±1.03)
  動不了(被單次線性化的 reactive CBF 頂進「門柱 ↔ 上牆」夾縫,沒有繞下穿縫的全域路徑)。

**對照實驗證明非編隊之過**:同場景跑 `w_form=0`(編隊完全關掉)**也死鎖**,door-peak f 0.95 還更糟。
→ **死因不是 formation vs CBF 僵局;降 w_form 的 dynamic tightening 無效**(病在 formation 上游的參考幾何)。

**安全全程守住**:inter-agent h≥0(min **+0.76**,leader 跑遠、翼狗卡住,離很遠)、node h≥0
(min **+0.059**,翼狗騎柱邊界不撞)。→ **safety 沒失效,失效的是 liveness**。

**與階段 1 var3 同構**:直線參考正對障礙 → 死鎖(var3);var3b 給彎參考 → 解開。這裡是同一個病:
翼狗的直線參考天生指向寬-V 槽(門柱後),不指向縫口。**正解在 planner 層**——給翼狗導向縫口的
彎/A\* 參考(階段 3/4 引入),**非 reactive breaker、非調 w_form**。依 `spec_updates_v2` C 節
reactive-patch 移除清單,**這裡不加任何 breaker**(那是階段 5 要拆的債)。故 squeeze 設為
`gating=False` 診斷 probe:每次仍跑 + 出圖 + 報告,但不 gate 階段 3。

## 測試 / 回歸

- **26 單元全綠**(constants 8、node 6、rti 2、edge 7、admm 4+1;新增 `test_three_dog_wiring_generic`)。
- **`verify_stage2` 無回歸**:247 週期五 gate 全綠(formation/CBF 預設 off 時位元不變)。
- **`verify_stage3` V/column 核心 gate 綠**(exit 0);squeeze probe 死鎖如實上報。

## 未做 / 已知

- **slack 用量未上報**:coordinator 把 `xi` 截到 `[:nz]` 丟掉 slack;要報需改 `node.solve` 回傳,暫略。
- squeeze 的柔性變形正向 demo(f 升起再回落)需 planner 層彎參考,留待階段 4。

## ⚠️ 環境限制（同階段 1/2）

`verify_stage3.py` **只能在 Docker 容器 `5cdd1d8f092e` 內跑**:`core.formation` import rospy(需 ROS
sourced),host 亦缺相容 osqp。容器:py3.8 / numpy 1.17.4 / **osqp 0.6.3** / **無 pytest** → 單元測試
用 inline runner 呼叫 `test_*` 函式。bind mount:host `/home/tony/LeggedControl_ws` == 容器
`/root/LeggedControl_ws`,PNG 直接落 host。跑法:

```bash
source /opt/ros/noetic/setup.bash
python3 legged_upper_control/admm/verify_stage3.py
```
