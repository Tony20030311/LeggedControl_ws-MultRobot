# 階段 1 — 單狗 node QP（完成）

> 施工圖見 `docs/opensource_reference.md` 階段 1。建在階段 0 守門層上:世界座標
> double integrator、trajectory-level multiple-shooting、Xiong 離散二階 HOCBF
> (node-local 障礙/牆)、OSQP 直接 binding。**不碰 ADMM/edge/共識/編隊**。
> **通過標準**:`verify_stage1.py` 閉迴路 gate 全綠 + 出圖目視。狀態:**綠**
> (5 場景:baseline+var1+var2+var3b 走完整 gate,var3 走安全 gate —— 見下「過擬合檢查」)。

## 做了什麼

在 `admm/` 新增 4 檔,改 2 檔:

| 檔案 | 角色 |
|---|---|
| `admm/node_subproblem.py` | `NodeSubproblem` — 單狗 node QP(OSQP 直接 binding)+ **cold-start 軟暖機**(見發現 1) |
| `admm/reference.py` | 直線折線 → arc-length 錨點(x_des,位置式);階段 3/4 換 A* 沿用;支援彎折線(var3b) |
| `admm/verify_stage1.py` | 閉迴路 gate + PNG,**5 場景批次**(baseline + 3 變體 + var3b) |
| `admm/test_node_qp.py` | matrix 組裝單元檢查(維度、動態往返、手算 CBF row) |
| `admm/constants.py`（改） | 加 bounds `MAX_VX/VY/AX/AY`(C4.4)+ h_{k+1} 梯度係數(見下) |
| `admm/test_constants.py`（改） | 加 h_{k+1} 係數守門(`A_TO_P1_COEF`,`CBF_CONSTR_COEF_P1`) |

## Node QP(`node_subproblem.py`)

決策向量 `z = [ξ (6N); s (n_slack)]`,`ξ=[x_1..x_N, a_0..a_{N-1}]`,index 一律 `4*N`
公式取自 constants。OSQP `½zᵀPz+qᵀz`,`l≤Az≤u`,**稀疏 pattern 一次組,每週期只
`update(q,l,u,Ax)` + warm_start**(spec E)。

- **P**(對角,固定):`2·blkdiag(Q×(N-1), P_term, R×N, w_pred·I)`。`Q=diag{q_px,q_py,0,0}`
  (**B7 只追位置**);`P_term=diag{10q,10q,q_v,q_v}`;`R=diag{r,r}`。**無共識項 ρ**(階段 2)。
- **q**(每週期):`-2Q·x_des`(位置)、terminal `-2P_term·x_des`。**無 formation**(階段 3)。
- **動態等式** `A_eq ξ=b_eq`:精確 `Ad,Bd`,`k=0` 用實測 `x_now`(`b_eq[0:4]=Ad·x_now`)。
- **bounds**:`|v|≤(0.55,0.35)`、`|a|≤(1.0,1.0)`(C4.4)。
- **CBF row**:離散三點,`k=0..N-2`;`k=0` 硬、`k=1..N-2` soft(slack `s≥0`,罰 `w_pred`)。

## 關鍵決定:CBF 用「完整線性化」(與你確認)

spec 細節 1 原寫「a_k 只進 h_{k+2}」(3Ts²)。實測發現:用精確 Bd 時 a_k 其實也經 ½Ts²
進 p_{k+1}→h_{k+1};**凍結 h_{k+1} 會讓繞障礙時穩定滲入 ~4cm(realized h=-0.039)**。
與你討論後改用**完整線性化**(a_k 同時進 h_{k+2} 與 h_{k+1}):

```
三點值   H̄ = h̄_{k+2} + COEF_HK1·h̄_{k+1} + COEF_HK·h̄_k        (=…−1.4·+0.49·)
障礙(∇2e): g = COEF_HK2·CBF_CONSTR_COEF·ē_{k+2} + COEF_HK1·CBF_CONSTR_COEF_P1·ē_{k+1}
牆(仿射,exact): g = (COEF_HK2·A_TO_P2_COEF + COEF_HK1·A_TO_P1_COEF)·n_w
A-row on a_k = −g ,  u = b_k = H̄ − g·ā_k                     (增量式,ā 來自操作點)
```

- 新增守門係數(`test_constants.py` 已 pin):`A_TO_P1_COEF=½Ts²=0.005`、
  `CBF_CONSTR_COEF_P1=Ts²=0.01`。原 `A_TO_P2_COEF=0.015`、`CBF_CONSTR_COEF=0.03` 不動。
- **牆因仿射→這樣是 exact**;障礙只丟 O(Ts⁴) 二次項。
- ⚠️ **階段 2 edge row 必須對稱跟進**:`a^i` 與 `a^j` 兩端都要加 h_{k+1} 項。

## Gate 結果(`verify_stage1.py`,綠)

閉迴路:真 plant `X_{t+1}=Ad·X+Bd·a_0`,直線參考故意瞄準障礙,一障礙 + 一牆。

- ✅ 到終點(172 週期)、✅ 每週期 QP solved
- ✅ **realized 閉迴路 h≥0 不破**:`min h_obs=0.0202`、`min h_wall=0.3991`
- 診斷(非 gate):`min 預測 h = −0.264`。**單次線性化 + soft 遠期步**下,receding-horizon
  的軟預測遠期會滲(w_pred 加到 20000 也壓不下,因遠期線性化操作點穿過障礙)。realized 才是
  安全依據(與你確認:預測值當診斷、量化上報,spec B9)。要讓預測也 h≥0 需 SQP 重線性化(未採)。
- 出圖:`docs/progress/stage1_traj.png`(參考/實際/障礙+margin/牆/realized h(t)+預測 h 診斷)。

軌跡行為正確:狗沿直線 → CBF 把它壓到 y≈−0.32 貼著 margin 圓外緣繞過 → h 貼 0 不破 → 回線到終點。

## 過擬合檢查(5 場景)+ 冷啟動修復 + deadlock 佐證

baseline 只有一種幾何,怕 node QP 過擬合。加 3 個變體 + 1 個佐證場景,`verify_stage1.py`
一次批次跑、逐場景記 pass/fail、不 abort;某場景破就如實印 min_h,絕不回頭調幾何(過擬合檢查的底線)。

| 場景 | gate | realized min h_obs | 結果 | 測什麼 |
|---|---|---|---|---|
| baseline | full | **+0.0202** | ✅ 到終點 | 對照組 |
| var1 偏心 | full | **+0.0088** | ✅ 到終點 | 障礙 (2.0,−0.55),狗往**上**繞(baseline 反向)—— 避障方向沒寫死 |
| var2 雙障礙 | full | **+0.0118** | ✅ 到終點 | (1.6,+0.45)+(2.6,−0.45) slalom;兩顆 h 都 active(controller 隨 `n_feat` 縮放,非改測試糊弄) |
| var3 貼近起點 | **safety** | **+0.0622** | ✅ feasible+h≥0 | 起點 h₀=+0.057 的緊初始條件 |
| var3b 彎參考 | full | **+0.0396** | ✅ 到終點 | 同 var3 障礙,but 彎折線參考 |

baseline/var1/var2 三個數字**逐位不變**(改冷啟動前後相同)——它們 cycle 0 障礙離起點 >1.4m、
horizon 內 CBF 全不 active,軟暖機解=硬解,`a0` 與線性化點無關 → 位元相同。

### 發現 1:冷啟動 infeasibility(var3)→ 用 plan A(軟暖機)修好

var3 原本 **cycle 0 `primal infeasible`**。根因(手算 + 程式逐位吻合):cycle 0 無 warm-start,拿
**直線 reference 當線性化操作點**;錨點間距僅 `v_cruise·Ts=0.03m`,障礙又只在 0.7m 外 → 前方第 2
個錨點 `pbar[2]=(0.06,0)` 已踩進 margin(h₂=−0.0029)→ 硬 k=0 三點值 `Hbar=−0.0159`,要求
`g·a₀≥+0.0159`,但 g 極小(‖g‖≈0.01,3cm 間距無操舵權)→ `|a|≤1` 內最多 +0.0114 < 需求 → 不可行。

**修法(plan A,`node_subproblem.py`)**:cold-start(`xbar is None`)先跑一次**軟 pass**——把 k=0
硬 row 的 `u` 設 +inf(丟掉),k≥1 soft row 仍把軌跡彎開 → 得到已繞障礙的可行解 → 拿它當操作點再跑
**硬 pass**(真正輸出)。**不動主問題稀疏結構**(只改 `u` 值,仍 update-only、warm-start)。穩態週期
(有 warm-start)不跑軟 pass、走原路徑 → baseline 不受影響。**k=0 永不軟化輸出**:軟暖機後硬 pass 仍
不可行就如實回報,不走「回退軟化 k=0」。這套軟暖機也是**階段 2 edge cold-start 的種子**(文件一 4.6
無耦合 MPC 填 z0)——階段 1 就把階段 2 的冷啟動機制建好。

修完:var3 每週期 feasible、**realized h_obs=+0.0622 全程 h≥0**(比 baseline 還安全)。

### 發現 2:head-on deadlock(var3 不到終點)→ 是直線參考建構物,非 controller 限制

var3 修好可行性後**仍卡在障礙前不到終點**(頂在 margin 右緣,h 平在 +0.062、40s 不動)。這是典型
**CBF head-on deadlock**:障礙近乎正對(偏 0.1),凍結線性化每週期在同一 stall 點重凍,terminal
直拉被 CBF 擋、橫向推力建不起來。

**關鍵:拆不開**。要觸發冷啟動 infeasibility,障礙 margin 必須倒伸回起點前 ~2 步且 h₀≥0 →
幾何上**逼近正對** → 必 deadlock。所以「能測冷啟動修復的幾何」=「會 deadlock 的幾何」。

**不在階段 1 加 deadlock-breaker**(擾動/偏置/SCP 重線性化正是 `spec_updates_v2` C 節明令別移植進
ADMM 的 reactive 補丁,現加就是階段 5 要拆的債)。改把 var3 判**安全 gate**(feasible+h≥0=冷啟動修復
的真驗收),到終點不列入。

### 發現 3(佐證):彎參考 → deadlock 消失(var3b)

var3b 同一顆正對障礙,但參考給**手動繞過的彎折線**(模擬 A*+inflation 在階段 3 會給的東西):
`(0,0)→(0.25,0.55)→(0.7,0.8)→(1.15,0.6)→(1.5,0.25)→(2.0,0)→(4,0)`。結果 **deadlock 消失、176
週期到終點、realized h_obs=+0.0396≥0**,且 `pred 診斷=+0.0381` 轉正(彎參考讓操作點從不穿障礙)。

→ **證明 var3 deadlock 純是 stage-1 直線參考建構物**,production 由 A* 供曲線參考時不觸發。把 deadlock
記為已知限制才站得住,階段 3 接 A* 少一個未爆彈。

## 環境備忘(重要)

- **verify 在容器內跑**(host 系統 matplotlib 在 numpy 2 下 import 就炸)。
  容器 `5cdd1d8f092e`:py3.8 / numpy 1.17.4 / matplotlib 3.1.2(正常)/ **osqp 0.6.3** / 無 pytest。
- bind mount:host `/home/tony/LeggedControl_ws` == 容器 `/root/LeggedControl_ws` → PNG 直接落 host。
- 容器無 pytest → 單元測試用 inline runner 呼叫 `test_*` 函式(見下)。

## 跑法(容器內)

```bash
CID=5cdd1d8f092e; PKG=/root/LeggedControl_ws/src/legged_control/legged_upper_control_pkg
docker start $CID
# 單元(無 pytest,直接呼叫 test_* 函式)
docker exec $CID bash -lc "cd $PKG/legged_upper_control/admm && python3 -c '
import importlib
for mod in (\"test_constants\",\"test_node_qp\"):
    m=importlib.import_module(mod)
    [getattr(m,n)() for n in dir(m) if n.startswith(\"test_\")]
print(\"unit OK\")'"
# gate + 出圖
docker exec $CID bash -lc "cd $PKG && python3 legged_upper_control/admm/verify_stage1.py"
```

## 下一步(尚未開始)

階段 2:兩狗 ADMM(`edge_subproblem.py` + `admm_coordinator.py` + `rti_linearizer.py`)——
唯一原創、無範本、最易錯。**閘門未過不進**;階段 1 已過,依指示**開工前先回報實作計畫**
(要建哪些 function、OSQP 矩陣怎麼組、verify 腳本怎麼寫),確認後再動手。edge CBF row 記得
對稱跟進本階段的完整線性化(h_{k+1} 項)。
