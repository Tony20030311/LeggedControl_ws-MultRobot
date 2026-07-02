# 階段 2 — 兩狗 ADMM node-edge splitting（完成）

> 施工圖見 `docs/opensource_reference.md` 階段 2 + 文件一 Part 4 + 文件二 C3/C4.5/C5 +
> `spec_updates_v2.md` B1/B8。這是全專案**唯一原創、無範本**的一階段。建在階段 0/1 上:
> 世界座標 double integrator、Xiong 離散二階 HOCBF、OSQP 直接 binding。
> **通過標準**:`verify_stage2.py` 閉迴路 gate 全綠 + 三殘差圖目視。狀態:**綠**。

## 範圍（刻意最小,隔離 splitting）

2 狗、1 條 edge `{1,2}`,complete graph 退化。**無靜態障礙、無編隊**(延到階段 3)。
唯一耦合 = inter-agent CBF 走 edge。這樣 `r_prim/r_dual` 曲線純反映 node-edge 對不對。

## 做了什麼

| 檔案 | 角色 |
|---|---|
| `admm/constants.py`（改） | 加 `RHO=20`、`SLACK_LAMBDA=5`、`D_MIN=0.6`、`P_ITERS=20`(+test 守門) |
| `admm/node_subproblem.py`（改） | 加**選用** consensus(`rho_consensus`/`n_neighbors`/`solve(consensus_target=)`),**預設 off → 階段 1 位元不變**(回歸五場景驗過) |
| `admm/rti_linearizer.py` | 凍結 edge CBF 係數(`g_k,H̄_k,ā`)+ `shift_xi` + `realized_hbar`。**契約:只吃真身 ξ** |
| `admm/edge_subproblem.py` | edge QP(2×6N+slack),OSQP 直接 binding、固定稀疏、每輪只 update q、warm-start。**k=0 硬 / k≥1 soft** |
| `admm/admm_coordinator.py` | 主迴圈 node→edge→dual,z 按 edge 存,殘差量測 |
| `admm/verify_stage2.py` | 兩狗閉迴路 gate + 三殘差圖 |
| `admm/test_{rti,edge_qp,admm}.py` | 單元(手算 CBF row、consensus P/q、dual、B1 求和、k=0 硬化) |

## node-edge 公式（文件一 4.2–4.6）

- **Node (19)**:`min J^i + (ρ/2)Σ_{j∈N(i)}‖ξ^i − z^{ij,i} + λ^{ij,i}‖²`。P 對角 `+ρ|N(i)|`、q `−ρ·Σ(z−λ)`。
- **Edge (20)**:`min φ(s) + (ρ/2)‖z^i−ξ^i−λ^i‖² + (ρ/2)‖z^j−ξ^j−λ^j‖²` s.t. inter-agent HOCBF(soft k≥1 / **hard k=0**)。
- **Dual (21)**:`λ += ξ − z`(scaled,**無 ρ 係數**)。
- **z 按 edge 存**(B1):`z[(1,2)][1/2]`,node 的 `Σ_{j∈N(i)}` 對此結構求和。
- **cold-start**(4.6):cycle 0 用無耦合 node QP 填 `z0`、`λ0=0`;之後 warm-start。無耦合 MPC = 階段 1 node QP。

### 操作點契約（node-edge 最隱晦處,寫死在 `rti_linearizer.py`）

edge 線性化的 `ē=p̄¹−p̄²`、`ā`、`g`、`H̄` **一律取真身 ξ̄,不取分身 z**。
- cold-start:`ξ̄ = 無耦合解 = z0`(三者由建構重合,無錯位)。
- 穩態:`ξ̄ = 上週期真身平移一格`。
- **決定性理由(三狗才露餡)**:碰撞幾何 `ē=p¹−p²` 是全域唯一真相;分身 z 是 edge-local(狗 1 在兩條邊有 `z^{12,1}`、`z^{13,1}` 兩份),無法組全域幾何。只有唯一真身 ξ 能。node 與 edge 必須共用這一個操作點,否則凍結集不自洽、r_prim 不乾淨下降。

### edge CBF row（完整線性化,對稱兩端）

`g_k = COEF_HK2·0.03·ē_{k+2} + COEF_HK1·0.01·ē_{k+1}`(**h_{k+2} 與 h_{k+1} 兩項**,偏離文件二 C3「只 h_{k+2}」—— 承 stage-1 定案)。`col_a^i=−g`、`col_a^j=+g`(一推一拉)。硬化 k=0 只 clamp 該 row 的 slack=0,**row 係數不碰**,故對稱兩端 + h_{k+1} 自動一起硬化。

## Checkpoint 3 的關鍵發現:全 soft edge 做不出避讓 → 改硬 k=0

**第一版(edge 全 soft,paper Eq.13 / spec C4.5)失敗**:ADMM **收斂了**(r_prim/r_dual 乾淨下降,splitting 機制正確),**但兩狗直直對穿、min realized h=−0.32、只隔 0.20m**,converge 到「不避讓」的解。

**四探針釘死根因**(不是猜、不是 bug):
| 探針 | 結果 | 排除 |
|---|---|---|
| λ_slk 5→1e4 | min h 不動 −0.32 | 不是 slack 成本 |
| SQP 每輪重線性化 | 一模一樣 −0.32 | 不是操作點 |
| λ_slk→1e6 | 才 SAFE 但 r_prim=0.21 不收斂 | 硬壓天文數字且毀收斂 |
| **硬 edge k=0** | **min h=+0.05、sep 0.64、r_prim=1.4e-15** | ← 解 |

**真因**:edge CBF 梯度 `g=3Ts²·ē≈0.003`(交會時 ē 小)太弱。全 soft 下 slack(`λ_slk·s²`)吸違反比推 z 加速度(成本 ρ)便宜,ADMM 收斂到不避讓。硬壓 λ_slk 到 1e6 才避讓、但 z 要求 |a| 遠超 ξ 的 ±1 bound → consensus 崩、不收斂。

**修法(採用):硬 edge k=0(k≥1 soft)**,與 **stage-1 node CBF 完全同構**(node 已是 k=0 硬/k≥1 soft)。實測避讓 + 收斂兩全。z 無 bound → 硬 k=0 edge QP **永不 infeasible**(z 加速度可任意大滿足,ξ 由 consensus 在 bound 內跟到,r_prim→0 證明跟得上)。這是 spec C4.5「all-soft」的**刻意偏離**,與 stage-1「完整線性化偏離細節1」同性質、Tony 拍板。

## ⚠️ h2_viol 低估 ~11×（寫進 verify 註解 + 此處）

`h2_viol = max_k(−H̄_k)^+` 是**三點診斷值**,三點係數和 = 1−1.4+0.49 = **0.09**,故 h 近乎定值時 `H̄≈0.09·h` → **h2_viol 低估 realized 破壞約 11×**(realized h=−0.32 時 h2_viol 只有 ~0.03)。
**安全真相一律用 realized `‖p¹−p²‖²−d²`,gate 用它;h2_viol 只當收斂診斷、且記得這個換算。**

## Gate 結果（`verify_stage2.py`,綠）

兩狗微偏移對向(dog1 (−3,+0.1)→(3,+0.1)、dog2 對稱),閉迴路每週期跑完整 ADMM 取 a₀ 推進雙 plant。

- ✅ 兩狗都到終點(247 週期)、✅ **realized inter-agent h≥0 全程**:`min h=+0.0505`(sep 0.64 > d_min 0.6,真的避開)
- ✅ **r_prim 單調下降**:`0.30 → 2.7e-5`(node-edge 鐵證)
- ✅ **r_dual 單調下降**:`6.83 → 3.7e-2`(文件一 6.2:primal+dual 都要降)
- 診斷 h2_viol ~0.034(低估,見上)
- 出圖:`docs/progress/stage2_residuals.png`(雙軌跡避讓 / realized h(t) / r_prim+r_dual 收斂 / h2_viol)

## 環境備忘

同階段 1:verify 只能在容器 `5cdd1d8f092e` 內跑(host matplotlib 在 numpy 2 下壞)。單元用 inline runner。

## 跑法（容器內）

```bash
CID=5cdd1d8f092e; PKG=/root/LeggedControl_ws/src/legged_control/legged_upper_control_pkg
docker start $CID
# 全單元（constants/node/rti/edge/admm,25 條）
docker exec $CID bash -lc "cd $PKG/legged_upper_control/admm && python3 -c '
import importlib
for mod in (\"test_constants\",\"test_node_qp\",\"test_rti\",\"test_edge_qp\",\"test_admm\"):
    m=importlib.import_module(mod); [getattr(m,n)() for n in dir(m) if n.startswith(\"test_\")]
print(\"unit OK\")'"
# gate + 出圖
docker exec $CID bash -lc "cd $PKG && python3 legged_upper_control/admm/verify_stage2.py"
```

## 下一步（尚未開始）

階段 3:三狗 + 編隊(formation gradient 塌成 node 線性項,B3/B4)。node CBF 障礙/牆重新啟用(node 已有機制)。edge 擴到 3 條、node 到 3 個。**開工前先回報計畫**。
