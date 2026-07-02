# 交接文件 — ADMM-CBF-DMPC 實作進度

> **用途**:貼給新的 claude.ai 對話當開場,讓它立刻進入狀態,跟 Claude Code 那邊的實際進度接軌。
> 我(Tony)在 claude.ai 這邊做設計/brainstorm,在 Claude Code(終端)那邊做實作。兩邊靠這份文件同步。

---

## 專案一句話

在 `qiayuanl/legged_control` (ROS Noetic + OCS2) 上,為 3 隻 Unitree 四足機器人實作
**ADMM-CBF-DMPC**(Zeng et al. 2026, arXiv:2603.19170)的分散式軌跡優化。核心是 node-edge
splitting——把 inter-agent 避撞耦合搬到 edge 子問題,node 各自並行解局部 QP。

## 我的環境

- **Claude Code**:v2.1.186,Opus 4.8 + xhigh effort,Claude Max
- **啟動位置**:`~/LeggedControl_ws/src/legged_control/legged_upper_control_pkg`(重要:在 package 層開,相對路徑才對)
- **工具鏈**:Superpowers、Context7 (MCP)、Graphify、Codex、planning-with-files、code-review
- **repo**:`Tony20030311/LeggedControl_ws-MultRobot`,package `legged_upper_control_pkg`

## 文件層(都在 package 內)

```
legged_upper_control_pkg/
├── CLAUDE.md                         # package 層,ADMM 新架構的實作憲法(主力)
├── docs/
│   ├── 文件一_論文完整數學推導.md      # 論文 (1)-(21) 完整數學,含補洞推導
│   ├── 文件二_架構藍圖.md             # C1-C6 架構設計(舊 2Ts² 係數已標作廢)
│   ├── opensource_reference.md       # 開源導航 + 階段 0-5 施工圖(參考,非規格)
│   ├── spec_updates_v2.md            # 規格補丁,最新,binding
│   └── paper.pdf                     # 論文原檔(備查)
├── references/                       # clone 的開源(4 個 repo)
│   ├── consensus_ADMM/               # ADMM 骨架(仿寫但 consensus 要換 node-edge)
│   ├── MPC-CBF/                      # 單機一階 CBF baseline(對拍)
│   ├── NMPC-DCLF-DCBF/              # 離散二階 HOCBF(對拍,NMPCDCBF2.m)
│   └── legged_planner/              # OCS2 adapter 範本
├── legged_upper_control/admm/        # ← 實作 code 放這
│   ├── constants.py                  # 全套係數唯一來源(階段 0 產出)
│   ├── test_constants.py             # 守門測試(7 條全綠)
│   └── __init__.py
└── .claude/settings.json             # hooks(graphify + 新加的 PostToolUse)
```

**文件優先序(衝突時上面贏)**:spec_updates_v2 > 文件二 > 文件一 > opensource_reference。

## 施工圖(階段 0-5,gate 未過不進下一階段)

- **階段 0**:數學守門層(constants + test)—— ✅ **已完成,7 條測試全綠**
- **階段 1**:單狗 node QP(node_subproblem.py,OSQP 直接 binding)—— ✅ **已完成,gate 綠**(見 `docs/progress/stage1.md`)
- **階段 2**:兩狗 ADMM(edge_subproblem + admm_coordinator + rti_linearizer)—— ⏭ **下一步**;唯一原創、無範本、最易錯
- **階段 3**:三狗 + 編隊(formation gradient)
- **階段 4**:接 OCS2(motion_adapter,padding + yaw 旁路)
- **階段 5**:調參 + 拆 reactive 補丁

## 已完成:階段 0(數學守門層)

**狀態**:7 條守門測試全綠。地基立好了。

**焊死的常數**(constants.py,全套係數唯一來源,一律從 N/Ts/γ 公式推出、不寫死):
- 參數:N=20, Ts=0.10, γ1=γ2=0.30, n_x=4 [px,py,vx,vy], n_u=2 [ax,ay], ξ_dim=6N=120
- 精確離散化:Ad=[[I2,Ts·I2],[0,I2]], Bd=[[½Ts²·I2],[Ts·I2]](廢除前向歐拉簡化)
- HOCBF:h_{k+2}-(2-γ1-γ2)h_{k+1}+(1-γ1-γ2+γ1γ2)h_k≥0,**γ1γ2 正號**
- 精確係數:a_k→p_{k+2} = 1.5Ts² = 0.015;CBF 約束係數 = 3Ts² = 0.03
- index 一律 4*N 公式,不 hardcode

**7 條測試**:hocbf 係數(0.49/-1.4)、sign-trap(擋負號 0.31)、bd 係數(0.015/0.03)、
bd 矩陣逐元素、Ad@Bd 自洽(==0.015)、index N=20、index 隨 N 縮放(防寫死 80)。

## 已完成:階段 1(單狗 node QP)

**狀態**:`verify_stage1.py` 閉迴路 gate 綠。詳見 `docs/progress/stage1.md`。

- node QP:6N 多重射擊 + soft-CBF slack,OSQP 0.6.3 直接 binding(固定稀疏、update 值、warm-start)。
- **CBF 完整線性化**(與 Tony 確認,偏離 spec 細節1「a_k 只進 h_{k+2}」):a_k 同時進 h_{k+2}(3Ts²)
  與 h_{k+1}(½Ts²)。凍結 h_{k+1} 會滲 ~4cm;完整版 realized h≥0。新守門係數 `A_TO_P1_COEF=0.005`、
  `CBF_CONSTR_COEF_P1=0.01`。**階段 2 edge row 要對稱跟進(a^i,a^j 兩端都加 h_{k+1} 項)。**
- gate 定義(與 Tony 確認):**realized 閉迴路 h≥0** 為準(0.0202/0.399),預測遠期(soft)滲 -0.264
  只當診斷上報(單次線性化 + soft,receding-horizon 正常;要預測也 h≥0 需 SQP,未採)。
- **環境**:verify 只能在容器 `5cdd1d8f092e` 內跑(host matplotlib 在 numpy 2 下壞);容器無 pytest,
  單元測試用 inline runner。PNG 經 bind mount 落 host。

## 進階段 2 的做法(開工前先回報計畫)

- **開工前先回報實作計畫**(要建哪些 function、OSQP 矩陣怎麼組、verify 腳本怎麼寫),Tony 確認後再動手。
- 骨架仿 `consensus_ADMM/admm_mpc.py`(三步驟)但 **consensus 換 node-edge**;數學真相 `文件一 Part 4`。
- **z 按 edge 存**(B1);edge CBF row 保留 `a^i,a^j` 雙變數 + **對稱跟進階段 1 的 h_{k+1} 完整線性化**。
- 線性化迴圈外一次(B8);ρ=20;15~20 輪不等收斂。verify:兩狗對向避開 + **r_prim 隨輪數下降** + 出圖。

## 工作方式備忘(這幾天確立的)

- **Superpowers 用在階段 2**(node-edge splitting,有設計、易錯、無範本)。階段 0/1 純規格或有範本,不硬套。
- **驗證是第一優先**:每階段 gate 寫成會 assert 的 verify 腳本,Claude Code 自跑自修到綠;
  行為性判斷(如 r_prim 收斂曲線形狀)產圖給我目視抽查。
- **CBF 係數只認 3Ts²**(spec 細節 1 + test_constants.py 守門),文件二舊的 2Ts² 已標作廢。
- 每階段結束寫一份 `docs/progress/stageN.md` 摘要。stage0.md 已完成。

