# 階段 0 — 數學守門層(完成)

> 施工圖見 `docs/opensource_reference.md` 階段 0。這是 ADMM-CBF-DMPC 的地基:
> 把四個敲定細節裡的常數焊死,之後組矩陣不用再擔心係數對不對。
> **通過標準**:`test_constants.py` 全綠。狀態:**7 條全綠**。

## 做了什麼

新增 `admm/` 子套件(階段 1+ 也放這裡),兩個檔:

| 檔案 | 角色 |
|---|---|
| `legged_upper_control/admm/constants.py` | 全套係數唯一來源(見下) |
| `legged_upper_control/admm/test_constants.py` | 守門測試,pin 住規格文件的 binding 值 |
| `legged_upper_control/admm/__init__.py` | 子套件標記 |

`constants.py` 內容:

- **鎖定參數**:`N=20`, `Ts=0.10`, `γ1=γ2=0.30`, `n_x=4 [px,py,vx,vy]`, `n_u=2 [ax,ay]`, `ξ_dim=6N=120`
- **精確離散化**:`Ad=[[I2, Ts·I2],[0, I2]]`、`Bd=[[½Ts²·I2],[Ts·I2]]`
  (廢除前向歐拉簡化,C3 線性化與 C4.3 動態約束共用同一套 Bd)
- **二階 HOCBF 係數**:`h_{k+2} - (2-γ1-γ2)h_{k+1} + (1-γ1-γ2+γ1γ2)h_k ≥ 0`
- **精確標量**:`a_k→p_{k+2}` = `1.5Ts²`、線性化 CBF 約束係數 = `3Ts²`
- **index 定位公式**:一律 `4*n` 公式(帶 `n` 參數),**不寫死數字**

## constants.py 是全套係數的唯一來源

這是階段 0 的核心原則(套件 CLAUDE.md「Coefficient discipline」):

- **係數錯不會崩**,只會讓行為微妙不對、事後極難 debug → 當場用測試擋掉最省事。
- 所有係數在 `constants.py` **從 N/Ts/γ 公式推出**,不寫死中間值 → 一個 sign flip 會被守門測試抓到。
- 之後所有階段(node QP、edge QP、RTILinearizer…)**一律 import `constants`**,
  不得在別處各自重算係數或硬寫 index 偏移(N 從 10 改 20,寫死的 `80` 會全錯)。
- 文件的 inline 框只是脈絡;**測試才是權威**。改任何係數先跑守門測試。

## 7 條測試各驗什麼

| # | 測試 | 驗什麼 | binding 值 | 來源 |
|---|---|---|---|---|
| 1 | `test_hocbf_coefficients` | HOCBF `h_k` / `h_{k+1}` 係數(γ1γ2 正號) | `0.49` / `-1.4` | spec 細節 2 |
| 2 | `test_hocbf_positive_sign_trap` | γ1γ2 誤植負號會變 `0.31`(差 37%)→ 明確擋掉;`HOCBF_COEFS=[0.49,-1.4,1.0]` | ≠0.31 | spec 細節 2 |
| 3 | `test_bd_coefficients` | `a_k→p_{k+2}` 與線性化 CBF 約束標量 | `0.015` / `0.03` | spec 細節 1 |
| 4 | `test_bd_matrix_exact` | `Bd` 矩陣逐元素;pos/vel 塊標量 | `½Ts²=0.005` / `Ts=0.1` | spec 細節 1 |
| 5 | `test_ad_matrix_and_a_to_p2_consistency` | `Ad` 矩陣;**`Ad@Bd` 位置塊 == 0.015**(矩陣自洽,非只對文字) | `0.015` | 文件二 C4.3 |
| 6 | `test_index_formulas_n20` | N=20 index 表:`x_1@0`,`x_20@76`,`a_0@80`,`a_19@118`;分量偏移 | — | spec 細節 4 |
| 7 | `test_index_formulas_scale_with_n` | N=10 時 `a_0` 挪回 `40`、`ξ_dim=60` → 防寫死 `80` | — | spec 細節 4 |

> 第 1、3 條是規格 `spec_updates_v2.md` 細節 1、2 明列的守門測試;其餘 4 條是白拿的自洽/防呆守門
> (sign-trap、`Ad@Bd` 交叉驗、N 縮放、矩陣逐元素),係數或矩陣哪天飄了會當場紅。

## 跑法

```bash
cd legged_upper_control_pkg
python3 -m pytest legged_upper_control/admm/test_constants.py -q
# 7 passed
```

已設 PostToolUse hook:編輯 `admm/` 下任何 `*.py` 後自動跑此守門測試(見 `.claude/settings.json`)。

## 下一步(尚未開始)

階段 1:單狗 node QP(`node_subproblem.py`),OSQP 直接 binding,import `constants`。
**閘門未過不進**——階段 0 已過,但依指示先停,不進階段 1。
