# 在新電腦上還原並接續 ADMM-CBF-DMPC 工作

搬機用的完整流程。假設你已把隨身碟上的 **6 個檔**拷到新機器的家目錄 `~/`：

| 檔 | 大小 | 內容 | 解到哪 |
|----|------|------|--------|
| `leggedcontrol_repo.bundle` | 123M | 全部程式碼 + git 歷史 + 最新 commit | clone 成 workspace |
| `legged_sil.tar.gz` | 3.9G | Docker 驗證環境(ROS+Gazebo+OCS2+手動 pip 套件) | `docker load` |
| `claude_mem.tar.gz` | 24K | Claude 記憶(7 個進度筆記，接續的核心) | `~/.claude/projects/...` |
| `graphify_out.tar.gz` | 278K | 知識圖(ARCH_PRIMER + graph.json) | workspace 根 |
| `claude_global.tar.gz` | 27K | 全域 `~/.claude`(graphify skill + plugins + 偏好) | `~/.claude/` |
| `claude_project.tar.gz` | 2.4K | 專案 `.claude` hooks（SessionStart 自動載 primer 等） | workspace 根 |

> ⚠️ **全程用 `/home/tony/LeggedControl_ws` 這個路徑**。Claude 記憶資料夾名、以及一個 PostToolUse hook 裡寫死的路徑都要對得上。若新機器 user 不是 `tony`，見文末「不同 user 路徑」。

---

## 1. 還原程式碼（bundle → clone）

```bash
git clone ~/leggedcontrol_repo.bundle /home/tony/LeggedControl_ws
cd /home/tony/LeggedControl_ws
```

**要幹麻**：bundle 是整個 repo 壓成一個檔，clone 它就攤開全部程式碼 + 歷史。此時 `origin` 指向 bundle 檔。
想接回 GitHub 當雲端備份（可選）：

```bash
git remote set-url origin git@github.com:Tony20030311/LeggedControl_ws-MultRobot.git
```

## 2. 補回 bundle 帶不走的（被 gitignore 的兩樣）

```bash
tar xzf ~/claude_project.tar.gz -C /home/tony/LeggedControl_ws   # .claude hooks
tar xzf ~/graphify_out.tar.gz  -C /home/tony/LeggedControl_ws   # graphify-out/ 知識圖
```

**要幹麻**：專案 `.claude/`(hooks)和 `graphify-out/` 都被 gitignore，不在 bundle 裡，要用 tar 補進 workspace。

## 3. 還原 Docker 驗證環境

```bash
docker load < ~/legged_sil.tar.gz
docker run -it --name LeggedControl_SIL \
  -v /home/tony/LeggedControl_ws:/root/LeggedControl_ws \
  legged_sil:portable bash
```

**要幹麻**：`docker load` 把筆電那個 ROS+Gazebo+OCS2 環境（含手動裝的 osqp/numpy/matplotlib）原封還原。
`-v` 把 host 的 workspace 掛進容器 `/root/LeggedControl_ws`。
要開 **Gazebo 畫面**再加：`-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`（那台要有顯示/GPU）；純 headless 驗數據不用。

## 4. 容器內重編（build/devel 沒搬，一定要重生）

```bash
# ↓ 在容器內
cd /root/LeggedControl_ws && source /opt/ros/noetic/setup.bash
catkin build          # 首次含 OCS2，約 10 分鐘
source devel/setup.bash
```

**要幹麻**：編譯產物是機器相關的、不能搬，要重編一次。之後跑系統照 repo 根 `CLAUDE.md` 的 Running the System。

## 5. 還原 Claude 記憶 + 全域設定（在 host，不是容器）

```bash
mkdir -p ~/.claude/projects/-home-tony-LeggedControl-ws
tar xzf ~/claude_mem.tar.gz    -C ~/.claude/projects/-home-tony-LeggedControl-ws
tar xzf ~/claude_global.tar.gz -C ~/.claude
```

**要幹麻**：`claude_mem` 是所有進度筆記（接續核心）；`claude_global` 是 graphify skill + plugins + 偏好。
⚠️ 若新機器**已有** `~/.claude/settings.json`，先備份再手動合併，別直接覆蓋。

## 6. （可選）裝 graphify 查圖工具

```bash
pip3 install --user graphifyy==0.8.45      # 套件名是兩個 y
```

**要幹麻**：讓 `graphify query` 能跑。不裝也行——`ARCH_PRIMER.md` + 記憶已足夠接續，這只是加速器。

## 7. 開 Claude Code 驗證接續

```bash
cd /home/tony/LeggedControl_ws && claude
```

問它「**我們進度到哪、接下來做什麼**」。答得出 **stage 0-4 + edge handoff + 待辦（梅花樁 / 障礙 arena / 慢速）** = 完全接上。
沒自動記得 → 叫它讀 `graphify-out/ARCH_PRIMER.md` 和 `~/.claude/projects/-home-tony-LeggedControl-ws/memory/MEMORY.md`。

---

## 附錄

**目前進度（搬機時）**：ADMM-CBF-DMPC stage 0→4 全綠、全 commit；edge-CBF handoff 安全（動態安全前綴截斷）已 commit + 7 次 Gazebo 驗過。
**待辦（依重要性）**：① 梅花樁密集壓測（會壓爆 ~2-active-edge 上限）② 障礙 arena（node/wall CBF 上 Gazebo）③ 三狗慢速查因 ④ `use_admm_target:=true` 乾淨啟動。

**跑系統的指令**（詳見根 `CLAUDE.md`）：
```bash
roslaunch legged_unitree_description obstacle_world.launch   # 或 five_dogs.launch
roslaunch legged_controllers fleet_bringup.launch
rosrun legged_controllers start_fleet.sh
rosrun legged_upper_control ocs2_fleet_publisher.py          # ADMM → OCS2
```

**不同 user 路徑**：若新機器不是 `/home/tony/...`，把記憶解到 `~/.claude/projects/-home-<user>-LeggedControl-ws/`（路徑的 `/` 換 `-`），且 `.claude/settings.json` 裡 PostToolUse hook 那條寫死的 `/home/tony/...` gate 會靜默跳過（不影響跑，只少個自動 test）——建議直接用同路徑最省事。
