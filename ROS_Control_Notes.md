# ROS Control 完整筆記

> **專案：NCRL_Dog**
> **範圍：ros_control 框架通用概念 ＋ 在四足機器人 repo 中的實際應用**
> **參考：ros_control 官方架構圖（Dave Coleman, 2013）、legged_control / legged_control_hil repo 原始碼**

---

## 一、ROS Control 是什麼？一句話總結

ROS Control 是一個**控制演算法與硬體實作完全解耦**的框架。
它讓同一套 Controller 不需修改任何一行程式碼，就能無縫切換 Gazebo 模擬器與真實機器人。

---

## 二、核心設計哲學：解耦

```
┌──────────────────────────────────────────────────────────────┐
│  控制演算法（Controller）                                      │
│  → 只管「讀 state、算數學、寫 command」                         │
│  → 完全不知道底下是 Gazebo 還是真實馬達                          │
├──────────────────────────────────────────────────────────────┤
│  Hardware Resource Interface Layer（資源層）                    │
│  → 存放 Handle（指標包裝器），作為 Controller 與 RobotHW 的橋樑    │
├──────────────────────────────────────────────────────────────┤
│  RobotHW（硬體抽象層）                                         │
│  → 負責跟真實硬體或模擬器溝通                                    │
│  → 實作 read() 和 write()                                     │
└──────────────────────────────────────────────────────────────┘
```

**解耦的好處：**
- 今天換了不同的硬體（例如從 Unitree A1 換到 Go1），只需要改寫 RobotHW 的 `read()` 和 `write()`，Controller 完全不用動。
- 今天從模擬器切換到實機，也只需要換一個 RobotHW 實作，Controller 同樣不用動。

---

## 三、架構全貌

以下對照 ros_control 官方架構圖（Dave Coleman, 2013），由下往上逐層說明。

### 3.1 RobotHW（硬體抽象層）

**職責：代表你的機器人（不管實體或模擬）可以給出哪些資源讓 Controller 讀取或寫入。**

RobotHW 做四件事：

1. **在 `init()` 中建立 Handle 並註冊到 Interface**
   - 這是最核心的一步。RobotHW 內部有一組變數（例如 `pos[12]`, `vel[12]`, `eff[12]`, `cmd[12]`），它把這些變數的**指標**包裝成 Handle，然後註冊到對應的 Interface 裡。

2. **在 `read()` 中從硬體/模擬器讀取資料，寫入內部變數**
   - 因為 Handle 包的是指標，Controller 透過 Handle 讀到的就是這些內部變數的最新值，不需要額外的 copy 或 publish。

3. **在 `write()` 中從內部變數讀出 command，送往硬體/模擬器**
   - Controller 在 `update()` 時已經透過 Handle 把 command 寫進了 RobotHW 的內部變數（同一塊記憶體），所以 RobotHW 只要讀取自己的變數就能送出去。

4. **呼叫 `registerInterface()` 向 ros_control 框架註冊自己提供了哪些 Interface**
   - 讓 Controller Manager 知道這個 RobotHW 有哪些可用資源。

**在 repo 中的實際程式碼：**

```cpp
// legged_hw/src/LeggedHW.cpp
bool LeggedHW::init(ros::NodeHandle& root_nh, ros::NodeHandle& /*robot_hw_nh*/) {
  if (!loadUrdf(root_nh)) {
    ROS_ERROR("Error occurred while setting up urdf");
    return false;
  }
  registerInterface(&jointStateInterface_);    // 註冊 Joint State Interface
  registerInterface(&hybridJointInterface_);   // 註冊 Hybrid Joint Interface（自定義）
  registerInterface(&imuSensorInterface_);     // 註冊 IMU Interface
  registerInterface(&contactSensorInterface_); // 註冊 Contact Interface
  return true;
}
```

**繼承關係：**

```
hardware_interface::RobotHW          ← ros_control 官方基底類別
    └── LeggedHW                     ← SIL 版基底（legged_control repo）
         └── UnitreeHW               ← Unitree 實機版（實作 read/write 對接 Unitree SDK）
    └── LeggedHilHW                  ← HIL 版基底（legged_control_hil repo）
         ├── LcmHW                   ← LCM 通訊版（實作 read/write 對接 LCM）
         └── ShmHW                   ← 共享記憶體版（實作 read/write 對接 SHM）
```

> **重點：如果你要支援一台新的機器人，就是繼承 `LeggedHW`（或 `LeggedHilHW`），然後實作 `read()` 和 `write()` 這兩個函式。**

---

### 3.2 Hardware Resource Interface Layer（資源層）

**職責：Controller 和 RobotHW 之間的資料池，存放已註冊的 Handle。**

為什麼需要這一層？

- Controller 不能直接跟模擬器或實機要資料（它不知道底下是誰）。
- 即使拿到了原始資料（例如 Encoder 的 tick 數），Controller 也不一定看得懂。
- 如果讓 Controller 直接跟硬體通訊，一旦硬體換了，Controller 也得跟著改。

所以這一層的功能是：**把轉換好的、Controller 看得懂的資料，用統一的格式（Handle）存放起來，讓 Controller 透過標準 API 存取。**

資源層分成兩個方向的 Interface：

#### Joint State Interface（讀方向）

Controller 讀取硬體狀態用。官方標準的 `JointStateInterface` 讓 Controller 可以讀取每個 joint 的 position、velocity、effort。

**特性：使用 `DontClaimResources`，代表多個 Controller 可以同時讀取同一顆 joint 的狀態，不會衝突。**

#### Joint Command Interface（寫方向）

Controller 寫入控制命令用。

**特性：使用 `ClaimResources`，代表同一時間只能有一個 Controller 寫命令到同一顆 joint。如果兩個 Controller 都想寫同一顆 joint 的 command，Controller Manager 會偵測到資源衝突並拒絕。**

官方標準的 Command Interface 有三種：

| Interface | 寫入的命令 | 適用場景 |
|-----------|----------|---------|
| `PositionJointInterface` | 目標位置 | 位置控制 |
| `VelocityJointInterface` | 目標速度 | 速度控制 |
| `EffortJointInterface` | 目標力矩 | 力矩控制 |

> **重要：我們的 repo 並沒有使用上述三種官方 Command Interface！**
> 原因是四足機器人使用的是 **Impedance Control（阻抗控制）**，一次需要寫入五個參數，而不是只寫一個力矩值。

---

### 3.3 我們 repo 的自定義 Interface

#### SIL 版：HybridJointInterface

定義在 `legged_common/hardware_interface/HybridJointInterface.h`

Handle 中包含的指令欄位（都是 `double*`）：

| 欄位 | 意義 | 由誰寫入 |
|------|------|---------|
| `posDes_` | 目標位置 | Controller（WBC 算出） |
| `velDes_` | 目標速度 | Controller（WBC 算出） |
| `kp_` | 位置增益 | Controller |
| `kd_` | 速度增益 | Controller |
| `ff_` | 前饋力矩 | Controller（WBC 算出） |

#### HIL 版：ImpedanceJointInterface

定義在 `legged_hil_interface/hardware_interface/ImpedanceJointInterface.h`

比 HybridJointHandle 多了時間戳記欄位：

| 額外欄位 | 意義 |
|---------|------|
| `state_sec_`, `state_nsec_` | 狀態時間戳記 |
| `cmd_sec_`, `cmd_nsec_` | 命令時間戳記 |

> 這些時間戳記是 HIL 架構下用來追蹤延遲（latency）和同步的。

#### 對應的底層 Impedance Control 公式

最終送到馬達驅動器（或模擬器）的力矩命令為：

```
τ_cmd = Kp × (P_des − P) + Kd × (V_des − V) + τ_ff
```

- `P_des`, `V_des` 由 NMPC 經過 WBC 算出
- `τ_ff` 由 WBC 算出的前饋力矩
- `Kp`, `Kd` 是阻抗控制增益
- `P`, `V` 是目前的關節位置和速度（由 `read()` 讀入）

**Controller 只負責算出這五個參數，最終的力矩計算（上面的公式）是在底層完成的**（實機上是馬達驅動器，模擬中是 Gazebo plugin）。

#### 為什麼分成 State Interface 和 Command Interface？

兩個原因：

1. **資料流方向相反：** 讀 state 是從硬體往 Controller 的方向；寫 command 是從 Controller 往硬體的方向。

2. **資源衝突管理不同：**

```cpp
// State Interface：不搶資源，多個 Controller 可以同時讀
class ContactSensorInterface
    : public HardwareResourceManager<ContactSensorHandle, DontClaimResources> {};

// Command Interface：搶資源，同時間只有一個 Controller 可以寫
class HybridJointInterface
    : public HardwareResourceManager<HybridJointHandle, ClaimResources> {};
```

---

### 3.4 Handle（指標包裝器）— 整個架構最精妙的設計

**一句話結論：Handle 的本質是指標包裝器（pointer wrapper），RobotHW 和 Controller 透過 Handle 共享同一塊記憶體，這就是 read→update→write 能夠 zero-copy 運作的根本。**

#### 什麼是 Handle？

RobotHW 內部可能有這些變數：

```cpp
// UnitreeHW 裡面的結構
struct UnitreeMotorData {
  double pos_, vel_, tau_;                 // state（由 read 寫入）
  double posDes_, velDes_, kp_, kd_, ff_;  // command（由 Controller 寫入）
};

UnitreeMotorData jointData_[12];  // 12 顆馬達的資料
```

但 Controller 不會直接存取這個 array。RobotHW 會在 `init()` 階段建立 Handle，把這些變數的「指標」包裝進去：

```cpp
// UnitreeHW::setupJoints() 裡面的實際程式碼
hardware_interface::JointStateHandle state_handle(
    joint.first,              // joint 名稱，例如 "LF_HFE"
    &jointData_[index].pos_,  // position 指標
    &jointData_[index].vel_,  // velocity 指標
    &jointData_[index].tau_   // effort 指標
);
jointStateInterface_.registerHandle(state_handle);

hybridJointInterface_.registerHandle(
    HybridJointHandle(
        state_handle,                    // 繼承 state handle 的讀取能力
        &jointData_[index].posDes_,      // position desired 指標
        &jointData_[index].velDes_,      // velocity desired 指標
        &jointData_[index].kp_,          // Kp 指標
        &jointData_[index].kd_,          // Kd 指標
        &jointData_[index].ff_           // feedforward 指標
    )
);
```

#### 運作原理圖解

```
RobotHW 內部                         Handle                          Controller
┌──────────────────┐                                              ┌──────────────┐
│ jointData_[0]    │                                              │              │
│   .pos_ = 1.23 ─────── JointStateHandle ──── getPosition() ───→│ 讀到 1.23    │
│   .vel_ = 0.45 ─────── (指標指向同一塊)  ──── getVelocity() ───→│ 讀到 0.45    │
│   .tau_ = 2.10 ─────── 記憶體位址)      ──── getEffort()   ───→│ 讀到 2.10    │
│                  │                                              │              │
│   .posDes_ ←─────────── HybridJointHandle ── setPosDes(0.8) ───│ 寫入 0.8     │
│   .velDes_ ←─────────── (同樣是指標)     ──── setVelDes(0.1) ───│ 寫入 0.1     │
│   .kp_     ←─────────── 指向同一塊       ──── setKp(100)    ───│ 寫入 100     │
│   .kd_     ←─────────── 記憶體)          ──── setKd(5)      ───│ 寫入 5       │
│   .ff_     ←─────────── )               ──── setFf(3.2)    ───│ 寫入 3.2     │
└──────────────────┘                                              └──────────────┘
```

**關鍵：箭頭兩端指向的是同一塊記憶體！**

- 當 RobotHW 在 `read()` 裡更新 `jointData_[0].pos_ = encoder_value` 時，Controller 呼叫 `handle.getPosition()` 直接就拿到最新值，不需要任何 copy 或 publish。
- 當 Controller 在 `update()` 裡呼叫 `handle.setPositionDesired(0.8)` 時，其實就是直接寫入了 `jointData_[0].posDes_`，RobotHW 在 `write()` 裡讀取自己的變數就能送出去。

#### 兩種 Handle 的差別

| Handle 類型 | 用途 | 包含的指標 |
|------------|------|----------|
| `JointStateHandle` | 讀狀態 | joint name + position + velocity + effort 的指標 |
| `HybridJointHandle` | 讀狀態＋寫命令 | 繼承 JointStateHandle 的全部，再加上 posDes + velDes + kp + kd + ff 的指標 |

> **OOP 提醒：** `HybridJointHandle` 繼承自 `JointStateHandle`。所以只要拿到一個 `HybridJointHandle`，你既可以讀 state（繼承來的），也可以寫 command（自己新增的）。

#### 為什麼不讓 Controller 直接存取 RobotHW 的內部變數？

三個原因：

1. **解耦** — 不管底下是真機還是 Gazebo，Controller 的用法都完全一樣。Controller 只認得 Handle API（`getPosition()`、`setCommand()`），不需要知道底下的變數長什麼樣子。

2. **資源管理** — Controller Manager 可以追蹤每個 Controller 拿了哪些 Handle，檢測是否有兩個 Controller 同時搶寫同一顆 joint。

3. **安全性** — Controller 只能透過註冊好的 Handle 存取特定資源，不能隨意存取 RobotHW 的所有內部資料。

> **比喻：** Handle 就像圖書館的借閱系統。圖書館裡有很多書（RobotHW 的內部變數），但你不能直接走進書庫自己拿。你要先透過系統（Interface）查詢書名（joint name），系統會給你一個借閱證（Handle），你憑借閱證就能合法地讀取或借走特定的書。圖書館管理員（Controller Manager）也能透過系統知道誰借了什麼書，避免兩個人搶同一本。

---

### 3.5 Controller（控制演算法）

**職責：真正的控制演算法。在 ros_control 的架構下，它只負責「讀 state → 跑演算法 → 寫 command」。**

Controller 不直接操作底層 Driver，也不知道硬體長什麼樣子。它只需要知道：
- 我要讀哪個 joint 的 state → 從某個 Interface 找到對應的 Handle
- 我要寫哪個 joint 的 command → 透過 Handle 呼叫 `setCommand()`

**Controller 在 `init()` 時取得 Handle 的流程：**

```
robot_hw（RobotHW 指標）
  → robot_hw->get<HybridJointInterface>()     // 拿到某個 Interface
    → hybridJointInterface.getHandle("LF_HFE") // 從 Interface 裡找到特定 joint 的 Handle
```

**在 repo 中的實際流程：**

在 SIL 版的 `LeggedController::init()` 和 HIL 版的 `LeggedHilController::init()` 裡面，Controller 會遍歷所有 joint name，從 RobotHW 拿到對應的 Handle 並存起來，之後每次 `update()` 都透過這些 Handle 讀 state、寫 command。

#### Controller 接收外部命令的方式

Controller 可以透過 ROS 介面（topic / service）接收外部命令。但在我們的 repo 中，上層的 reference 不是一般的 ROS topic：

- **使用者的速度指令和步態切換** → 透過 ROS topic 進入
- **MPC 的 reference trajectory** → MPC 跑在另一個 thread，透過 `mpcMrtInterface_`（OCS2 的 MRT 機制）將 policy 傳給 Controller 的 `update()` 使用

---

### 3.6 Controller Manager（排程器 / 管理員）

**Controller Manager 不是「層」，它是一個排程器。** 它不在資料流路徑上，而是站在旁邊管理事情的角色。

Controller Manager 的職責：

1. **生命週期管理** — 提供 ROS service 來載入、卸載、啟動、停止 Controller
   - `load_controller`
   - `unload_controller`
   - `switch_controller`
   - `list_controllers`

2. **Update 排程** — 在控制迴圈中，依序呼叫所有正在運行的 Controller 的 `update()`

3. **資源衝突檢查** — 追蹤哪些資源被哪些 Controller 使用，如果兩個 Controller 同時想寫同一顆 joint 的 command（`ClaimResources`），會拒絕啟動。

**在 repo 中的建立方式：**

```cpp
// LeggedHWLoop 或 LeggedHilHWLoop 的 constructor 裡
controllerManager_.reset(
    new controller_manager::ControllerManager(hardwareInterface_.get(), nh_)
);
```

Controller Manager 在建立時，就會拿到 RobotHW 的指標，所以它知道有哪些 Interface 可用。

---

## 四、控制迴圈：read() → update() → write()

**這是 ROS Control 最核心的心跳。** 整個迴圈在同一個 thread 中執行，以固定頻率循環。

### 4.1 單次迴圈的完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        HWLoop::update()                         │
│                                                                 │
│  ① 計算 elapsed time，檢查是否超過 cycle time threshold          │
│                          ↓                                      │
│  ② hardwareInterface_->read(time, period)                       │
│     RobotHW 從硬體/模擬器讀取最新狀態，寫入內部變數                 │
│     （此時 Handle 指向的記憶體已更新，Controller 可以讀到最新值）    │
│                          ↓                                      │
│  ③ controllerManager_->update(time, period)                     │
│     Controller Manager 依序呼叫所有 active Controller 的 update()  │
│     Controller 從 Handle 讀 state → 跑演算法 → 透過 Handle 寫 cmd │
│     （此時 RobotHW 內部的 cmd 變數已被更新）                       │
│                          ↓                                      │
│  ④ hardwareInterface_->write(time, period)                      │
│     RobotHW 從內部變數讀出 command，送往硬體/模擬器                 │
│                          ↓                                      │
│  ⑤ sleep_until(下一個週期)                                       │
│     等待至下一個週期開始                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 對應的實際程式碼

以下節錄自 `LeggedHWLoop.cpp`（SIL 版）：

```cpp
void LeggedHWLoop::update() {
  const auto currentTime = Clock::now();
  const Duration desiredDuration(1.0 / loopHz_);

  // 計算本次迴圈的 elapsed time
  Duration time_span = std::chrono::duration_cast<Duration>(currentTime - lastTime_);
  elapsedTime_ = ros::Duration(time_span.count());
  lastTime_ = currentTime;

  // 檢查是否超時
  const double cycle_time_error = (elapsedTime_ - ros::Duration(desiredDuration.count())).toSec();
  if (cycle_time_error > cycleTimeErrorThreshold_) {
    ROS_WARN_STREAM("Cycle time exceeded error threshold by: " << cycle_time_error << "s");
  }

  // ② 讀取硬體狀態
  hardwareInterface_->read(ros::Time::now(), elapsedTime_);

  // ③ 呼叫所有 Controller 的 update()
  controllerManager_->update(ros::Time::now(), elapsedTime_);

  // ④ 將命令送往硬體
  hardwareInterface_->write(ros::Time::now(), elapsedTime_);

  // ⑤ 等待下一個週期
  const auto sleepTill = currentTime + std::chrono::duration_cast<Clock::duration>(desiredDuration);
  std::this_thread::sleep_until(sleepTill);
}
```

### 4.3 迴圈頻率設定

| 版本 | 設定檔案 | 頻率參數 | 典型值 |
|------|--------|---------|-------|
| SIL（Unitree Go1） | `legged_unitree_hw/config/go1.yaml` | `loop_frequency` | 500 Hz |
| HIL | `legged_hil_hw/config/legged_hil.yaml` | `loop_frequency` | 依設定 |

### 4.4 資料流全景圖

```
              硬體/模擬器
                  │
            ┌─────┴─────┐
            │  read()   │  ← RobotHW 把硬體資料寫入內部變數
            └─────┬─────┘
                  │
                  ▼ （Handle 指標共享，zero-copy）
            ┌───────────┐
            │ Interface │  ← 存放 Handle 的資料池
            └─────┬─────┘
                  │
                  ▼ （Controller 透過 Handle 讀 state）
            ┌───────────┐
            │ Controller│  ← 跑演算法，透過 Handle 寫 command
            │  update() │
            └─────┬─────┘
                  │
                  ▼ （Handle 指標共享，zero-copy）
            ┌───────────┐
            │ Interface │  ← 同一個資料池，cmd 已被更新
            └─────┬─────┘
                  │
            ┌─────┴─────┐
            │  write()  │  ← RobotHW 把內部變數的 cmd 送往硬體
            └─────┬─────┘
                  │
              硬體/模擬器
```

---

## 五、Handle 註冊的完整範例

以下是 `UnitreeHW::setupJoints()` 的實際程式碼，展示了 Handle 是如何建立並註冊的：

```cpp
bool UnitreeHW::setupJoints() {
  for (const auto& joint : urdfModel_->joints_) {
    // 1. 根據 joint name 判斷是哪條腿、哪個關節
    int leg_index = 0, joint_index = 0;
    if (joint.first.find("RF") != std::string::npos) leg_index = FR_;
    else if (joint.first.find("LF") != std::string::npos) leg_index = FL_;
    // ... 以此類推

    if (joint.first.find("HAA") != std::string::npos) joint_index = 0;
    else if (joint.first.find("HFE") != std::string::npos) joint_index = 1;
    else if (joint.first.find("KFE") != std::string::npos) joint_index = 2;

    int index = leg_index * 3 + joint_index;

    // 2. 建立 JointStateHandle，傳入 state 變數的指標
    hardware_interface::JointStateHandle state_handle(
        joint.first,               // joint name
        &jointData_[index].pos_,   // position 指標
        &jointData_[index].vel_,   // velocity 指標
        &jointData_[index].tau_    // effort 指標
    );
    jointStateInterface_.registerHandle(state_handle);

    // 3. 建立 HybridJointHandle，傳入 command 變數的指標
    hybridJointInterface_.registerHandle(
        HybridJointHandle(
            state_handle,                   // 繼承 state 讀取能力
            &jointData_[index].posDes_,     // posDes 指標
            &jointData_[index].velDes_,     // velDes 指標
            &jointData_[index].kp_,         // Kp 指標
            &jointData_[index].kd_,         // Kd 指標
            &jointData_[index].ff_          // feedforward 指標
        )
    );
  }
  return true;
}
```

**流程總結：**
1. 從 URDF 遍歷所有 joint
2. 根據 joint name 計算在 `jointData_[12]` 中的 index
3. 把 state 變數的指標包成 `JointStateHandle` → 註冊到 `JointStateInterface`
4. 把 command 變數的指標包成 `HybridJointHandle` → 註冊到 `HybridJointInterface`

這之後，任何 Controller 只要拿到 `HybridJointHandle`，就能讀寫這顆 joint 的所有資料。

---

## 六、我們 repo 中的四種 Interface 總覽

| Interface | Handle | 資料內容 | 方向 | 資源策略 |
|-----------|--------|---------|------|---------|
| `JointStateInterface` | `JointStateHandle` | position, velocity, effort | 硬體→Controller（讀） | `DontClaimResources` |
| `HybridJointInterface` (SIL) / `ImpedanceJointInterface` (HIL) | `HybridJointHandle` / `ImpedanceJointHandle` | posDes, velDes, kp, kd, ff（HIL 額外有 timestamp） | Controller→硬體（寫） | `ClaimResources` |
| `ImuSensorInterface` | `ImuSensorHandle` | orientation[4], angular_velocity[3], linear_acceleration[3], covariance | 硬體→Controller（讀） | `DontClaimResources` |
| `ContactSensorInterface` | `ContactSensorHandle` | isContact (bool) | 硬體→Controller（讀） | `DontClaimResources` |

---

## 七、關於 Transmission（官方機制 vs. 我們的做法）

### 官方 ros_control 的 Transmission 機制

官方架構圖中有 **Effort Transmissions** 和 **Forward State Transmissions** 兩個區塊，位於 RobotHW 內部。這是 ros_control 提供的機制，用來處理 Joint Space 和 Actuator Space 之間的轉換。

典型的應用場景：
- 減速比轉換（例如馬達轉 100 圈 = joint 轉 1 圈）
- 差動機構（一個 joint 由兩顆馬達驅動）

```
Controller 算出的是 Joint Space 的值（例如 joint torque = 5 Nm）
    ↓ Effort Transmission（乘上減速比）
送到馬達的是 Actuator Space 的值（例如 motor current = 0.5 A）
```

### 我們 repo 的做法

**我們的 repo 並沒有使用 ros_control 官方的 Transmission 機制。** Joint Space 和 Actuator Space 之間的轉換（如果有需要的話），是直接在 `read()` / `write()` 函式裡面自行處理的。

例如在 `UnitreeHW::write()` 中：

```cpp
void UnitreeHW::write(const ros::Time& /*time*/, const ros::Duration& /*period*/) {
  for (int i = 0; i < 12; ++i) {
    lowCmd_.motorCmd[i].q   = static_cast<float>(jointData_[i].posDes_);
    lowCmd_.motorCmd[i].dq  = static_cast<float>(jointData_[i].velDes_);
    lowCmd_.motorCmd[i].Kp  = static_cast<float>(jointData_[i].kp_);
    lowCmd_.motorCmd[i].Kd  = static_cast<float>(jointData_[i].kd_);
    lowCmd_.motorCmd[i].tau = static_cast<float>(jointData_[i].ff_);
  }
  safety_->PositionLimit(lowCmd_);
  safety_->PowerProtect(lowCmd_, lowState_, powerLimit_);
  udp_->SetSend(lowCmd_);
  udp_->Send();
}
```

這裡直接把 `jointData_` 裡的值打包成 Unitree SDK 的 `lowCmd_` 格式，透過 UDP 送出去。沒有經過任何 Transmission 物件。

---

## 八、啟動入口

### SIL 版（Gazebo 模擬）

SIL 版透過 Gazebo plugin 方式啟動。Gazebo 會自動載入 `gazebo_ros_control` plugin，這個 plugin 內部會建立 RobotHWSim（模擬版的 RobotHW），然後啟動控制迴圈。

### HIL 版

HIL 版有自己的 main 函式作為啟動入口：

```cpp
// legged_hil_lcm_hw/src/legged_hil_lcm_hw_node.cpp
int main(int argc, char** argv) {
  ros::init(argc, argv, "legged_hil_hw");
  ros::NodeHandle nh;
  ros::NodeHandle robotHwNh("~");

  ros::AsyncSpinner spinner(3);  // 3 個 thread 處理 callback
  spinner.start();

  // 建立 RobotHW
  std::shared_ptr<legged::LcmHW> lcmHw = std::make_shared<legged::LcmHW>();
  lcmHw->init(nh, robotHwNh);

  // 啟動控制迴圈
  legged::LeggedHilHWLoop controlLoop(nh, lcmHw);

  ros::waitForShutdown();
  return 0;
}
```

HIL 版也支援動態選擇後端通訊方式（LCM 或 SHM），在 `legged_hil_hybrid_hw_node.cpp` 中透過 ROS parameter 決定。

---

## 九、Repo 對照表

| 概念 | SIL（legged_control） | HIL（legged_control_hil） |
|------|----------------------|--------------------------|
| RobotHW 基底類別 | `legged_hw/LeggedHW.h` | `legged_hil_hw/LeggedHilHW.h` |
| RobotHW 實機實作 | `legged_unitree_hw/UnitreeHW.h` | `legged_hil_lcm_hw/LcmHW.h`, `legged_hil_shm_hw/ShmHW.h` |
| 控制迴圈 | `legged_hw/LeggedHWLoop.cpp` | `legged_hil_hw/LeggedHilHWLoop.cpp` |
| Joint Command Interface | `HybridJointInterface.h` | `ImpedanceJointInterface.h` |
| Contact Interface | `legged_common/.../ContactSensorInterface.h` | `legged_hil_interface/.../ContactSensorInterface.h` |
| 啟動入口 | Gazebo plugin 自動載入 | `legged_hil_lcm_hw_node.cpp` |
| 迴圈頻率設定 | `go1.yaml`（500 Hz） | `legged_hil.yaml` |

---

## 十、如果我要修改某個行為，應該改哪裡？

| 我想要... | 應該改的位置 |
|----------|------------|
| 支援一台新的機器人硬體 | 繼承 `LeggedHW`，實作新的 `read()` 和 `write()` |
| 新增一種感測器資料（例如 LiDAR） | ① 定義新的 Handle 和 Interface ② 在 RobotHW 的 `init()` 中建立 Handle 並 `registerInterface()` ③ 在 Controller 的 `init()` 中取得 Handle |
| 修改控制迴圈頻率 | 修改 yaml 設定檔中的 `loop_frequency` |
| 修改控制演算法 | 改 Controller 的 `update()` 裡面的邏輯 |
| 修改 impedance control 的參數組合 | 修改 `HybridJointHandle` / `ImpedanceJointHandle` 的欄位定義 |
| 新增安全保護 | 在 `write()` 裡面加入（例如 `safety_->PositionLimit()`） |

---

## 十一、必須記住的核心概念

1. **Handle = 指標包裝器** — RobotHW 和 Controller 透過 Handle 共享同一塊記憶體，這是整個架構 zero-copy 運作的根本。

2. **Controller Manager 是排程器，不是層** — 它管生命週期、管 update 順序、管資源衝突，但它本身不在資料流路徑上。

3. **我們的 repo 用自定義的 HybridJointInterface / ImpedanceJointInterface** — 一次寫五個參數（posDes, velDes, kp, kd, ff），對應 impedance control 公式。

4. **Transmission 在我們的 repo 中未使用官方機制** — 轉換邏輯直接寫在 `read()` / `write()` 裡面。

5. **read → update → write 是同一個 thread 裡的三步驟** — 由 HWLoop 類以固定頻率驅動，是整個控制系統的心跳。

6. **State Interface 和 Command Interface 分開的原因** — 不只是資料流方向不同，更關鍵的是資源管理策略不同：State 允許多讀，Command 只允許獨佔寫入。

7. **要支援新硬體，只需要實作新的 `read()` 和 `write()`** — Controller 完全不用改，這就是解耦的價值。
