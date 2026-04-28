#!/usr/bin/env python3
"""
fleet_manager_cbf.py — 模組化多機編隊管理器（含 CBF 安全濾波器）

架構:
    FleetManagerCBF
    ├── StateCollector        讀取 /dogN/ground_truth/state
    ├── FormationPlanner      產生每個 follower 的 formation target
    ├── NominalController     PID 追蹤 → u_nominal (followers)
    ├── LeaderCmdRelay        訂閱 /leader/cmd_vel_raw → leader 的 u_nominal
    ├── CBFSafetyFilter       robot-robot + obstacle + wall avoidance (ALL dogs)
    ├── VelocityLimiter       限制 vx, vy, wz
    └── CmdVelPublisher       發布 /dogN/cmd_vel (ALL dogs)

改動重點 (vs 舊版):
    - Leader 也是 QP 決策變數，也受 CBF 保護
    - Leader 的 nominal 來自 /dog1/cmd_vel_raw (你用 rostopic pub 發到這裡)
    - CBF 發布安全修正後的速度到 /dog1/cmd_vel (所有狗都經過 CBF)
    - Wall CBF 也保護所有狗（含 leader）

不碰底層 C++ / OCS2 / MPC / WBC。
依賴: numpy, cvxpy (pip install numpy cvxpy)

啟動方式 (不再需要 rosparam load):
    rosrun legged_controllers formation_managerCBF.py
"""

import math
import os
import yaml
import numpy as np
import cvxpy as cp
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ── 讀取 YAML config ──
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbf_params.yaml")

def _load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

_CFG = _load_config(_CONFIG_PATH)


# ═══════════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════════

def quaternion_to_yaw(q):
    """四元數 → yaw 角"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    """角度歸一化到 [-pi, pi]"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot2d(yaw):
    """2D 旋轉矩陣 R(yaw): body → world"""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])


# ═══════════════════════════════════════════════════════════════
# Module 1: StateCollector
# ═══════════════════════════════════════════════════════════════

class RobotState:
    """單隻狗的狀態"""
    __slots__ = ("x", "y", "yaw", "vx_world", "vy_world", "received")

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx_world = 0.0
        self.vy_world = 0.0
        self.received = False

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def vel_world(self):
        return np.array([self.vx_world, self.vy_world])


class StateCollector:
    """
    訂閱所有狗的 /dogN/ground_truth/state，維護 states dict。
    """

    def __init__(self, dog_names):
        self.states = {name: RobotState() for name in dog_names}
        self._subs = []

        for name in dog_names:
            sub = rospy.Subscriber(
                f"/{name}/ground_truth/state",
                Odometry,
                self._odom_cb,
                callback_args=name,
                queue_size=1,
            )
            self._subs.append(sub)

    def _odom_cb(self, msg, name):
        s = self.states[name]
        s.x = msg.pose.pose.position.x
        s.y = msg.pose.pose.position.y
        s.yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        R = rot2d(s.yaw)
        v_body = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.linear.y])
        v_world = R @ v_body
        s.vx_world = v_world[0]
        s.vy_world = v_world[1]

        s.received = True

    def all_received(self, names):
        return all(self.states[n].received for n in names)


# ═══════════════════════════════════════════════════════════════
# Module 2: FormationPlanner
# ═══════════════════════════════════════════════════════════════

class FormationPlanner:
    """
    根據 leader 的位姿 + 預設 offset，計算每個 follower 的目標位姿。
    """

    def __init__(self, offsets):
        self.offsets = {k: [float(v[0]), float(v[1])] for k, v in offsets.items()}

    def compute_target(self, leader_state, follower_name):
        dx, dy = self.offsets.get(follower_name, [-1.0, 0.0])
        yaw = leader_state.yaw

        x_ref = leader_state.x + math.cos(yaw) * dx - math.sin(yaw) * dy
        y_ref = leader_state.y + math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw_ref = yaw
        return x_ref, y_ref, yaw_ref

    def update_offsets(self, new_offsets):
        self.offsets = {k: [float(v[0]), float(v[1])] for k, v in new_offsets.items()}
        rospy.loginfo("[FormationPlanner] offsets updated: %s", self.offsets)


# ═══════════════════════════════════════════════════════════════
# Module 3: NominalController
# ═══════════════════════════════════════════════════════════════

class NominalController:
    """
    P 控制器：根據 (目標位姿 - 當前位姿) 產生 body frame 的 u_nominal。
    """

    def __init__(self, kp_x, kp_y, kp_yaw, pos_tol, yaw_tol):
        self.kp_x = kp_x
        self.kp_y = kp_y
        self.kp_yaw = kp_yaw
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol

    def compute(self, follower_state, x_ref, y_ref, yaw_ref):
        ex_w = x_ref - follower_state.x
        ey_w = y_ref - follower_state.y
        e_yaw = wrap_to_pi(yaw_ref - follower_state.yaw)

        c, s = math.cos(follower_state.yaw), math.sin(follower_state.yaw)
        ex_b =  c * ex_w + s * ey_w
        ey_b = -s * ex_w + c * ey_w

        if math.hypot(ex_w, ey_w) < self.pos_tol and abs(e_yaw) < self.yaw_tol:
            return 0.0, 0.0, 0.0

        vx = self.kp_x * ex_b
        vy = self.kp_y * ey_b
        wz = self.kp_yaw * e_yaw
        return vx, vy, wz


# ═══════════════════════════════════════════════════════════════
# Module 3.5: LeaderCmdRelay
# ═══════════════════════════════════════════════════════════════

class LeaderCmdRelay:
    """
    訂閱 /<leader>/cmd_vel_raw，取得 leader 的 nominal 指令。
    你用 rostopic pub 發到 /dog1/cmd_vel_raw，這裡接收後交給 CBF 處理。

    資料流:
        rostopic pub /dog1/cmd_vel_raw → LeaderCmdRelay → u_nom_leader
        → CBF → u_safe_leader → /dog1/cmd_vel → MPC
    """

    def __init__(self, leader_name):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.received = False

        rospy.Subscriber(
            f"/{leader_name}/cmd_vel_raw",
            Twist,
            self._cb,
            queue_size=1,
        )

    def _cb(self, msg):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z
        self.received = True

    def get_nominal(self):
        """回傳 (vx, vy), wz"""
        return (self.vx, self.vy), self.wz


# ═══════════════════════════════════════════════════════════════
# Module 4: CBFSafetyFilter
# ═══════════════════════════════════════════════════════════════

class CBFSafetyFilter:
    """
    CBF-QP 安全濾波器。所有狗都是決策變數。

    決策變數: u = [vx_dog1, vy_dog1, vx_dog2, vy_dog2, vx_dog3, vy_dog3]
    全部在 body frame，wz pass-through 不進 QP。
    """

    def __init__(self, gamma_robot, d_min, gamma_obs=1.0, gamma_wall=1.0):
        self.gamma_robot = gamma_robot
        self.d_min = d_min
        self.d_min_sq = d_min ** 2
        self.gamma_obs = gamma_obs
        self.gamma_wall = gamma_wall

        self.obstacles = []
        self.walls = []

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        rospy.loginfo("[CBFSafetyFilter] %d obstacles loaded", len(obstacles))

    def set_walls(self, walls):
        self.walls = walls
        rospy.loginfo("[CBFSafetyFilter] %d walls loaded", len(walls))

    def solve(self, all_dogs, states, u_nom_dict):
        """
        Parameters
        ----------
        all_dogs   : list[str]          所有狗名（含 leader）
        states     : dict[str, RobotState]
        u_nom_dict : dict[str, tuple(vx, vy)]  每隻狗的 body frame nominal

        Returns
        -------
        dict[str, tuple(vx, vy)]  所有狗的安全修正後速度
        """
        n_dogs = len(all_dogs)
        n_vars = 2 * n_dogs
        dog_idx = {name: i for i, name in enumerate(all_dogs)}

        # 組裝 nominal 向量
        u_nom = np.zeros(n_vars)
        for name in all_dogs:
            i = dog_idx[name]
            u_nom[2 * i]     = u_nom_dict[name][0]
            u_nom[2 * i + 1] = u_nom_dict[name][1]

        A_rows = []
        b_rows = []

        # ── Pairwise robot-robot CBF ──
        for ia in range(n_dogs):
            for ib in range(ia + 1, n_dogs):
                na, nb = all_dogs[ia], all_dogs[ib]
                sa, sb = states[na], states[nb]
                if not sa.received or not sb.received:
                    continue

                dp = sa.pos - sb.pos
                h = float(dp @ dp) - self.d_min_sq

                a_row = np.zeros(n_vars)

                # 兩隻狗都是決策變數
                col_a = 2 * dog_idx[na]
                a_row[col_a:col_a + 2] = 2.0 * dp @ rot2d(sa.yaw)

                col_b = 2 * dog_idx[nb]
                a_row[col_b:col_b + 2] = -2.0 * dp @ rot2d(sb.yaw)

                b_val = -self.gamma_robot * h

                A_rows.append(a_row)
                b_rows.append(b_val)

        # ── Obstacle CBF (所有狗) ──
        for obs in self.obstacles:
            p_obs = np.array(obs["pos"][:2])
            r_obs = float(obs["radius"])

            for name in all_dogs:
                s = states[name]
                if not s.received:
                    continue

                dp = s.pos - p_obs
                h_obs = float(dp @ dp) - r_obs ** 2

                a_row = np.zeros(n_vars)
                col = 2 * dog_idx[name]
                a_row[col:col + 2] = 2.0 * dp @ rot2d(s.yaw)

                b_val = -self.gamma_obs * h_obs

                A_rows.append(a_row)
                b_rows.append(b_val)

        # ── Wall CBF (所有狗) ──
        for wall in self.walls:
            n_w = np.array(wall["normal"][:2], dtype=float)
            p_w = np.array(wall["point"][:2], dtype=float)
            d_safe = float(wall.get("d_safe", 0.4))

            for name in all_dogs:
                s = states[name]
                if not s.received:
                    continue

                h_wall = float(n_w @ (s.pos - p_w)) - d_safe

                a_row = np.zeros(n_vars)
                col = 2 * dog_idx[name]
                a_row[col:col + 2] = n_w @ rot2d(s.yaw)

                b_val = -self.gamma_wall * h_wall

                A_rows.append(a_row)
                b_rows.append(b_val)

        # ── 建構 QP ──
        u = cp.Variable(n_vars)
        objective = cp.Minimize(cp.sum_squares(u - u_nom))

        constraints = []
        if A_rows:
            A = np.array(A_rows)
            b = np.array(b_rows)
            constraints.append(A @ u >= b)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            rospy.logwarn("[CBF] Solver failed, returning nominal")
            return dict(u_nom_dict)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            rospy.logwarn("[CBF] QP status=%s, returning nominal", prob.status)
            return dict(u_nom_dict)

        u_sol = u.value

        result = {}
        for name in all_dogs:
            i = dog_idx[name]
            result[name] = (float(u_sol[2 * i]), float(u_sol[2 * i + 1]))

        return result


# ═══════════════════════════════════════════════════════════════
# Module 5: VelocityLimiter
# ═══════════════════════════════════════════════════════════════

class VelocityLimiter:
    def __init__(self, max_vx, max_vy, max_wz):
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_wz = max_wz

    def clamp(self, vx, vy, wz):
        vx = max(-self.max_vx, min(vx, self.max_vx))
        vy = max(-self.max_vy, min(vy, self.max_vy))
        wz = max(-self.max_wz, min(wz, self.max_wz))
        return vx, vy, wz


# ═══════════════════════════════════════════════════════════════
# Module 6: CmdVelPublisher
# ═══════════════════════════════════════════════════════════════

class CmdVelPublisher:
    """
    管理所有狗（含 leader）的 /dogN/cmd_vel Publisher。
    """

    def __init__(self, all_dog_names):
        self._pubs = {}
        for name in all_dog_names:
            self._pubs[name] = rospy.Publisher(
                f"/{name}/cmd_vel", Twist, queue_size=1
            )

    def publish(self, name, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = wz
        self._pubs[name].publish(cmd)

    def publish_zero(self, names):
        zero = Twist()
        for name in names:
            self._pubs[name].publish(zero)


# ═══════════════════════════════════════════════════════════════
# 主控: FleetManagerCBF
# ═══════════════════════════════════════════════════════════════

class FleetManagerCBF:
    """
    主控流程（每個 cycle @ rate_hz）:

        Leader:    /dog1/cmd_vel_raw → LeaderCmdRelay → u_nom_leader
        Followers: odom → FormationPlanner → NominalController → u_nom_follower
        All:       u_nom → CBFSafetyFilter → VelocityLimiter → CmdVelPublisher

    操作方式:
        以前: rostopic pub /dog1/cmd_vel ...
        現在: rostopic pub /dog1/cmd_vel_raw ...
              (CBF 會修正後再發到 /dog1/cmd_vel)
    """

    def __init__(self):
        rospy.init_node("fleet_manager_cbf")

        # ── 基本參數 ──
        self.leader_name = rospy.get_param("~leader_name", "dog1")
        self.follower_names = rospy.get_param("~follower_names", ["dog2", "dog3"])
        self.all_dogs = [self.leader_name] + self.follower_names
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.stop_without_leader = rospy.get_param("~stop_without_leader", True)

        # ── Module 1: StateCollector ──
        self.state_collector = StateCollector(self.all_dogs)

        # ── Module 2: FormationPlanner ──
        default_offsets = _CFG.get("offsets", {"dog2": [-1.0, 1.0], "dog3": [-1.0, -1.0]})
        offsets = rospy.get_param("~offsets", default_offsets)
        self.planner = FormationPlanner(offsets)

        # ── Module 3: NominalController ──
        self.controller = NominalController(
            kp_x=rospy.get_param("~kp_x", 0.8),
            kp_y=rospy.get_param("~kp_y", 0.8),
            kp_yaw=rospy.get_param("~kp_yaw", 1.0),
            pos_tol=rospy.get_param("~pos_tolerance", 0.15),
            yaw_tol=rospy.get_param("~yaw_tolerance", 0.15),
        )

        # ── Module 3.5: LeaderCmdRelay ──
        self.leader_relay = LeaderCmdRelay(self.leader_name)

        # ── Module 4: CBFSafetyFilter ──
        self.cbf_enabled = rospy.get_param("~cbf_enabled", _CFG.get("cbf_enabled", True))
        self.cbf_filter = CBFSafetyFilter(
            gamma_robot=rospy.get_param("~cbf_gamma", _CFG.get("cbf_gamma", 2.0)),
            d_min=rospy.get_param("~cbf_d_min", _CFG.get("cbf_d_min", 1.0)),
            gamma_obs=rospy.get_param("~cbf_gamma_obs", _CFG.get("cbf_gamma_obs", 2.0)),
            gamma_wall=rospy.get_param("~cbf_gamma_wall", _CFG.get("cbf_gamma_wall", 2.0)),
        )

        # 障礙物 & 牆壁: YAML → rosparam 可覆蓋
        obstacles = rospy.get_param("~obstacles", _CFG.get("obstacles", []))
        if obstacles:
            self.cbf_filter.set_obstacles(obstacles)

        walls = rospy.get_param("~walls", _CFG.get("walls", []))
        if walls:
            self.cbf_filter.set_walls(walls)

        # ── Module 5: VelocityLimiter ──
        self.limiter = VelocityLimiter(
            max_vx=rospy.get_param("~max_vx", 0.5),
            max_vy=rospy.get_param("~max_vy", 0.3),
            max_wz=rospy.get_param("~max_wz", 0.8),
        )

        # ── Module 6: CmdVelPublisher (ALL dogs) ──
        self.cmd_pub = CmdVelPublisher(self.all_dogs)

        # ── 等待連線 ──
        rospy.sleep(1.0)
        rospy.loginfo("=" * 60)
        rospy.loginfo("[FleetManagerCBF] READY")
        rospy.loginfo("  leader     = %s", self.leader_name)
        rospy.loginfo("  followers  = %s", self.follower_names)
        rospy.loginfo("  offsets    = %s", self.planner.offsets)
        rospy.loginfo("  CBF        = %s (gamma=%.2f, d_min=%.2f)",
                      self.cbf_enabled,
                      self.cbf_filter.gamma_robot,
                      self.cbf_filter.d_min)
        rospy.loginfo("  obstacles  = %d", len(self.cbf_filter.obstacles))
        rospy.loginfo("  walls      = %d", len(self.cbf_filter.walls))
        rospy.loginfo("  rate       = %.0f Hz", self.rate_hz)
        rospy.loginfo("  NOTE: send leader cmd to /%s/cmd_vel_raw", self.leader_name)
        rospy.loginfo("=" * 60)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        states = self.state_collector.states

        while not rospy.is_shutdown():
            # ── 前置檢查 ──
            if self.stop_without_leader and not states[self.leader_name].received:
                self.cmd_pub.publish_zero(self.all_dogs)
                rate.sleep()
                continue

            leader = states[self.leader_name]

            # ── Step 1: Leader nominal (from cmd_vel_raw) ──
            leader_nom, leader_wz = self.leader_relay.get_nominal()

            # ── Step 2: Follower formation targets + PID ──
            u_nom = {}
            wz_nom = {}

            u_nom[self.leader_name] = leader_nom
            wz_nom[self.leader_name] = leader_wz

            for name in self.follower_names:
                x_ref, y_ref, yaw_ref = self.planner.compute_target(leader, name)
                vx, vy, wz = self.controller.compute(
                    states[name], x_ref, y_ref, yaw_ref
                )
                u_nom[name] = (vx, vy)
                wz_nom[name] = wz

            # ── Step 3: CBF safety filter (ALL dogs) ──
            if self.cbf_enabled:
                u_safe = self.cbf_filter.solve(
                    self.all_dogs,
                    states,
                    u_nom,
                )
            else:
                u_safe = u_nom

            # ── Step 4: Velocity limiter + publish (ALL dogs) ──
            for name in self.all_dogs:
                vx, vy = u_safe[name]
                wz = wz_nom[name]
                vx, vy, wz = self.limiter.clamp(vx, vy, wz)
                self.cmd_pub.publish(name, vx, vy, wz)

            rate.sleep()


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        manager = FleetManagerCBF()
        manager.spin()
    except rospy.ROSInterruptException:
        pass