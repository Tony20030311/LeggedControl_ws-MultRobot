#!/usr/bin/env python3
"""
formation_managerCBF.py — 模組化多機編隊管理器（含 CBF 安全濾波器）

架構 (Step 1):
    FleetManagerCBF
    ├── StateCollector           讀取 /dogN/ground_truth/state
    ├── AStarPlanner             A* 網格路徑規劃（啟動時建立障礙地圖）
    ├── PurePursuitController    Pure Pursuit 路徑追蹤（全向輪體態）
    ├── LeaderNavigator          AUTO(A*+PP) / KEYBOARD 模式切換
    │   └── LeaderCmdRelay       訂閱 /dog1/cmd_vel_raw 鍵盤模式
    ├── FormationPlanner         產生 follower formation target（Step 2+）
    ├── NominalController        PID 追蹤Step 2+，followers_stationary=false）
    ├── CBFSafetyFilter          robot-robot + obstacle + wall（ALL dogs）
    ├── VelocityLimiter          限制 vx, vy, wz
    └── CmdVelPublisher          發布 /dogN/cmd_vel（ALL dogs）

Step 1 資料流:
    AUTO:  /dog1/goal → LeaderNavigator → A* → waypoints
                      → PurePursuit → u_nom → CBF → /dog1/cmd_vel
    KBOARD: /dog1/cmd_vel_raw → LeaderCmdRelay → u_nom → CBF → /dog1/cmd_vel
    Followers: 發零速度（CBF 仍然啟動，保護所有狗）


啟動: rosrun legged_controllers formation_managerCBF.py
"""

import heapq
import math
import os
import threading
import yaml
import numpy as np
import cvxpy as cp
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

# ── 讀取 YAML config ──
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbf_params.yaml") #__file__是這個檔案本身的路徑
# os.path.abspath取資料夾 所以就是取跟這個檔案一樣路徑的 cbf_params.yaml

def _load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {} # 讀yaml 如果沒有這個檔案 回傳空的


_CFG = _load_config(_CONFIG_PATH) # 整個yaml存成_CFG 例如 _CFG.get("cbf_d_min",1.0) 代表從yaml讀cbf_d_min 讀不到就是用default = 1.0


# ═══════════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════════

def quaternion_to_yaw(q): # 四元數是ROS表示姿態的格式 轉成yaw （繞z)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi # 把角度包在 -pi 到 pi 之間 所以370度會變成10度 


def rot2d(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]]) # 旋轉矩陣 world frame （Gazebo) 去轉 body frame （MPC)


def closest_point_on_aabb(point, center, size):
    half = 0.5 * np.array(size[:2], dtype=float)
    center = np.array(center[:2], dtype=float)
    return np.minimum(np.maximum(point, center - half), center + half)


# ═══════════════════════════════════════════════════════════════
# Module 1: StateCollector
# ═══════════════════════════════════════════════════════════════


# Gazebo
#   ↓ 發布 /dogN/ground_truth/state
# StateCollector（訂閱，持續更新）
#   ↓ 寫入
# RobotState（三份，每隻狗一份）
#   ↑ 讀取
#   ├── FormationPlanner  → 讀 leader 的 x, y, yaw，算 follower 目標點
#   ├── NominalController → 讀 follower 的 x, y, yaw，算 PID 誤差
#   ├── PurePursuitController → 讀 leader 的 x, y, yaw，算追蹤速度
#   ├── CBFSafetyFilter   → 讀所有狗的 x, y, yaw，算安全約束
#   └── LeaderNavigator   → 讀 leader 的 x, y，判斷是否到達目標

class RobotState: # 整個系統共享這個狀態 這邊都是world frame 以上五個讀取Robot State的 需要body frame就自己轉
    __slots__ = ("x", "y", "yaw", "vx_world", "vy_world", "received")
    # __slots__ 是 Python 的記憶體優化，明確宣告這個 class 只有這幾個欄位，不允許動態新增屬性。因為每個 cycle 都要讀這個 class，節省記憶體有意義。
    def __init__(self):
        self.x = 0.0          # world frame x 座標（公尺）
        self.y = 0.0          # world frame y 座標（公尺）
        self.yaw = 0.0        # 朝向角（rad，world frame）
        self.vx_world = 0.0   # world frame x 方向速度
        self.vy_world = 0.0   # world frame y 方向速度
        self.received = False  # 是否已收到第一筆 odom 資料

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def vel_world(self):
        return np.array([self.vx_world, self.vy_world])


class StateCollector: # 負責寫入狀態到 RobotState 而且是寫成world frame
    def __init__(self, dog_names):
        self.states = {name: RobotState() for name in dog_names}
        self._subs = []
        for name in dog_names:
            sub = rospy.Subscriber(
                f"/{name}/ground_truth/state", # 訂閱狗的ground_truth/state
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
        s.yaw = quaternion_to_yaw(msg.pose.pose.orientation)   # 從訊息裡取出位置和yaw，存進 RobotState 
        R = rot2d(s.yaw) 
        v_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        v_world = R @ v_body
        s.vx_world = v_world[0]
        s.vy_world = v_world[1]
        s.received = True
        # 訊息裡的速度是 body frame（MPC 輸出的），CBF 需要 world frame 的速度。所以用旋轉矩陣 R 把 body frame 速度轉成 world frame，存進 RobotState。
    def all_received(self, names):
        return all(self.states[n].received for n in names)


# ═══════════════════════════════════════════════════════════════
# Module 2: FormationPlanner
# ═══════════════════════════════════════════════════════════════

class FormationPlanner: # 根據 leader 的位置和朝向，算出每隻 follower 應該站在哪裡 
    def __init__(self, offsets):
        self.offsets = {k: [float(v[0]), float(v[1])] for k, v in offsets.items()} #讀yaml裡面的offset 這是 leader的body frame定義的 不管leader朝向 都一樣

    def compute_target(self, leader_state, follower_name):
        dx, dy = self.offsets.get(follower_name, [-1.0, 0.0])
        yaw = leader_state.yaw # follower 的 yaw 跟 leader yaw 要一樣
        # 知道leader 在 world frame 的位置：(xL,yL) + leader 的朝向：ψ + follower相對leader的offset
        # 找follower 在 world frame 的位置 (xref,yref)(x_{ref}, y_{ref})
        x_ref = leader_state.x + math.cos(yaw) * dx - math.sin(yaw) * dy
        y_ref = leader_state.y + math.sin(yaw) * dx + math.cos(yaw) * dy # 作法就是把body frame 的 offset(dx,dy) 轉到 world frame
        return x_ref, y_ref, yaw
        # 這邊是follower根據leader 找出目標點應該要去哪 還沒涉及到應該怎麼走過去 （/cmd_vel
    def update_offsets(self, new_offsets):
        self.offsets = {k: [float(v[0]), float(v[1])] for k, v in new_offsets.items()}
        rospy.loginfo("[FormationPlanner] offsets updated: %s", self.offsets) # 變換隊形用


# ═══════════════════════════════════════════════════════════════
# Module 3: NominalController
# ═══════════════════════════════════════════════════════════════
# FormationPlanner.compute_target()
#         ↓ (x_ref, y_ref, yaw_ref)
# NominalController.compute()
#         ↓ (vx, vy, wz)

class NominalController: # formationPlanner規劃出目標點 來到這裡進行速度規劃 （P control 位置誤差 → 速度指令）
    def __init__(self, kp_x, kp_y, kp_yaw, pos_tol, yaw_tol,
                 align_yaw=True, align_yaw_while_moving=False,
                 moving_yaw_min_dist=0.25):
        self.kp_x = kp_x
        self.kp_y = kp_y
        self.kp_yaw = kp_yaw
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol # 五個參數，從 self.controller = NominalController 傳進來：
        self.align_yaw = bool(align_yaw)
        self.align_yaw_while_moving = bool(align_yaw_while_moving)
        self.moving_yaw_min_dist = float(moving_yaw_min_dist)

    def compute(self, follower_state, x_ref, y_ref, yaw_ref): # 四個輸入
        ex_w = x_ref - follower_state.x  
        ey_w = y_ref - follower_state.y
        e_yaw = wrap_to_pi(yaw_ref - follower_state.yaw) # 先算world frame下的誤差 
        
        c, s = math.cos(follower_state.yaw), math.sin(follower_state.yaw)
        ex_b =  c * ex_w + s * ey_w
        ey_b = -s * ex_w + c * ey_w # 把位置誤差從 world frame 轉成 body frame 
        
        pos_err = math.hypot(ex_w, ey_w)

        if pos_err < self.pos_tol and (not self.align_yaw or abs(e_yaw) < self.yaw_tol):
            return 0.0, 0.0, 0.0 # 位置誤差 < 0.15m 且朝向誤差 < 0.15rad，認為到達，輸出零速度。

        if self.align_yaw:
            wz = self.kp_yaw * e_yaw
        elif self.align_yaw_while_moving and pos_err > self.moving_yaw_min_dist:
            wz = self.kp_yaw * e_yaw
        else:
            wz = 0.0
        return self.kp_x * ex_b, self.kp_y * ey_b, wz # 算速度 
        # vx = kp x ex_b , vy = 0.8 x ey_b , wz = 1.0 * e_yaw

# ═══════════════════════════════════════════════════════════════
# Module A: AStarPlanner
# ═══════════════════════════════════════════════════════════════

class AStarPlanner: # 規劃路徑
    """
    Grid-based A* 路徑規劃器。

    障礙地圖在 __init__ 時一次性建立（numpy 向量化，快速）。
    plan() 每次收到新 goal 才呼叫，回傳 world frame 的 (x, y) waypoint list。

    robot_radius = 0.25m：A* 膨脹半徑，比 CBF d_min=1.0 小，
    因為 CBF 負責真正的安全保護；A* 只需要保證路徑幾何上可行。
    """

    _MOVES = [
        (1,  0, 1.0),          (-1,  0, 1.0),
        (0,  1, 1.0),          ( 0, -1, 1.0),
        (1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ] # 8 個移動方向，每個是 (dx, dy, cost) 斜線就是根號2

    def __init__(self, resolution, robot_radius, obstacles,
                 x_min=0.0, x_max=10.0, y_min=-5.0, y_max=5.0,
                 boundary_margin=0.45, rect_obstacles=None):
        self.res          = float(resolution) # astar_resolution: 0.1 格子大小決定的是路徑的精細程度
        self.robot_radius = float(robot_radius) # astar_robot_radius: 0.25
        self.boundary_margin = float(boundary_margin)
        self.rect_obstacles = rect_obstacles or []
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max) # yaml 參數
        
        self.nx = int(round((self.x_max - self.x_min) / self.res)) + 1
        self.ny = int(round((self.y_max - self.y_min) / self.res)) + 1 # 算整張地圖有幾格
        
        self._omap = self._build_map(obstacles, self.rect_obstacles)
        rospy.loginfo(
            "[AStarPlanner] grid %dx%d (%.1fm×%.1fm, res=%.2fm), "
            "free=%d, obstacles=%d, rects=%d",
            self.nx, self.ny,
            self.x_max - self.x_min, self.y_max - self.y_min,
            self.res, int(np.sum(~self._omap)), len(obstacles),
            len(self.rect_obstacles),
        ) 

    def _build_map(self, obstacles, rect_obstacles): # 建立障礙地圖 ture = 有障礙 false代表可以走 
        """每個 grid cell 標記是否被佔用（含 robot_radius 膨脹）。"""
        xs = np.arange(self.nx) * self.res + self.x_min   # shape (nx,)
        ys = np.arange(self.ny) * self.res + self.y_min   # shape (ny,)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')        # shape (nx, ny)

        inflate = self.robot_radius # astar_robot_radius: 0.25 膨脹半徑
        omap = np.zeros((self.nx, self.ny), dtype=bool)

        # 地圖四邊牆壁
        wall_inflate = self.boundary_margin
        omap[XX <= self.x_min + wall_inflate] = True
        omap[XX >= self.x_max - wall_inflate] = True
        omap[YY <= self.y_min + wall_inflate] = True
        omap[YY >= self.y_max - wall_inflate] = True # 四周牆壁不給走

        # 圓柱障礙物（含左牆模擬障礙物）
        for obs in obstacles:
            ox = float(obs['pos'][0])
            oy = float(obs['pos'][1])
            inflate = self.robot_radius
            r = float(obs.get('astar_radius', obs['radius'])) + inflate # 0.6 (cbf) + 0.15 代表A*規劃的路線必距離障礙物0.75m
            omap[(XX - ox) ** 2 + (YY - oy) ** 2 <= r ** 2] = True 
            # 圓的方程式 

        for rect in rect_obstacles:
            cx = float(rect["center"][0])
            cy = float(rect["center"][1])
            sx = float(rect["size"][0])
            sy = float(rect["size"][1])
            margin = float(rect.get("astar_margin", self.robot_radius))
            hx = 0.5 * sx + margin
            hy = 0.5 * sy + margin
            omap[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = True

        return omap

    def _w2g(self, x, y): # 把現實座標轉成格子編號
        """World → grid index。"""
        return (int(round((x - self.x_min) / self.res)),
                int(round((y - self.y_min) / self.res)))
 
    def _g2w(self, ix, iy): # A* 找到路徑後，用這個把 grid index 轉回 world frame 座標，才能給 Pure Pursuit 
        """Grid index → world center。"""
        return (self.x_min + ix * self.res, self.y_min + iy * self.res)

    def _is_free(self, ix, iy): # 能不能走的條件 1. 地圖範圍內 2.true of false
        return (0 <= ix < self.nx and 0 <= iy < self.ny
                and not self._omap[ix, iy])

    #     plan() 收到 world frame 座標
    #         ↓ _w2g
    # grid index（A* 在這裡搜尋）
    #   每次展開鄰居前 → _is_free 檢查能不能走
    #         ↓ 找到路徑後 _g2w
    # world frame 座標 → 回傳給 Pure Pursuit
    
    def plan(self, start, goal): # 給起點和終點，跑 A* 找出一條路徑，回傳 waypoints list
        """
        Parameters
        ----------
        start : (x, y) world frame
        goal  : (x, y) world frame

        Returns
        -------
        list[(x, y)] waypoints in world frame，若無解回傳 []。
        """
        sx, sy = self._w2g(*start)
        gx, gy = self._w2g(*goal)

        if not self._is_free(gx, gy):
            rospy.logwarn("[A*] Goal grid(%d,%d) is occupied → abort", gx, gy)
            return [] # 終點在障礙物裡，直接回傳空 list。不需要搜尋，一定找不到路
        
        if not self._is_free(sx, sy):
            rospy.logwarn("[A*] Start grid(%d,%d) is occupied → trying anyway", sx, sy)

        g_cost = {(sx, sy): 0.0} # g(n)
        came_from = {} # 記錄的是「每個節點從哪來」，從終點一路往回找：
        heap = [(math.hypot(sx - gx, sy - gy), 0.0, sx, sy)] 

        while heap:
            _, g, cx, cy = heapq.heappop(heap) 

            # 跳過已被更好路徑取代的節點 同一個節點可能被 push 進 heap 多次 下一個放進去openset的格子可能會影響之前的格子
            if g > g_cost.get((cx, cy), float('inf')) + 1e-9:
                continue

            if (cx, cy) == (gx, gy): # 是不是終點？
                # 重建路徑
                path = []
                node = (gx, gy)
                while node in came_from:
                    path.append(self._g2w(*node))
                    node = came_from[node]
                path.append(self._g2w(sx, sy))
                path.reverse()
                return path

            for dx, dy, step_cost in self._MOVES: # 不是終點的話 就繼續展開八個方向 障礙物跳過 找到更短路徑 就push進去heap
                nbx, nby = cx + dx, cy + dy
                if not self._is_free(nbx, nby):
                    continue
                if dx != 0 and dy != 0:
                    if not self._is_free(cx + dx, cy) or not self._is_free(cx, cy + dy):
                        continue
                new_g = g + step_cost
                if new_g < g_cost.get((nbx, nby), float('inf')):
                    g_cost[(nbx, nby)] = new_g
                    came_from[(nbx, nby)] = (cx, cy)
                    h = math.hypot(nbx - gx, nby - gy) # h(n)
                    heapq.heappush(heap, (new_g + h, new_g, nbx, nby))

        rospy.logwarn("[A*] No path found from (%.2f,%.2f) to (%.2f,%.2f)",
                      start[0], start[1], goal[0], goal[1])
        return []

    def segment_crosses_occupied(self, start, goal):
        """檢查 start→goal 直線是否穿過 A* occupied cells。"""
        sx, sy = self._w2g(*start)
        gx, gy = self._w2g(*goal)
        steps = max(abs(gx - sx), abs(gy - sy), 1)
        for k in range(steps + 1):
            t = float(k) / float(steps)
            ix = int(round(sx + (gx - sx) * t))
            iy = int(round(sy + (gy - sy) * t))
            if not self._is_free(ix, iy):
                return True
        return False

    def nearest_free(self, goal):
        gx, gy = self._w2g(*goal)
        if self._is_free(gx, gy):
            return goal
        for r in range(1, 20):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx+dx, gy+dy
                    if self._is_free(nx, ny):
                        free = self._g2w(nx, ny)
                        rospy.logwarn(
                            "[A*] Goal occupied → nearest free (%.2f, %.2f)",
                            free[0], free[1]
                        )
                        return free
        return goal


# ═══════════════════════════════════════════════════════════════
# Module B: PurePursuitController
# ═══════════════════════════════════════════════════════════════

class PurePursuitController: # 每個 cycle 拿到 waypoints list 和機器人當前狀態，輸出這個 cycle 應該走的速度。
    """
    全向（holonomic）Pure Pursuit，輸出 body frame (vx, vy, wz)。

    演算法:
      1. 在 waypoints 中找出距離 look_ahead 的前視點
      2. 計算指向前視點的 body frame 速度向量（v_cruise 大小）
      3. wz 比例控制追蹤路徑航向角
    """

    def __init__(self, look_ahead=0.8, v_cruise=0.3, kp_yaw=1.2, goal_tol=0.3):
        self.look_ahead = float(look_ahead)
        self.v_cruise   = float(v_cruise)
        self.kp_yaw     = float(kp_yaw)
        self.goal_tol   = float(goal_tol) # yaml參數

    def compute(self, state, waypoints): # 兩個input leader的robot state A*給的waypoint
        """
        Parameters
        ----------
        state     : RobotState
        waypoints : list[(x, y)] world frame

        Returns
        -------
        ((vx_body, vy_body), wz)
        """
        if not waypoints:
            return (0.0, 0.0), 0.0

        goal = np.array(waypoints[-1])
        pos  = state.pos

        if float(np.linalg.norm(pos - goal)) < self.goal_tol: # 計算當前位置到終點的直線距離。距離 < 0.3m 就認為到達，輸出零速度。
            return (0.0, 0.0), 0.0

        la_x, la_y = self._find_lookahead(pos, waypoints)
        dx = la_x - pos[0]
        dy = la_y - pos[1]
        dist = math.hypot(dx, dy) # 根據lookahead （前視點）算出從當前位置到前視點的方向向量 (dx, dy) 和距離 dist。

        if dist < 1e-3:
            return (0.0, 0.0), 0.0

        # 追蹤前視點方向的 wz
        desired_yaw = math.atan2(dy, dx)
        wz = self.kp_yaw * wrap_to_pi(desired_yaw - state.yaw)

        # World frame 速度（固定大小 v_cruise）→ body frame
        vx_w = self.v_cruise * dx / dist # dx / dist 和 dy / dist 是做正規化，讓長度變成 1 變成單位向量
        vy_w = self.v_cruise * dy / dist # 再乘上v_cruise ＝ 0.3 
        c, s  = math.cos(state.yaw), math.sin(state.yaw)
        vx_b  =  c * vx_w + s * vy_w
        vy_b  = -s * vx_w + c * vy_w # 把 world frame 速度轉成 body frame，就是乘上旋轉矩陣（轉置）
        # 旋轉矩陣 本身的定義是 body frame → world frame 故 v_world = R @ v_body
        return (vx_b, vy_b), wz

    def _find_lookahead(self, pos, waypoints):
        """找到路徑上距離 look_ahead 最近的前視點。"""
        pos = np.array(pos)

        # 先找最近的 waypoint
        dists = [float(np.linalg.norm(pos - np.array(wp))) for wp in waypoints]
        closest_idx = int(np.argmin(dists))

        # 從最近點往前找第一個 >= look_ahead 的 waypoint
        for i in range(closest_idx, len(waypoints)):
            if float(np.linalg.norm(pos - np.array(waypoints[i]))) >= self.look_ahead:
                return waypoints[i]

        return waypoints[-1]


# ═══════════════════════════════════════════════════════════════
# Module 3.5: LeaderCmdRelay
# ═══════════════════════════════════════════════════════════════

class LeaderCmdRelay:
    """
    訂閱 /<leader>/cmd_vel_raw，取得 leader 的 nominal 指令（鍵盤模式）。

    資料流:
        rostopic pub /dog1/cmd_vel_raw → LeaderCmdRelay → u_nom
        → CBF → /dog1/cmd_vel → MPC
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
        """Returns ((vx, vy), wz)"""
        return (self.vx, self.vy), self.wz


# ═══════════════════════════════════════════════════════════════
# Module C: LeaderNavigator
# ═══════════════════════════════════════════════════════════════

class LeaderNavigator:
    """
    Leader 指令來源管理器，支援兩種模式:
        KEYBOARD: 訂閱 /dog1/cmd_vel_raw（鍵盤直接操控）
        AUTO:     A* 規劃路徑 + Pure Pursuit 追蹤

    收到 /dog1/goal (geometry_msgs/PoseStamped) → 切換 AUTO 模式
    到達目標（goal_tol 內）→ 自動回到 KEYBOARD 模式

    執行緒安全：goal callback 與主 spin loop 透過 _lock 互斥。
    A* 計算在 lock 外執行，避免阻塞 ROS callback。
    """

    MODE_KEYBOARD = "KEYBOARD"
    MODE_AUTO     = "AUTO"

    def __init__(self, leader_name, astar, pursuer):
        self._relay   = LeaderCmdRelay(leader_name)
        self._astar   = astar
        self._pursuer = pursuer

        self._lock      = threading.Lock()
        self._mode      = self.MODE_KEYBOARD
        self._goal      = None    # (gx, gy) world frame，None 表示無目標
        self._waypoints = []      # list[(x, y)]，空表示需要重新規劃

        rospy.Subscriber(
            f"/{leader_name}/goal",
            PoseStamped,
            self._goal_cb,
            queue_size=1,
        )
        rospy.loginfo("[LeaderNavigator] ready. Publish to /%s/goal to enter AUTO mode.",
                      leader_name)

    def _goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        with self._lock:
            self._goal      = (gx, gy)
            self._waypoints = []           # 令下一個 cycle 重新規劃
            self._mode      = self.MODE_AUTO
        rospy.loginfo("[LeaderNavigator] New goal (%.2f, %.2f) → AUTO", gx, gy)

    @property
    def current_mode(self):
        with self._lock:
            return self._mode

    @property
    def has_goal(self):
        """AUTO 模式且仍有未到達 goal → True"""
        with self._lock:
            return self._mode == self.MODE_AUTO and self._goal is not None

    def force_replan(self, reason=""):
        """
        外部觸發：清空 waypoints，下個 cycle 從當前位置重跑 A*。
        L2 stuck recovery 用：被 CBF 推離原路徑時呼叫。
        """
        with self._lock:
            if self._mode == self.MODE_AUTO and self._goal is not None:
                self._waypoints = []
                rospy.logwarn("[LeaderNavigator] force_replan: %s", reason or "external trigger")
                return True
        return False

    def abort_to_keyboard(self, reason=""):
        """
        放棄當前 goal，回到 KEYBOARD 模式。
        L2 stuck recovery 達到 replan 上限時用：明確告訴系統「這個 goal 不可達，交還給使用者」。
        回到 KEYBOARD 後 leader nominal 改由 /dogN/cmd_vel_raw 決定，
        使用者可以鍵盤倒退脫困、或下新 goal 重試。
        """
        with self._lock:
            if self._mode == self.MODE_AUTO:
                self._mode = self.MODE_KEYBOARD
                self._goal = None
                self._waypoints = []
                rospy.logwarn("[LeaderNavigator] AUTO aborted (%s) → KEYBOARD. "
                              "Send a new /goal or use cmd_vel_raw to take over.",
                              reason or "external")
                return True
        return False

    def get_nominal(self, leader_state):
        """
        主控 loop 每個 cycle 呼叫。
        Returns ((vx_body, vy_body), wz)
        """
        with self._lock:
            mode      = self._mode
            goal      = self._goal
            waypoints = list(self._waypoints)   # shallow copy，避免持鎖過久

        # ── KEYBOARD 模式 ──
        if mode == self.MODE_KEYBOARD:
            return self._relay.get_nominal()

        # ── AUTO 模式 ──
        if goal is None:
            with self._lock:
                self._mode = self.MODE_KEYBOARD
            return self._relay.get_nominal()

        # 若 waypoints 為空，執行 A*（在 lock 外，避免阻塞 callback）
        if not waypoints:
            rospy.loginfo("[LeaderNavigator] Planning A* path to (%.2f, %.2f) ...", *goal)
            safe_start = self._astar.nearest_free(tuple(leader_state.pos))
            safe_goal = self._astar.nearest_free(goal)
            new_wps = self._astar.plan(safe_start, safe_goal)

            if not new_wps:
                rospy.logwarn("[LeaderNavigator] A* failed → KEYBOARD fallback")
                with self._lock:
                    self._mode = self.MODE_KEYBOARD
                return self._relay.get_nominal()

            with self._lock:
                if self._goal == goal:          # 規劃期間 goal 沒有改變
                    self._waypoints = new_wps
                    waypoints       = new_wps
                else:
                    return (0.0, 0.0), 0.0      # goal 改變了，下一 cycle 重算

            rospy.loginfo("[LeaderNavigator] Path ready: %d waypoints", len(waypoints))

        # 判斷是否到達目標
        dist = math.hypot(leader_state.x - goal[0], leader_state.y - goal[1])
        if dist < self._pursuer.goal_tol:
            rospy.loginfo("[LeaderNavigator] Goal reached (%.3fm) → KEYBOARD", dist)
            with self._lock:
                self._mode      = self.MODE_KEYBOARD
                self._goal      = None
                self._waypoints = []
            return (0.0, 0.0), 0.0

        return self._pursuer.compute(leader_state, waypoints)

        # 啟動 → KEYBOARD（靜止）
        #         ↓ 發 /dog1/goal
        # AUTO（A* 走）
        #   ├── 鍵盤指令 → 被忽略
        #   ├── 到達目標 → KEYBOARD（靜止）
        #   │               ↓ 再發 /dog1/goal
        #   │             AUTO（繼續走）
        #   └── A* 失敗  → KEYBOARD（靜止）


# ═══════════════════════════════════════════════════════════════
# Module C.5: FollowerPathTracker
# ═══════════════════════════════════════════════════════════════

class FollowerPathTracker:
    """
    Follower 的 formation target 會移動；當直線追蹤會穿牆時，用 A* + Pure
    Pursuit 先繞到 target 附近，再交回 NominalController 做精準收斂。
    """

    def __init__(self, astar, pursuer, nominal_controller,
                 replan_dist=0.35, replan_cycles=20,
                 direct_dist=0.8):
        self._astar = astar
        self._pursuer = pursuer
        self._nominal = nominal_controller
        self._replan_dist = float(replan_dist)
        self._replan_cycles = int(replan_cycles)
        self._direct_dist = float(direct_dist)
        self._tracks = {}

    def reset(self, name=None):
        if name is None:
            self._tracks.clear()
        else:
            self._tracks.pop(name, None)

    def compute(self, name, state, target):
        x_ref, y_ref, yaw_ref = target
        goal = (x_ref, y_ref)

        if not state.received:
            return 0.0, 0.0, 0.0

        dist_to_goal = math.hypot(state.x - x_ref, state.y - y_ref)
        direct_blocked = self._astar.segment_crosses_occupied(tuple(state.pos), goal)

        # 只要直線不穿牆，就維持原本 formation PID；A* 只負責牆隔開 target 的情況。
        if not direct_blocked:
            self.reset(name)
            return self._nominal.compute(state, x_ref, y_ref, yaw_ref)

        track = self._tracks.get(name)
        needs_replan = (
            track is None
            or not track["waypoints"]
            or math.hypot(track["goal"][0] - goal[0], track["goal"][1] - goal[1]) > self._replan_dist
            or track["age"] >= self._replan_cycles
        )

        if needs_replan:
            safe_start = self._astar.nearest_free(tuple(state.pos))
            safe_goal = self._astar.nearest_free(goal)
            waypoints = self._astar.plan(safe_start, safe_goal)
            if not waypoints:
                rospy.logwarn_throttle(
                    1.0,
                    "[FollowerPathTracker] %s A* failed, fallback to formation PID",
                    name,
                )
                self.reset(name)
                return self._nominal.compute(state, x_ref, y_ref, yaw_ref)

            track = {"goal": goal, "waypoints": waypoints, "age": 0}
            self._tracks[name] = track
            rospy.loginfo(
                "[FollowerPathTracker] %s path ready: %d waypoints → formation target (%.2f, %.2f)",
                name, len(waypoints), x_ref, y_ref,
            )

        track["age"] += 1
        (vx, vy), wz_path = self._pursuer.compute(state, track["waypoints"])

        return vx, vy, wz_path

# ═══════════════════════════════════════════════════════════════
# Module 4: CBFSafetyFilter
# ═══════════════════════════════════════════════════════════════

class CBFSafetyFilter: # 收到所有狗的 u_nominal，解一個 QP，輸出最接近 u_nominal 但同時滿足所有安全約束的 u_safe
    """
    CBF-QP 安全濾波器。所有狗都是決策變數。

    決策變數: u = [vx_dog1, vy_dog1, vx_dog2, vy_dog2, vx_dog3, vy_dog3]
    全部在 body frame，wz pass-through 不進 QP。
    """

    def __init__(self, gamma_robot, d_min, gamma_obs=1.0, gamma_wall=1.0,
                 slack_lambda=1e4, slack_warn_threshold=0.05,
                 lookahead_tau=0.15):
        self.gamma_robot = gamma_robot
        self.d_min       = d_min
        self.d_min_sq    = d_min ** 2
        self.gamma_obs   = gamma_obs
        self.gamma_wall  = gamma_wall
        self.obstacles   = []
        self.rect_obstacles = []
        self.walls       = []
        # ── L1: Slack 變數參數（軟化約束，避免 QP infeasible）──
        self.slack_lambda = float(slack_lambda)
        self.slack_warn_threshold = float(slack_warn_threshold)
        self.last_max_slack = 0.0
        # ── Predictive CBF: 用 (p + v·τ) 取代 p，補償 MPC tracking lag ──
        # 對 obstacle / wall 才用，robot-robot 不用（兩狗都會動，預測互相抵銷）
        # τ=0.15s ≈ MPC 的 latency；太大 → 過度保守、難進窄道；太小 → 撞牆風險回來
        self.lookahead_tau = float(lookahead_tau)

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        rospy.loginfo("[CBFSafetyFilter] %d obstacles loaded", len(obstacles))

    def set_rect_obstacles(self, rect_obstacles):
        self.rect_obstacles = rect_obstacles
        rospy.loginfo("[CBFSafetyFilter] %d rect obstacles loaded", len(rect_obstacles))

    def set_walls(self, walls):
        self.walls = walls
        rospy.loginfo("[CBFSafetyFilter] %d walls loaded", len(walls))

    def solve(self, all_dogs, states, u_nom_dict):
        n_dogs  = len(all_dogs)
        n_vars  = 2 * n_dogs
        dog_idx = {name: i for i, name in enumerate(all_dogs)}

        u_nom = np.zeros(n_vars)
        for name in all_dogs:
            i = dog_idx[name]
            u_nom[2 * i]     = u_nom_dict[name][0]
            u_nom[2 * i + 1] = u_nom_dict[name][1] # → u_nom = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] dog1 vx vy ～

        A_rows, b_rows = [], [] # 線性限制

        # ── Pairwise robot-robot CBF ── 對每一隻狗各建立一個constraint
        for ia in range(n_dogs):
            for ib in range(ia + 1, n_dogs):
                na, nb = all_dogs[ia], all_dogs[ib]
                sa, sb = states[na], states[nb]
                if not sa.received or not sb.received:
                    continue
                dp = sa.pos - sb.pos # 兩狗位子的向量差
                h  = float(dp @ dp) - self.d_min_sq # h = dp^2 -d_min^2 h>=0 在safe set 
                a_row = np.zeros(n_vars)
                col_a = 2 * dog_idx[na]
                a_row[col_a:col_a + 2] = 2.0 * dp @ rot2d(sa.yaw)
                col_b = 2 * dog_idx[nb]
                a_row[col_b:col_b + 2] = -2.0 * dp @ rot2d(sb.yaw)
                A_rows.append(a_row)
                b_rows.append(-self.gamma_robot * h) # CBF constraint 用Au<=b 表示

        # ── Obstacle CBF（所有狗）── 障礙物對狗
        # Predictive: 用 p_pred = p + v_world·τ 計算 h，補償 tracking lag
        # 等效於把障礙物 inflate 一個 v·τ 的「速度敏感緩衝」
        for obs in self.obstacles:
            p_obs = np.array(obs["pos"][:2])
            r_obs = float(obs["radius"])
            for name in all_dogs:
                s = states[name]
                if not s.received:
                    continue
                p_pred = s.pos + s.vel_world * self.lookahead_tau
                dp     = p_pred - p_obs
                h_obs  = float(dp @ dp) - r_obs ** 2
                a_row  = np.zeros(n_vars)
                col    = 2 * dog_idx[name]
                a_row[col:col + 2] = 2.0 * dp @ rot2d(s.yaw)
                A_rows.append(a_row)
                b_rows.append(-self.gamma_obs * h_obs)

        # ── Finite rectangle obstacle CBF（左牆這種有洞口的牆段）──
        for rect in self.rect_obstacles:
            center = np.array(rect["center"][:2], dtype=float)
            size = np.array(rect["size"][:2], dtype=float)
            d_safe = float(rect.get("d_safe", 0.35))
            for name in all_dogs:
                s = states[name]
                if not s.received:
                    continue
                p_pred = s.pos + s.vel_world * self.lookahead_tau
                p_closest = closest_point_on_aabb(p_pred, center, size)
                dp = p_pred - p_closest
                dist = float(np.linalg.norm(dp))
                if dist < 1e-4:
                    away = p_pred - center
                    if abs(away[0]) > abs(away[1]):
                        dp = np.array([math.copysign(1.0, away[0] or 1.0), 0.0])
                    else:
                        dp = np.array([0.0, math.copysign(1.0, away[1] or 1.0)])
                    dist = 1.0
                h_rect = dist ** 2 - d_safe ** 2
                a_row = np.zeros(n_vars)
                col = 2 * dog_idx[name]
                a_row[col:col + 2] = 2.0 * dp @ rot2d(s.yaw)
                A_rows.append(a_row)
                b_rows.append(-self.gamma_obs * h_rect)

        # ── Wall CBF（所有狗）── 牆壁對狗（同樣加入 lookahead 預測）
        for wall in self.walls:
            n_w   = np.array(wall["normal"][:2], dtype=float)
            p_w   = np.array(wall["point"][:2],  dtype=float)
            d_safe = float(wall.get("d_safe", 0.4))
            for name in all_dogs:
                s = states[name]
                if not s.received:
                    continue
                p_pred  = s.pos + s.vel_world * self.lookahead_tau
                h_wall  = float(n_w @ (p_pred - p_w)) - d_safe
                a_row   = np.zeros(n_vars)
                col     = 2 * dog_idx[name]
                a_row[col:col + 2] = n_w @ rot2d(s.yaw)
                A_rows.append(a_row)
                b_rows.append(-self.gamma_wall * h_wall)

        # ── QP（L1: Soft CBF with slack）──
        # 原問題: min ‖u-u_nom‖²  s.t. Au ≥ b
        # 軟化後: min ‖u-u_nom‖² + λ‖ε‖²  s.t. Au ≥ b - ε,  ε ≥ 0
        # λ 大 → ε 幾乎為 0（接近硬約束），λ 小 → 容許更多違反
        # 任何輸入下都保證可行：ε 取夠大就一定能滿足
        u = cp.Variable(n_vars) # 三隻狗六個速度項
        objective_terms = [cp.sum_squares(u - u_nom)]
        constraints = []
        eps = None
        if A_rows:
            A = np.array(A_rows)
            b = np.array(b_rows)
            eps = cp.Variable(A.shape[0], nonneg=True)
            constraints.append(A @ u >= b - eps)
            objective_terms.append(self.slack_lambda * cp.sum_squares(eps))
        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            rospy.logwarn("[CBF] Solver error, returning zero")
            self.last_max_slack = float('inf')
            return {name: (0.0, 0.0) for name in all_dogs}

        if prob.status not in ("optimal", "optimal_inaccurate"):
            # 加 slack 後幾乎不可能走到這裡；若仍然 infeasible 表示數值問題
            rospy.logwarn("[CBF] QP status=%s (with slack!), returning zero", prob.status)
            self.last_max_slack = float('inf')
            return {name: (0.0, 0.0) for name in all_dogs}

        # 記錄最大違反量；若超出 threshold，表示 leader 真的被擠壓（外層可據此 replan）
        if eps is not None and eps.value is not None:
            self.last_max_slack = float(np.max(eps.value))
            if self.last_max_slack > self.slack_warn_threshold:
                rospy.logwarn_throttle(
                    1.0,
                    "[CBF] slack active, max ε=%.3f (constraints being relaxed)",
                    self.last_max_slack,
                )
        else:
            self.last_max_slack = 0.0

        u_sol = u.value
        return {name: (float(u_sol[2 * dog_idx[name]]),
                       float(u_sol[2 * dog_idx[name] + 1]))
                for name in all_dogs}
        # 收到所有狗的 u_nominal
        #         ↓
        # 建立三類 constraint：
        #   robot-robot（3 個）
        #   obstacle（3狗 × 11障礙物 = 33 個）
        #   wall（3狗 × 3牆 = 9 個）
        #   共 45 個 constraint
        #         ↓
        # 解 QP：最小化 ‖u - u_nom‖²
        #         ↓
        # 回傳 u_safe

# ═══════════════════════════════════════════════════════════════
# Module 5: VelocityLimiter
# ═══════════════════════════════════════════════════════════════

class VelocityLimiter: # CBF 的 QP 只保證安全約束，不限制速度大小。理論上 QP 可能輸出很大的速度（例如緊急迴避時），這層確保速度不超過 MPC 能處理的範圍。
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
    def __init__(self, all_dog_names):
        self._pubs = {
            name: rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=1)
            for name in all_dog_names
        }

    def publish(self, name, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x  = vx
        cmd.linear.y  = vy
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

    Step 1:
        Leader:    LeaderNavigator (AUTO=A*+PP / KEYBOARD=cmd_vel_raw) → u_nom
        Followers: zero velocity
        All:       u_nom → CBFSafetyFilter → VelocityLimiter → CmdVelPublisher

    Step 2+ (followers_stationary=false):
        Followers re-enable PID formation tracking
    """

    def __init__(self):
        rospy.init_node("fleet_manager_cbf") 

        # ── 基本參數 ──
        self.leader_name    = rospy.get_param("~leader_name",   "dog1") # 誰是leader
        self.follower_names = rospy.get_param("~follower_names", ["dog2", "dog3"]) # 誰是follower
        self.all_dogs       = [self.leader_name] + self.follower_names # 所有狗
        self.rate_hz        = rospy.get_param("~rate", 20.0) # 執行頻率
        self.stop_without_leader = rospy.get_param("~stop_without_leader", True) # 沒有leader全部停止
        self.followers_stationary = rospy.get_param(
            "~followers_stationary",
            _CFG.get("followers_stationary", True),
        )
        # 接下來是 依序初始化所有模組 參數都從yaml檔讀
        # ── Module 1: StateCollector ──
        self.state_collector = StateCollector(self.all_dogs)

        # ── Module 2: FormationPlanner ──
        default_offsets = _CFG.get("offsets", {"dog2": [-1.0, 1.0], "dog3": [-1.0, -1.0]})
        offsets = rospy.get_param("~offsets", default_offsets)
        self.planner = FormationPlanner(offsets)

        # ── Module 3: NominalController (Step 2+) Module 3 要的參數 
        self.controller = NominalController(  
            kp_x=rospy.get_param("~kp_x", _CFG.get("kp_x", 0.8)),
            kp_y=rospy.get_param("~kp_y", _CFG.get("kp_y", 0.8)),
            kp_yaw=rospy.get_param("~kp_yaw", _CFG.get("kp_yaw", 1.0)),
            pos_tol=rospy.get_param(
                "~pos_tolerance", _CFG.get("pos_tolerance", 0.15)),
            yaw_tol=rospy.get_param(
                "~yaw_tolerance", _CFG.get("yaw_tolerance", 0.15)),
            align_yaw=rospy.get_param(
                "~follower_align_yaw", _CFG.get("follower_align_yaw", False)),
            align_yaw_while_moving=rospy.get_param(
                "~follower_align_yaw_while_moving",
                _CFG.get("follower_align_yaw_while_moving", True)),
            moving_yaw_min_dist=rospy.get_param(
                "~follower_yaw_move_dist", _CFG.get("follower_yaw_move_dist", 0.25)),
        )

        # ── Module A: AStarPlanner ──
        obstacles = rospy.get_param("~obstacles", _CFG.get("obstacles", []))
        rect_obstacles = rospy.get_param("~rect_obstacles", _CFG.get("rect_obstacles", []))
        self.astar = AStarPlanner(
            resolution=rospy.get_param(
                "~astar_resolution", _CFG.get("astar_resolution", 0.1)),
            robot_radius=rospy.get_param(
                "~astar_robot_radius", _CFG.get("astar_robot_radius", 0.25)),
            obstacles=obstacles,
            x_min=rospy.get_param("~map_x_min", _CFG.get("map_x_min", 0.0)),
            x_max=rospy.get_param("~map_x_max", _CFG.get("map_x_max", 10.0)),
            y_min=rospy.get_param("~map_y_min", _CFG.get("map_y_min", -5.0)),
            y_max=rospy.get_param("~map_y_max", _CFG.get("map_y_max",  5.0)),
            boundary_margin=rospy.get_param(
                "~astar_boundary_margin", _CFG.get("astar_boundary_margin", 0.45)),
            rect_obstacles=rect_obstacles,
        )

        # ── Module B: PurePursuitController ──
        self.pursuer = PurePursuitController(
            look_ahead=rospy.get_param("~pp_look_ahead", _CFG.get("pp_look_ahead", 0.8)),
            v_cruise=rospy.get_param(  "~pp_v_cruise",   _CFG.get("pp_v_cruise",   0.3)),
            kp_yaw=rospy.get_param(    "~pp_kp_yaw",     _CFG.get("pp_kp_yaw",     1.2)),
            goal_tol=rospy.get_param(  "~pp_goal_tol",   _CFG.get("pp_goal_tol",   0.3)),
        )

        # ── Module C: LeaderNavigator ──
        self.navigator = LeaderNavigator(self.leader_name, self.astar, self.pursuer)

        # ── Module C.5: FollowerPathTracker ──
        self.follower_tracker = FollowerPathTracker(
            self.astar,
            self.pursuer,
            self.controller,
            replan_dist=rospy.get_param(
                "~follower_replan_dist", _CFG.get("follower_replan_dist", 0.35)),
            replan_cycles=rospy.get_param(
                "~follower_replan_cycles", _CFG.get("follower_replan_cycles", 20)),
            direct_dist=rospy.get_param(
                "~follower_direct_dist", _CFG.get("follower_direct_dist", 0.8)),
        )

        # ── Module 4: CBFSafetyFilter ──
        self.cbf_enabled = rospy.get_param("~cbf_enabled", _CFG.get("cbf_enabled", True))
        self.cbf_filter  = CBFSafetyFilter(
            gamma_robot=rospy.get_param("~cbf_gamma",      _CFG.get("cbf_gamma",      2.0)),
            d_min=rospy.get_param(      "~cbf_d_min",      _CFG.get("cbf_d_min",      1.0)),
            gamma_obs=rospy.get_param(  "~cbf_gamma_obs",  _CFG.get("cbf_gamma_obs",  2.0)),
            gamma_wall=rospy.get_param( "~cbf_gamma_wall", _CFG.get("cbf_gamma_wall", 2.0)),
            slack_lambda=rospy.get_param(
                "~cbf_slack_lambda", _CFG.get("cbf_slack_lambda", 1e4)),
            slack_warn_threshold=rospy.get_param(
                "~cbf_slack_warn", _CFG.get("cbf_slack_warn", 0.05)),
            lookahead_tau=rospy.get_param(
                "~cbf_lookahead_tau", _CFG.get("cbf_lookahead_tau", 0.15)),
        )
        if obstacles:
            self.cbf_filter.set_obstacles(obstacles)
        if rect_obstacles:
            self.cbf_filter.set_rect_obstacles(rect_obstacles)
        walls = rospy.get_param("~walls", _CFG.get("walls", []))
        if walls:
            self.cbf_filter.set_walls(walls)

        # ── Module 5: VelocityLimiter ──
        self.limiter = VelocityLimiter(
            max_vx=rospy.get_param("~max_vx", _CFG.get("max_vx", 0.5)),
            max_vy=rospy.get_param("~max_vy", _CFG.get("max_vy", 0.3)),
            max_wz=rospy.get_param("~max_wz", _CFG.get("max_wz", 0.8)),
        )

        # ── Module 6: CmdVelPublisher ──
        self.cmd_pub = CmdVelPublisher(self.all_dogs)

        # ── L2 stuck detection 參數 ──
        # leader 速度低於 stuck_speed_threshold 連續 stuck_replan_cycles 次 → 觸發 A* replan
        # 觸發後 stuck_replan_cooldown 個 cycle 內不再偵測，避免抖動
        self.stuck_speed_threshold = float(rospy.get_param(
            "~stuck_speed_threshold", _CFG.get("stuck_speed_threshold", 0.05)))
        self.stuck_replan_cycles = int(rospy.get_param(
            "~stuck_replan_cycles", _CFG.get("stuck_replan_cycles", 10)))
        self.stuck_replan_cooldown = int(rospy.get_param(
            "~stuck_replan_cooldown", _CFG.get("stuck_replan_cooldown", 30)))
        self.stuck_max_replans = int(rospy.get_param(
            "~stuck_max_replans", _CFG.get("stuck_max_replans", 3)))
        self._stuck_counter = 0
        self._cooldown_counter = 0
        self._consec_replans = 0  # 同一 stuck 已連續 replan 幾次（達上限就放棄，避免無限重試）

        rospy.sleep(1.0)
        rospy.loginfo("=" * 65)
        rospy.loginfo("[FleetManagerCBF] READY  (Step 1: A* Navigation)")
        rospy.loginfo("  leader      = %s", self.leader_name)
        rospy.loginfo("  followers   = %s  (stationary=%s)",
                      self.follower_names, self.followers_stationary)
        rospy.loginfo("  navigator   = KEYBOARD by default")
        rospy.loginfo("    → rostopic pub /%s/goal geometry_msgs/PoseStamped "
                      "... to switch AUTO", self.leader_name)
        rospy.loginfo("    → or: rostopic pub /%s/cmd_vel_raw geometry_msgs/Twist "
                      "... for KEYBOARD", self.leader_name)
        rospy.loginfo("  A*          resolution=%.2fm, robot_radius=%.2fm",
                      self.astar.res, self.astar.robot_radius)
        rospy.loginfo("  PurePursuit look_ahead=%.1fm, v_cruise=%.2fm/s",
                      self.pursuer.look_ahead, self.pursuer.v_cruise)
        rospy.loginfo("  followers   A* replan_dist=%.2fm, direct_dist=%.2fm",
                      self.follower_tracker._replan_dist,
                      self.follower_tracker._direct_dist)
        rospy.loginfo("  CBF         = %s (gamma=%.2f, d_min=%.2f, slack λ=%.0f, τ=%.2fs)",
                      self.cbf_enabled, self.cbf_filter.gamma_robot, self.cbf_filter.d_min,
                      self.cbf_filter.slack_lambda, self.cbf_filter.lookahead_tau)
        rospy.loginfo("  obstacles   = %d,  walls = %d",
                      len(self.cbf_filter.obstacles), len(self.cbf_filter.walls))
        rospy.loginfo("  rect_obs    = %d", len(self.cbf_filter.rect_obstacles))
        rospy.loginfo("  stuck       = v<%.2fm/s for %d cycles → replan (max %dx, cooldown %d)",
                      self.stuck_speed_threshold, self.stuck_replan_cycles,
                      self.stuck_max_replans, self.stuck_replan_cooldown)
        rospy.loginfo("  rate        = %.0f Hz", self.rate_hz)
        rospy.loginfo("=" * 65)

    def spin(self): # 主迴圈，每個 cycle 做四件事 1. 等待leader 2. 取得 leader 的 nominal 速度  3. 取得所有狗的 nominal 速度 4. CBF過濾
        rate   = rospy.Rate(self.rate_hz)
        states = self.state_collector.states
        _last_logged_mode = [None]

        while not rospy.is_shutdown():
            # ── 前置檢查：等待 leader 狀態 ──
            if self.stop_without_leader and not states[self.leader_name].received:
                self.cmd_pub.publish_zero(self.all_dogs)
                rate.sleep()
                continue

            leader = states[self.leader_name]

            # ── Step 1: Leader nominal（AUTO or KEYBOARD）──
            leader_nom, leader_wz = self.navigator.get_nominal(leader)

            mode = self.navigator.current_mode
            if mode != _last_logged_mode[0]:
                rospy.loginfo("[FleetManagerCBF] Leader mode → %s", mode)
                _last_logged_mode[0] = mode

            # ── Step 2: Follower nominal ──
            u_nom  = {self.leader_name: leader_nom}
            wz_nom = {self.leader_name: leader_wz}

            for name in self.follower_names:
                if self.followers_stationary:
                    # Step 1: Follower 靜止；CBF 仍然啟動，確保安全
                    self.follower_tracker.reset(name)
                    u_nom[name]  = (0.0, 0.0)
                    wz_nom[name] = 0.0
                else:
                    # Step 2+: path-aware formation tracking
                    x_ref, y_ref, yaw_ref = self.planner.compute_target(leader, name)
                    vx, vy, wz = self.follower_tracker.compute(
                        name, states[name], (x_ref, y_ref, yaw_ref))
                    u_nom[name]  = (vx, vy)
                    wz_nom[name] = wz

            # ── Step 3: CBF safety filter（所有狗）──
            if self.cbf_enabled:
                u_safe = self.cbf_filter.solve(self.all_dogs, states, u_nom)
            else:
                u_safe = u_nom

            # ── Step 4: Velocity limiter + publish（所有狗）──
            for name in self.all_dogs:
                vx, vy = u_safe[name]
                wz     = wz_nom[name]
                vx, vy, wz = self.limiter.clamp(vx, vy, wz)
                self.cmd_pub.publish(name, vx, vy, wz)

            # ── L2: Stuck detection + replan ──
            # 條件：leader 在 AUTO 且 CBF 後速度幾乎為 0（被卡住），不在 cooldown
            self._update_stuck_state(u_safe[self.leader_name])

            rate.sleep()

    def _update_stuck_state(self, leader_u_safe):
        """
        每 cycle 呼叫。判斷 leader 是否被 CBF 卡死，必要時觸發 force_replan。

        狀態機：
            cooldown > 0       → 正在 cooldown，遞減後跳出
            未 stuck            → counter 歸零
            stuck 累積到上限    → force_replan，進入 cooldown，遞增 consec_replans
            consec_replans 超上限 → 放棄（避免無限重試），等使用者下新 goal 才重置
        """
        if not self.navigator.has_goal:
            self._stuck_counter = 0
            self._consec_replans = 0
            return

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        speed = math.hypot(leader_u_safe[0], leader_u_safe[1])
        if speed < self.stuck_speed_threshold:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
            # leader 動了 → 認定逃出 stuck，重置 replan 上限計數
            self._consec_replans = 0
            return

        if self._stuck_counter < self.stuck_replan_cycles:
            return

        if self._consec_replans >= self.stuck_max_replans:
            # 上限到了 → 真死局。放棄 AUTO，切回 KEYBOARD，讓使用者接管。
            self.navigator.abort_to_keyboard(
                "%d replans, leader still stuck" % self.stuck_max_replans)
            self._consec_replans = 0
            self._stuck_counter = 0
            self._cooldown_counter = 0
            return

        slack = self.cbf_filter.last_max_slack
        reason = "leader speed≈0 for %.2fs (max slack ε=%.3f)" % (
            self._stuck_counter / self.rate_hz, slack)
        if self.navigator.force_replan(reason):
            self._consec_replans += 1
            self._stuck_counter = 0
            self._cooldown_counter = self.stuck_replan_cooldown


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        manager = FleetManagerCBF()
        manager.spin()
    except rospy.ROSInterruptException:
        pass
