#!/usr/bin/env python3
"""
Formation_manager_unified_twoOrderCBF.py — 二階 CBF 上層 QP 多機編隊管理器

基於 formation_managerCBF_door.py 改寫：
  - 保留: StateCollector, AStarPlanner, PurePursuitController,
          LeaderNavigator, VelocityLimiter, CmdVelPublisher
  - 刪除: FormationPlanner, NominalController, FollowerPathTracker,
          DoorPassageManager
  - 新增: LaplacianFormation, FormationSwitcher
  - 升級: CBFSafetyFilter → TwoOrderCBFQPController
          (用 acceleration HOCBF 產生安全 cmd_vel)

架構:
    FleetManagerUQP
    ├── StateCollector           讀取 /dogN/ground_truth/state
    ├── AStarPlanner             A* 網格路徑規劃
    ├── PurePursuitController    Pure Pursuit 路徑追蹤
    ├── LeaderNavigator          AUTO goal / KEYBOARD 模式切換（virtual center）
    │   └── LeaderCmdRelay       訂閱 /formation/cmd_vel_raw 手動/搖桿模式
    ├── LaplacianFormation       隊形切換 offsets + Laplacian 診斷 cost
    ├── FormationSwitcher        自動偵測窄門 → 切換 L̂_des
    ├── Per-dog A* front-end     AUTO 時每隻狗各自規劃到 assigned final slot
    ├── TwoOrderCBFQPController  二階 CBF QP (acceleration → cmd_vel)
    ├── VelocityLimiter          限制 vx, vy, wz
    └── CmdVelPublisher          發布 /dogN/cmd_vel

資料流:
    StateCollector → positions
    LeaderNavigator → AUTO goal 或 manual u_ref_center
    LaplacianFormation → centroid-relative offsets
    FormationSwitcher → 可能切換 L̂_des
    AUTO: per-dog A* + Pure Pursuit → u_nom_i，加 QP Laplacian formation cost
    KEYBOARD/fallback: centroid-relative target tracking → u_nom_i
    TwoOrderCBFQPController → u_safe × 3
        decision: a_i = [ax_i, ay_i], u_next = u_measured + a_i dt
        objective: w_track·Σ_i‖a_i-a_nom_i‖² + w_formation·f̈_form
                 + w_accel·‖a‖²
        constraint: HOCBF pairwise + obstacle + rect + wall
    VelocityLimiter → CmdVelPublisher → /dogN/cmd_vel → OCS2 MPC

啟動: rosrun legged_controllers Formation_manager_unified_twoOrderCBF.py
"""

import heapq
import itertools
import math
import os
import threading
import yaml
import numpy as np
import cvxpy as cp
import rospy
from geometry_msgs.msg import Point, Twist, PoseStamped, PoseArray, Vector3
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Float32
from visualization_msgs.msg import Marker, MarkerArray

# ── 讀取 YAML config ──
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Cbf_params_twoOrderCBF.yaml"
)


def _load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


_CFG = _load_config(_CONFIG_PATH)


# ═══════════════════════════════════════════════════════════════
# 工具函式（不動）
# ═══════════════════════════════════════════════════════════════

def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot2d(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])


def closest_point_on_aabb(point, center, size):
    half = 0.5 * np.array(size[:2], dtype=float)
    center = np.array(center[:2], dtype=float)
    return np.minimum(np.maximum(point, center - half), center + half)


# ═══════════════════════════════════════════════════════════════
# Module 1: StateCollector（不動）
# ═══════════════════════════════════════════════════════════════

class RobotState:
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
        v_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        v_world = R @ v_body
        s.vx_world = v_world[0]
        s.vy_world = v_world[1]
        s.received = True

    def all_received(self, names):
        return all(self.states[n].received for n in names)


# ═══════════════════════════════════════════════════════════════
# Module A: AStarPlanner（不動）
# ═══════════════════════════════════════════════════════════════

class AStarPlanner:
    """Grid-based A* 路徑規劃器，障礙地圖在 __init__ 時一次性建立。"""

    _MOVES = [
        (1,  0, 1.0),          (-1,  0, 1.0),
        (0,  1, 1.0),          ( 0, -1, 1.0),
        (1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def __init__(self, resolution, robot_radius, obstacles,
                 x_min=0.0, x_max=10.0, y_min=-5.0, y_max=5.0,
                 boundary_margin=0.45, rect_obstacles=None,
                 forbidden_zones=None):
        self.res          = float(resolution)
        self.robot_radius = float(robot_radius)
        self.boundary_margin = float(boundary_margin)
        self.obstacles = obstacles or []
        self.rect_obstacles = rect_obstacles or []
        self.forbidden_zones = forbidden_zones or []
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.nx = int(round((self.x_max - self.x_min) / self.res)) + 1
        self.ny = int(round((self.y_max - self.y_min) / self.res)) + 1
        self._block_narrow_wall_gaps = False
        self._wall_gap_min_width = 1.0
        self._wall_gap_lateral_margin = 0.15
        self._wall_gap_obstacle_indices = None
        self._omap = self._build_map(self.robot_radius)
        self._map_cache = {round(self.robot_radius, 3): self._omap}
        self._planning_margin = self.robot_radius
        self._planning_label = "base"
        self._fallback_to_base_map = True
        self._goal_min_obstacle_clearance = 0.20
        self._goal_min_wall_clearance = 0.65
        rospy.loginfo(
            "[AStarPlanner] grid %dx%d (%.1fm×%.1fm, res=%.2fm), "
            "free=%d, obstacles=%d, rects=%d",
            self.nx, self.ny,
            self.x_max - self.x_min, self.y_max - self.y_min,
            self.res, int(np.sum(~self._omap)), len(self.obstacles),
            len(self.rect_obstacles),
        )

    def set_planning_margin(self, margin, label="formation", fallback_to_base=True):
        self._planning_margin = max(0.0, float(margin))
        self._planning_label = label or "formation"
        self._fallback_to_base_map = bool(fallback_to_base)

    def set_goal_candidate_clearance(self, min_obstacle_clearance=0.20,
                                     min_wall_clearance=0.65):
        self._goal_min_obstacle_clearance = max(
            0.0, float(min_obstacle_clearance))
        self._goal_min_wall_clearance = max(0.0, float(min_wall_clearance))

    def set_narrow_wall_gap_blocking(self, enabled, min_width=1.0,
                                     lateral_margin=0.15,
                                     obstacle_indices=None):
        self._block_narrow_wall_gaps = bool(enabled)
        self._wall_gap_min_width = max(0.0, float(min_width))
        self._wall_gap_lateral_margin = max(0.0, float(lateral_margin))
        if obstacle_indices is None:
            self._wall_gap_obstacle_indices = None
        else:
            self._wall_gap_obstacle_indices = set(int(idx) for idx in obstacle_indices)
        self._omap = self._build_map(self.robot_radius)
        self._map_cache = {round(self.robot_radius, 3): self._omap}
        rospy.loginfo(
            "[AStarPlanner] narrow wall gap blocking=%s, width=%.2fm, lateral=%.2fm, obstacles=%s, free=%d",
            self._block_narrow_wall_gaps,
            self._wall_gap_min_width,
            self._wall_gap_lateral_margin,
            "all" if self._wall_gap_obstacle_indices is None
            else sorted(self._wall_gap_obstacle_indices),
            int(np.sum(~self._omap)),
        )

    def _active_map(self):
        return self._map_for_margin(self._planning_margin)

    def _map_for_margin(self, margin):
        key = round(max(0.0, float(margin)), 3)
        if key not in self._map_cache:
            self._map_cache[key] = self._build_map(key)
            rospy.loginfo(
                "[AStarPlanner] planning map '%s' margin=%.2fm, free=%d",
                self._planning_label, key, int(np.sum(~self._map_cache[key]))
            )
        return self._map_cache[key]

    def _build_map(self, inflate):
        xs = np.arange(self.nx) * self.res + self.x_min
        ys = np.arange(self.ny) * self.res + self.y_min
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        inflate = float(inflate)
        omap = np.zeros((self.nx, self.ny), dtype=bool)
        wall_inflate = max(self.boundary_margin, inflate)
        omap[XX <= self.x_min + wall_inflate] = True
        omap[XX >= self.x_max - wall_inflate] = True
        omap[YY <= self.y_min + wall_inflate] = True
        omap[YY >= self.y_max - wall_inflate] = True
        for obs in self.obstacles:
            ox, oy = float(obs['pos'][0]), float(obs['pos'][1])
            r = float(obs.get('astar_radius', obs['radius'])) + inflate
            omap[(XX - ox) ** 2 + (YY - oy) ** 2 <= r ** 2] = True
        for rect in self.rect_obstacles:
            cx, cy = float(rect["center"][0]), float(rect["center"][1])
            sx, sy = float(rect["size"][0]), float(rect["size"][1])
            margin = max(float(rect.get("astar_margin", self.robot_radius)), inflate)
            hx, hy = 0.5 * sx + margin, 0.5 * sy + margin
            omap[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = True
        for zone in self.forbidden_zones:
            cx, cy = float(zone["center"][0]), float(zone["center"][1])
            sx, sy = float(zone["size"][0]), float(zone["size"][1])
            margin = float(zone.get("margin", 0.0))
            hx, hy = 0.5 * sx + margin, 0.5 * sy + margin
            omap[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = True
        if self._block_narrow_wall_gaps:
            self._apply_narrow_outer_wall_gap_blockers(
                omap, XX, YY, inflate, wall_inflate)
        return omap

    def _apply_narrow_outer_wall_gap_blockers(self, omap, XX, YY, inflate, wall_inflate):
        min_width = self._wall_gap_min_width
        if min_width <= 0.0:
            return
        x_right = self.x_max - wall_inflate
        y_top = self.y_max - wall_inflate
        y_bottom = self.y_min + wall_inflate

        for obs_idx, obs in enumerate(self.obstacles):
            if (self._wall_gap_obstacle_indices is not None
                    and obs_idx not in self._wall_gap_obstacle_indices):
                continue
            ox, oy = float(obs["pos"][0]), float(obs["pos"][1])
            r = float(obs.get("astar_radius", obs["radius"])) + inflate
            lateral = r + self._wall_gap_lateral_margin

            gap_right = x_right - (ox + r)
            if 0.0 <= gap_right < min_width:
                omap[(XX >= ox + r) & (XX <= x_right)
                     & (np.abs(YY - oy) <= lateral)] = True
                rospy.loginfo_throttle(
                    5.0,
                    "[A*] block narrow gap obs%d-right_wall width=%.2fm",
                    obs_idx, gap_right,
                )

            gap_top = y_top - (oy + r)
            if 0.0 <= gap_top < min_width:
                omap[(YY >= oy + r) & (YY <= y_top)
                     & (np.abs(XX - ox) <= lateral)] = True
                rospy.loginfo_throttle(
                    5.0,
                    "[A*] block narrow gap obs%d-top_wall width=%.2fm",
                    obs_idx, gap_top,
                )

            gap_bottom = (oy - r) - y_bottom
            if 0.0 <= gap_bottom < min_width:
                omap[(YY >= y_bottom) & (YY <= oy - r)
                     & (np.abs(XX - ox) <= lateral)] = True
                rospy.loginfo_throttle(
                    5.0,
                    "[A*] block narrow gap obs%d-bottom_wall width=%.2fm",
                    obs_idx, gap_bottom,
                )

    def _w2g(self, x, y):
        return (int(round((x - self.x_min) / self.res)),
                int(round((y - self.y_min) / self.res)))

    def _g2w(self, ix, iy):
        return (self.x_min + ix * self.res, self.y_min + iy * self.res)

    def _is_free(self, ix, iy, omap=None):
        if omap is None:
            omap = self._active_map()
        return (0 <= ix < self.nx and 0 <= iy < self.ny
                and not omap[ix, iy])

    def plan(self, start, goal):
        omap = self._active_map()
        path = self._plan_on_map(start, goal, omap)
        if path:
            return path
        base_map = self._map_for_margin(self.robot_radius)
        if (self._fallback_to_base_map
                and omap is not base_map
                and abs(self._planning_margin - self.robot_radius) > 1e-6):
            rospy.logwarn(
                "[A*] No path with %s margin %.2fm → fallback to base margin %.2fm",
                self._planning_label, self._planning_margin, self.robot_radius)
            path = self._plan_on_map(start, goal, base_map)
            if path:
                return path
        rospy.logwarn("[A*] No path found from (%.2f,%.2f) to (%.2f,%.2f)",
                      start[0], start[1], goal[0], goal[1])
        return []

    def _plan_on_map(self, start, goal, omap):
        sx, sy = self._w2g(*start)
        gx, gy = self._w2g(*goal)
        if not self._is_free(gx, gy, omap):
            return []
        if not self._is_free(sx, sy, omap):
            rospy.logwarn("[A*] Start grid(%d,%d) is occupied → trying anyway", sx, sy)
        g_cost = {(sx, sy): 0.0}
        came_from = {}
        heap = [(math.hypot(sx - gx, sy - gy), 0.0, sx, sy)]
        while heap:
            _, g, cx, cy = heapq.heappop(heap)
            if g > g_cost.get((cx, cy), float('inf')) + 1e-9:
                continue
            if (cx, cy) == (gx, gy):
                path = []
                node = (gx, gy)
                while node in came_from:
                    path.append(self._g2w(*node))
                    node = came_from[node]
                path.append(self._g2w(sx, sy))
                path.reverse()
                return path
            for dx, dy, step_cost in self._MOVES:
                nbx, nby = cx + dx, cy + dy
                if not self._is_free(nbx, nby, omap):
                    continue
                if dx != 0 and dy != 0:
                    if (not self._is_free(cx + dx, cy, omap)
                            or not self._is_free(cx, cy + dy, omap)):
                        continue
                new_g = g + step_cost
                if new_g < g_cost.get((nbx, nby), float('inf')):
                    g_cost[(nbx, nby)] = new_g
                    came_from[(nbx, nby)] = (cx, cy)
                    h = math.hypot(nbx - gx, nby - gy)
                    heapq.heappush(heap, (new_g + h, new_g, nbx, nby))
        return []

    def segment_crosses_occupied(self, start, goal):
        omap = self._active_map()
        sx, sy = self._w2g(*start)
        gx, gy = self._w2g(*goal)
        steps = max(abs(gx - sx), abs(gy - sy), 1)
        for k in range(steps + 1):
            t = float(k) / float(steps)
            ix = int(round(sx + (gx - sx) * t))
            iy = int(round(sy + (gy - sy) * t))
            if not self._is_free(ix, iy, omap):
                return True
        return False

    def _nearest_free_candidates(self, goal, max_dist=None):
        omap = self._active_map()
        goal = tuple(goal)
        gx, gy = self._w2g(*goal)
        max_r = 20
        if max_dist is not None:
            max_r = max(1, int(math.ceil(float(max_dist) / self.res)))

        candidates = []
        seen = set()

        def add_grid(ix, iy):
            if (ix, iy) in seen or not self._is_free(ix, iy, omap):
                return
            wx, wy = self._g2w(ix, iy)
            dist = math.hypot(wx - goal[0], wy - goal[1])
            if max_dist is not None and dist > float(max_dist) + 1e-9:
                return
            seen.add((ix, iy))
            candidates.append((dist, (wx, wy)))

        add_grid(gx, gy)

        # Prefer the radial escape direction when the requested goal lies
        # inside an inflated obstacle. This keeps goals near C/D on the same
        # side of the obstacle instead of snapping to a random free grid near
        # the wall gap.
        inflate = max(0.0, float(self._planning_margin))
        for obs in self.obstacles:
            center = np.array(obs["pos"][:2], dtype=float)
            r = float(obs.get("astar_radius", obs["radius"])) + inflate
            delta = np.array(goal, dtype=float) - center
            dist = float(np.linalg.norm(delta))
            if dist > r + self.res:
                continue
            if dist < 1e-6:
                delta = np.array([1.0, 0.0])
                dist = 1.0
            projected = center + delta / dist * (r + self.res)
            pix, piy = self._w2g(float(projected[0]), float(projected[1]))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    add_grid(pix + dx, piy + dy)

        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                add_grid(gx + dx, gy - r)
                add_grid(gx + dx, gy + r)
            for dy in range(-r + 1, r):
                add_grid(gx - r, gy + dy)
                add_grid(gx + r, gy + dy)

        candidates.sort(key=lambda item: item[0])
        return [point for _, point in candidates]

    def nearest_free(self, goal, max_dist=None, label="Goal"):
        candidates = self._nearest_free_candidates(goal, max_dist=max_dist)
        if not candidates:
            if max_dist is not None:
                rospy.logwarn("[A*] %s occupied; no free cell within %.2fm",
                              label, max_dist)
                return None
            return goal
        free = candidates[0]
        if math.hypot(free[0] - goal[0], free[1] - goal[1]) > 1e-6:
            rospy.logwarn("[A*] %s occupied (%s margin %.2fm) → nearest free (%.2f, %.2f)",
                          label, self._planning_label, self._planning_margin,
                          free[0], free[1])
        return free

    def find_reachable_goal(self, start, goal, max_dist=None):
        omap = self._active_map()
        gx, gy = self._w2g(*goal)
        goal_is_free = self._is_free(gx, gy, omap)
        candidates = self._nearest_free_candidates(goal, max_dist=max_dist)
        if not candidates:
            if max_dist is not None:
                rospy.logwarn("[A*] Goal occupied; no reachable candidate within %.2fm", max_dist)
            return None, []
        if not goal_is_free:
            safe_candidates = [
                candidate for candidate in candidates
                if self._goal_candidate_has_clearance(candidate)
            ]
            if safe_candidates:
                candidates = safe_candidates
            else:
                rospy.logwarn(
                    "[A*] Goal occupied; no formation-safe free candidate near requested goal")
                return None, []
        for candidate in candidates:
            path = self._plan_on_map(start, candidate, omap)
            if path:
                adjust_dist = math.hypot(candidate[0] - goal[0],
                                         candidate[1] - goal[1])
                if adjust_dist > 1e-6:
                    rospy.logwarn(
                        "[A*] Goal adjusted to reachable free point (%.2f, %.2f), Δ=%.2fm",
                        candidate[0], candidate[1], adjust_dist)
                return candidate, path
        if max_dist is not None:
            rospy.logwarn("[A*] No reachable free goal within %.2fm of requested goal", max_dist)
        else:
            rospy.logwarn("[A*] No reachable free goal near requested goal")
        return None, []

    def _goal_candidate_has_clearance(self, point):
        point = np.array(point, dtype=float)
        inflate = max(0.0, float(self._planning_margin))
        for obs in self.obstacles:
            center = np.array(obs["pos"][:2], dtype=float)
            r = float(obs.get("astar_radius", obs["radius"])) + inflate
            clearance = float(np.linalg.norm(point - center)) - r
            if clearance < self._goal_min_obstacle_clearance:
                return False

        wall_inflate = max(self.boundary_margin, inflate)
        clearances = [
            point[0] - (self.x_min + wall_inflate),
            (self.x_max - wall_inflate) - point[0],
            point[1] - (self.y_min + wall_inflate),
            (self.y_max - wall_inflate) - point[1],
        ]
        return min(clearances) >= self._goal_min_wall_clearance


# ═══════════════════════════════════════════════════════════════
# Module B: PurePursuitController（不動）
# ═══════════════════════════════════════════════════════════════

class PurePursuitController:
    """全向 Pure Pursuit，輸出 body frame (vx, vy, wz)。"""

    def __init__(self, look_ahead=0.8, v_cruise=0.3, kp_yaw=1.2,
                 goal_tol=0.3, astar=None):
        self.look_ahead = float(look_ahead)
        self.v_cruise   = float(v_cruise)
        self.kp_yaw     = float(kp_yaw)
        self.goal_tol   = float(goal_tol)
        self._astar     = astar
        self._progress_idx = 0
        self._path_signature = None

    def compute(self, state, waypoints):
        if not waypoints:
            return (0.0, 0.0), 0.0
        goal = np.array(waypoints[-1])
        pos  = state.pos
        if float(np.linalg.norm(pos - goal)) < self.goal_tol:
            return (0.0, 0.0), 0.0
        la_x, la_y = self._find_lookahead(pos, waypoints)
        dx, dy = la_x - pos[0], la_y - pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return (0.0, 0.0), 0.0
        desired_yaw = math.atan2(dy, dx)
        wz = self.kp_yaw * wrap_to_pi(desired_yaw - state.yaw)
        vx_w = self.v_cruise * dx / dist
        vy_w = self.v_cruise * dy / dist
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        vx_b =  c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w
        return (vx_b, vy_b), wz

    def _find_lookahead(self, pos, waypoints):
        pos = np.array(pos)
        signature = self._make_path_signature(waypoints)
        if signature != self._path_signature:
            self._path_signature = signature
            self._progress_idx = 0

        search_end = min(len(waypoints), self._progress_idx + 40)
        if self._progress_idx >= len(waypoints):
            self._progress_idx = max(0, len(waypoints) - 1)
        local = waypoints[self._progress_idx:search_end]
        if local:
            local_dists = [
                float(np.linalg.norm(pos - np.array(wp, dtype=float)))
                for wp in local
            ]
            closest_idx = self._progress_idx + int(np.argmin(local_dists))
        else:
            closest_idx = self._progress_idx
        self._progress_idx = max(self._progress_idx, closest_idx)

        candidates = []
        arc_len = 0.0
        prev = pos
        for i in range(self._progress_idx, len(waypoints)):
            wp = np.array(waypoints[i], dtype=float)
            arc_len += float(np.linalg.norm(wp - prev))
            prev = wp
            if arc_len >= self.look_ahead or i == len(waypoints) - 1:
                candidates.append(tuple(wp))
                break
            candidates.append(tuple(wp))

        if not candidates:
            return waypoints[-1]

        for candidate in reversed(candidates):
            if self._line_of_sight_clear(pos, candidate):
                return candidate

        # If every carrot would cut through an occupied/forbidden cell,
        # keep making small monotonic progress along the A* polyline instead
        # of jumping to a later path segment and cutting the corner.
        min_step = max(0.15, min(0.35, 0.4 * self.look_ahead))
        for candidate in candidates:
            if float(np.linalg.norm(np.array(candidate) - pos)) >= min_step:
                return candidate
        next_idx = min(self._progress_idx + 1, len(waypoints) - 1)
        return waypoints[next_idx]

    def _make_path_signature(self, waypoints):
        if not waypoints:
            return None
        first = tuple(round(v, 3) for v in waypoints[0])
        last = tuple(round(v, 3) for v in waypoints[-1])
        return (len(waypoints), first, last)

    def _line_of_sight_clear(self, pos, waypoint):
        if self._astar is None:
            return True
        return not self._astar.segment_crosses_occupied(tuple(pos), waypoint)


# ═══════════════════════════════════════════════════════════════
# Module C: LeaderCmdRelay + LeaderNavigator（不動）
# ═══════════════════════════════════════════════════════════════

class LeaderCmdRelay:
    def __init__(self, cmd_topic):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.received = False
        self.stamp = rospy.Time(0)
        self._cmd_topic = cmd_topic
        rospy.Subscriber(self._cmd_topic, Twist, self._cb, queue_size=1)

    def _cb(self, msg):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z
        self.received = True
        self.stamp = rospy.Time.now()

    def get_nominal(self):
        return (self.vx, self.vy), self.wz

    def clear(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.received = False
        self.stamp = rospy.Time(0)

    def is_active(self, timeout, deadband):
        if not self.received:
            return False
        if timeout > 0.0 and (rospy.Time.now() - self.stamp).to_sec() > timeout:
            return False
        return (abs(self.vx) > deadband
                or abs(self.vy) > deadband
                or abs(self.wz) > deadband)


class LeaderNavigator:
    """Virtual-center 指令管理: KEYBOARD / AUTO (A*+PP)"""

    MODE_KEYBOARD = "KEYBOARD"
    MODE_AUTO     = "AUTO"

    def __init__(self, goal_topic, cmd_topic, astar, pursuer,
                 max_goal_adjust_dist=0.75):
        self._relay   = LeaderCmdRelay(cmd_topic)
        self._astar   = astar
        self._pursuer = pursuer
        self._max_goal_adjust_dist = float(max_goal_adjust_dist)
        self._lock      = threading.Lock()
        self._mode      = self.MODE_KEYBOARD
        self._goal      = None
        self._tracking_goal = None
        self._waypoints = []
        self._unreachable_hold = False
        self._unreachable_reason = ""
        self._goal_topic = goal_topic
        self._cmd_topic = cmd_topic
        rospy.Subscriber(self._goal_topic, PoseStamped, self._goal_cb, queue_size=1)
        rospy.loginfo("[LeaderNavigator] ready. Publish to %s to enter AUTO; "
                      "cmd raw topic is %s.",
                      self._goal_topic, self._cmd_topic)

    def _goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        with self._lock:
            self._goal = (gx, gy)
            self._tracking_goal = None
            self._waypoints = []
            self._unreachable_hold = False
            self._unreachable_reason = ""
            self._mode = self.MODE_AUTO
        rospy.loginfo("[LeaderNavigator] New goal (%.2f, %.2f) → AUTO", gx, gy)

    @property
    def current_mode(self):
        with self._lock:
            return self._mode

    @property
    def has_goal(self):
        with self._lock:
            return self._mode == self.MODE_AUTO and self._goal is not None

    @property
    def current_goal(self):
        with self._lock:
            return self._goal

    @property
    def unreachable_hold(self):
        with self._lock:
            return self._unreachable_hold

    @property
    def tracking_goal(self):
        with self._lock:
            return self._tracking_goal or self._goal

    def finish_auto_goal(self, reason=""):
        with self._lock:
            if self._mode != self.MODE_AUTO:
                return False
            self._mode = self.MODE_KEYBOARD
            self._goal = None
            self._tracking_goal = None
            self._waypoints = []
            self._unreachable_hold = False
            self._unreachable_reason = ""
        rospy.loginfo("[LeaderNavigator] Goal reached%s → KEYBOARD",
                      " (%s)" % reason if reason else "")
        return True

    def force_replan(self, reason=""):
        with self._lock:
            if self._mode == self.MODE_AUTO and self._goal is not None:
                self._waypoints = []
                self._tracking_goal = None
                rospy.logwarn("[LeaderNavigator] force_replan: %s",
                              reason or "external trigger")
                return True
        return False

    def force_via_waypoint(self, start, via, reason=""):
        with self._lock:
            if self._mode != self.MODE_AUTO or self._goal is None:
                return False
            goal = self._goal
        safe_start = self._astar.nearest_free(start)
        safe_via   = self._astar.nearest_free(via)
        safe_goal, second = self._astar.find_reachable_goal(
            safe_via, goal, max_dist=self._max_goal_adjust_dist)
        if safe_goal is None:
            rospy.logwarn("[LeaderNavigator] force_via_waypoint failed: no reachable safe goal")
            return False
        first  = self._astar.plan(safe_start, safe_via)
        if not first or not second:
            rospy.logwarn("[LeaderNavigator] force_via_waypoint failed via=(%.2f,%.2f)",
                          via[0], via[1])
            return False
        waypoints = first + second[1:]
        with self._lock:
            if self._mode == self.MODE_AUTO and self._goal == goal:
                self._tracking_goal = safe_goal
                self._waypoints = waypoints
                rospy.logwarn("[LeaderNavigator] recovery via (%.2f,%.2f): %s",
                              safe_via[0], safe_via[1], reason or "deadlock recovery")
                return True
        return False

    def abort_to_keyboard(self, reason=""):
        with self._lock:
            if self._mode == self.MODE_AUTO:
                self._mode = self.MODE_KEYBOARD
                self._goal = None
                self._tracking_goal = None
                self._waypoints = []
                rospy.logwarn("[LeaderNavigator] AUTO aborted (%s) → KEYBOARD",
                              reason or "external")
                return True
        return False

    def hold_unreachable_goal(self, reason=""):
        reason = reason or "unreachable goal"
        with self._lock:
            self._mode = self.MODE_KEYBOARD
            self._goal = None
            self._tracking_goal = None
            self._waypoints = []
            self._unreachable_hold = True
            self._unreachable_reason = reason
            self._relay.clear()
        rospy.logwarn("[LeaderNavigator] AUTO goal rejected (%s) → HOLD ZERO",
                      reason)

    def clear_unreachable_hold(self, reason=""):
        with self._lock:
            if not self._unreachable_hold:
                return False
            self._unreachable_hold = False
            self._unreachable_reason = ""
        rospy.loginfo("[LeaderNavigator] unreachable hold cleared%s",
                      " (%s)" % reason if reason else "")
        return True

    def manual_command_active(self, timeout=0.4, deadband=1e-3):
        return self._relay.is_active(float(timeout), float(deadband))

    def get_nominal(self, leader_state):
        with self._lock:
            mode      = self._mode
            goal      = self._goal
            tracking_goal = self._tracking_goal
            waypoints = list(self._waypoints)
        if mode == self.MODE_KEYBOARD:
            return self._relay.get_nominal()
        if goal is None:
            with self._lock:
                self._mode = self.MODE_KEYBOARD
            return self._relay.get_nominal()
        if not waypoints:
            rospy.loginfo("[LeaderNavigator] Planning A* to (%.2f,%.2f)...", *goal)
            safe_start = self._astar.nearest_free(tuple(leader_state.pos))
            safe_goal, new_wps = self._astar.find_reachable_goal(
                safe_start, goal, max_dist=self._max_goal_adjust_dist)
            if safe_goal is None:
                rospy.logwarn("[LeaderNavigator] A* failed: no reachable safe goal")
                self.hold_unreachable_goal(
                    "no reachable safe goal near requested target")
                return (0.0, 0.0), 0.0
            if not new_wps:
                rospy.logwarn("[LeaderNavigator] A* failed: empty path")
                self.hold_unreachable_goal("empty A* path")
                return (0.0, 0.0), 0.0
            with self._lock:
                if self._goal == goal:
                    self._tracking_goal = safe_goal
                    self._waypoints = new_wps
                    waypoints = new_wps
                    tracking_goal = safe_goal
                else:
                    return (0.0, 0.0), 0.0
            rospy.loginfo("[LeaderNavigator] Path ready: %d waypoints", len(waypoints))
        reached_goal = tracking_goal or goal
        dist = math.hypot(leader_state.x - reached_goal[0],
                          leader_state.y - reached_goal[1])
        if dist < self._pursuer.goal_tol:
            rospy.loginfo("[LeaderNavigator] Goal reached (%.3fm) → KEYBOARD", dist)
            with self._lock:
                self._mode = self.MODE_KEYBOARD
                self._goal = None
                self._tracking_goal = None
                self._waypoints = []
            return (0.0, 0.0), 0.0
        return self._pursuer.compute(leader_state, waypoints)


# ═══════════════════════════════════════════════════════════════
# Module NEW-1: LaplacianFormation（新增）
# ═══════════════════════════════════════════════════════════════

class LaplacianFormation:
    """
    Laplacian-based formation similarity metric (ZJU FAST-Lab, ICRA 2022).

    給定 N 隻狗的 2D 位置，計算 normalized Laplacian L̂ 和
    formation similarity cost f = ‖L̂ − L̂_des‖²_F，以及 ∂f/∂p_i。

    數學推導:
        w_ij = ‖p_i − p_j‖²                    (邊權重 = 距離平方)
        A_ij = w_ij (i≠j), 0 (i=j)             (adjacency matrix)
        D_ii = Σ_j w_ij                         (degree matrix)
        L = D − A                               (Laplacian)
        L̂ = D^{-1/2} L D^{-1/2}               (normalized Laplacian)
        L̂_ij = −w_ij / √(d_i · d_j)  (i≠j)    (展開形式)
        L̂_ii = 1                                (對角線恆為 1)
        f = ‖L̂ − L̂_des‖²_F                    (Frobenius 範數)

    梯度（chain rule, 論文 Eq. 5-9）:
        ∂f/∂p_i = Σ_j (∂f/∂w_ij) · (∂w_ij/∂p_i)
        ∂w_ij/∂p_i = 2(p_i − p_j)
        ∂f/∂w_ij = tr{(∂f/∂L̂)ᵀ · (∂L̂/∂w_ij)}
        ∂f/∂L̂ = 2(L̂ − L̂_des)
    """

    def __init__(self, formation_configs):
        """
        Parameters
        ----------
        formation_configs : dict
            {name: [(x1,y1), (x2,y2), (x3,y3)], ...}
            每種隊形的 centroid-relative offset，並同時計算 Laplacian 診斷 cost
        """
        self._L_hat_des_cache = {}
        self._offset_cache = {}
        for name, positions in formation_configs.items():
            offsets = [np.array(p, dtype=float) for p in positions]
            L_hat = self._compute_L_hat(np.array(offsets, dtype=float))
            self._L_hat_des_cache[name] = L_hat
            self._offset_cache[name] = offsets
            rospy.loginfo("[LaplacianFormation] '%s' → L̂_des registered "
                          "(off-diagonal: %s)", name,
                          np.array2string(L_hat[np.triu_indices(len(positions), k=1)],
                                          precision=3))

        self._current_name = None
        self.L_hat_des = None

    def set_formation(self, name):
        """切換目標隊形。只需替換 L̂_des 常數矩陣。"""
        if name not in self._L_hat_des_cache:
            rospy.logwarn("[LaplacianFormation] Unknown formation '%s'", name)
            return
        if name != self._current_name:
            self.L_hat_des = self._L_hat_des_cache[name]
            self._current_name = name
            rospy.loginfo("[LaplacianFormation] → '%s'", name)

    @property
    def current_formation(self):
        return self._current_name

    @property
    def current_offsets(self):
        if self._current_name not in self._offset_cache:
            return None
        return self._offset_cache[self._current_name]

    def compute(self, positions):
        """
        計算 Laplacian formation cost f 和每隻狗的 world frame 梯度。

        Parameters
        ----------
        positions : list[np.ndarray]  每隻狗的 2D world frame 位置

        Returns
        -------
        f : float                      formation similarity cost (≥0, 0=完美)
        grad_p : list[np.ndarray]      ∂f/∂p_i, 每隻狗一個 (2,) 向量
        """
        N = len(positions)
        pos = np.array(positions, dtype=float)  # (N, 2)

        if self.L_hat_des is None:
            return 0.0, [np.zeros(2) for _ in range(N)]

        # ── Step 1-4: 算 w, A, D, L̂ ──
        L_hat, w_matrix, d_vec = self._compute_L_hat_with_internals(pos)

        # ── Step 5: f = ‖L̂ − L̂_des‖²_F ──
        diff = L_hat - self.L_hat_des
        f = float(np.sum(diff ** 2))

        # ── Step 6-7: ∂f/∂p_i（解析梯度）──
        # ∂f/∂L̂ = 2(L̂ − L̂_des)
        df_dL = 2.0 * diff

        grad_p = []
        for i in range(N):
            grad_i = np.zeros(2)
            for j in range(N):
                if i == j:
                    continue
                w_ij = w_matrix[i, j]
                d_i, d_j = d_vec[i], d_vec[j]

                if d_i < 1e-12 or d_j < 1e-12:
                    # 兩狗幾乎重疊 → 退化，給零梯度避免 NaN
                    continue

                # ∂f/∂w_ij = tr{(∂f/∂L̂)ᵀ · (∂L̂/∂w_ij)}
                # ∂L̂/∂w_ij 的推導:
                #   L̂_ij = −w_ij / √(d_i·d_j)
                #   改變 w_ij 會同時影響 d_i, d_j（因為 d_i = Σ_k w_ik）
                #   所以 ∂L̂/∂w_ij 是一個 NxN 矩陣
                #
                # 完整推導太長，這裡用數值微分驗證過的解析公式。
                # 詳見 _df_dw() method。
                df_dw_ij = self._df_dw(i, j, w_matrix, d_vec, L_hat, df_dL, N)

                # ∂w_ij/∂p_i = 2(p_i − p_j)
                dw_dp = 2.0 * (pos[i] - pos[j])

                grad_i += df_dw_ij * dw_dp

            grad_p.append(grad_i)

        return f, grad_p

    @staticmethod
    def _compute_L_hat(pos):
        """只算 L̂，用於預計算 L̂_des。"""
        N = len(pos)
        w = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d2 = float(np.sum((pos[i] - pos[j]) ** 2))
                w[i, j] = w[j, i] = d2
        d = np.sum(w, axis=1)
        L_hat = np.eye(N)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i] > 1e-12 and d[j] > 1e-12:
                    val = -w[i, j] / math.sqrt(d[i] * d[j])
                    L_hat[i, j] = L_hat[j, i] = val
        return L_hat

    @staticmethod
    def _compute_L_hat_with_internals(pos):
        """算 L̂ 並回傳 w_matrix 和 d_vec 供梯度計算用。"""
        N = len(pos)
        w = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d2 = float(np.sum((pos[i] - pos[j]) ** 2))
                w[i, j] = w[j, i] = d2
        d = np.sum(w, axis=1)
        L_hat = np.eye(N)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i] > 1e-12 and d[j] > 1e-12:
                    val = -w[i, j] / math.sqrt(d[i] * d[j])
                    L_hat[i, j] = L_hat[j, i] = val
        return L_hat, w, d

    @staticmethod
    def _df_dw(i, j, w, d, L_hat, df_dL, N):
        """
        計算 ∂f/∂w_ij（標量）。

        改變 w_ij 會影響 L̂ 的多個元素（不只 L̂_ij），因為 d_i 和 d_j 也會變。
        具體來說，改變 w_ij 影響:
          - L̂_ij 本身
          - L̂_ik (k≠j): 因為 d_i 改變
          - L̂_jk (k≠i): 因為 d_j 改變

        公式（對稱性已處理，i<j or j<i 都行）:
        """
        d_i, d_j = d[i], d[j]
        if d_i < 1e-12 or d_j < 1e-12:
            return 0.0

        result = 0.0

        # ── 1. L̂_ij 對 w_ij 的直接影響 ──
        # L̂_ij = −w_ij / √(d_i · d_j)
        # ∂L̂_ij/∂w_ij = −1/√(d_i·d_j) + w_ij/(2·d_i·√(d_i·d_j)) + w_ij/(2·d_j·√(d_i·d_j))
        # 解釋: 第一項是分子的微分; 第二、三項是 d_i, d_j 增加導致分母變大
        sqrt_didj = math.sqrt(d_i * d_j)
        direct = -1.0 / sqrt_didj + w[i, j] / (2.0 * d_i * sqrt_didj) \
                                   + w[i, j] / (2.0 * d_j * sqrt_didj)

        # df_dL 是對稱的，L̂ 是對稱的，所以 (i,j) 和 (j,i) 各貢獻一次
        result += (df_dL[i, j] + df_dL[j, i]) * direct

        # ── 2. L̂_ik (k≠j) 對 w_ij 的影響（透過 d_i 改變）──
        # L̂_ik = −w_ik / √(d_i · d_k)
        # ∂L̂_ik/∂w_ij = w_ik / (2 · d_i · √(d_i · d_k))
        # （w_ij 增加 → d_i 增加 → √(d_i) 增加 → L̂_ik 的絕對值減小）
        for k in range(N):
            if k == i or k == j:
                continue
            d_k = d[k]
            if d_k < 1e-12:
                continue
            sqrt_didk = math.sqrt(d_i * d_k)
            effect_ik = w[i, k] / (2.0 * d_i * sqrt_didk)
            result += (df_dL[i, k] + df_dL[k, i]) * effect_ik

        # ── 3. L̂_jk (k≠i) 對 w_ij 的影響（透過 d_j 改變）──
        for k in range(N):
            if k == i or k == j:
                continue
            d_k = d[k]
            if d_k < 1e-12:
                continue
            sqrt_djdk = math.sqrt(d_j * d_k)
            effect_jk = w[j, k] / (2.0 * d_j * sqrt_djdk)
            result += (df_dL[j, k] + df_dL[k, j]) * effect_jk

        return result

    def numerical_gradient(self, positions, eps=1e-5):
        """數值微分驗證用。不在 real-time loop 呼叫。"""
        N = len(positions)
        grad = []
        for i in range(N):
            gi = np.zeros(2)
            for d in range(2):
                pos_plus = [p.copy() for p in positions]
                pos_minus = [p.copy() for p in positions]
                pos_plus[i][d] += eps
                pos_minus[i][d] -= eps
                f_plus, _ = self.compute(pos_plus)
                f_minus, _ = self.compute(pos_minus)
                gi[d] = (f_plus - f_minus) / (2.0 * eps)
            grad.append(gi)
        return grad


# ═══════════════════════════════════════════════════════════════
# Module NEW-2: FormationSwitcher（新增）
# ═══════════════════════════════════════════════════════════════

class FormationSwitcher:
    """
    自動偵測是否需要穿門 → 切換隊形。

    規則:
        1. leader 和 goal 在門的不同側 + 距離門 < trigger_dist → 切門口隊形
        2. 離門 > release_dist 或不需要穿門 → 切回 default
        3. 門口隊形仍用 centroid-relative offsets，可以是窄 V 或單列。
    """

    def __init__(self, laplacian, door_x=6.0, default_formation="V",
                 passage_formation="line", trigger_dist=2.0,
                 release_dist=2.0):
        self._laplacian = laplacian
        self._door_x = float(door_x)
        self._default = default_formation
        self._passage = passage_formation
        self._trigger_dist = float(trigger_dist)
        self._release_dist = float(release_dist)
        self._in_door_mode = False
        # 啟動時設為預設隊形
        self._laplacian.set_formation(self._default)

    @property
    def enabled(self):
        return True

    @property
    def door_x(self):
        return self._door_x

    def update(self, leader_state, goal, robot_states=None):
        """每 cycle 呼叫。根據 leader 與 goal 的位置決定是否切換隊形。"""
        x_samples = [float(leader_state.x)]
        if robot_states:
            for state in robot_states.values():
                if getattr(state, "received", False):
                    x_samples.append(float(state.x))
        nearest_door_dist = min(abs(x - self._door_x) for x in x_samples)

        if goal is None:
            if self._in_door_mode:
                if nearest_door_dist > self._release_dist:
                    self._in_door_mode = False
                    self._laplacian.set_formation(self._default)
                    return True
            return False

        leader_side = self._side(leader_state.x)
        goal_side = self._side(goal[0])
        needs_crossing = (goal_side != 0
                          and (leader_side == 0 or leader_side != goal_side))

        if needs_crossing and nearest_door_dist <= self._trigger_dist:
            if not self._in_door_mode:
                self._in_door_mode = True
                self._laplacian.set_formation(self._passage)
                rospy.loginfo("[FormationSwitcher] → '%s' (approaching door)",
                              self._passage)
                return True
        elif self._in_door_mode and nearest_door_dist > self._release_dist:
            self._in_door_mode = False
            self._laplacian.set_formation(self._default)
            rospy.loginfo("[FormationSwitcher] → '%s' (cleared door)",
                          self._default)
            return True

        return False

    def _side(self, x, deadband=0.15):
        if x < self._door_x - deadband:
            return -1
        if x > self._door_x + deadband:
            return 1
        return 0

    def recovery_waypoint(self, leader):
        """stuck recovery 用：給一個門口附近的通過點。"""
        if leader.x < self._door_x - 0.15:
            return (self._door_x - 0.75, 0.0)
        if leader.x < self._door_x + 0.55:
            return (self._door_x + 0.8, 0.0)
        return (max(self._door_x + 0.8, leader.x - 0.45), 0.0)


# ═══════════════════════════════════════════════════════════════
# Module 4': TwoOrderCBFQPController（relative-degree-2 CBF）
# ═══════════════════════════════════════════════════════════════

class TwoOrderCBFQPController:
    """
    上層二階 CBF QP：用 acceleration 作為決策變數，再積分成 cmd_vel。

    決策變數:
        a = [ax1, ay1, ax2, ay2, ax3, ay3]，body frame acceleration
        u_next = u_measured + a * dt

    CBF:
        h(p) >= 0
        psi1 = hdot + gamma1 * h
        psi1_dot + gamma2 * psi1 >= 0

    也就是:
        hddot + (gamma1 + gamma2) * hdot + gamma1 * gamma2 * h >= 0

    QP 公式:
        min  w_track · ||a - a_des||²
           + w_formation · grad_f^T (0.5 * R_i a_i * dt²)
           + w_accel · ||a||²

        s.t. A_hocbf · a >= b_hocbf
             等價於 hddot + (gamma1 + gamma2) * hdot
                  + gamma1 * gamma2 * h >= 0
             |a| <= a_max
             |u_next| <= u_max
    """

    def __init__(self, gamma_robot, d_min, gamma_obs=1.0, gamma_wall=1.0,
                 gamma_robot_1=None, gamma_robot_2=None,
                 gamma_obs_1=None, gamma_obs_2=None,
                 gamma_wall_1=None, gamma_wall_2=None,
                 lookahead_tau=0.15,
                 w_path=1.0, w_track=5.0, w_formation=0.0, w_reg=0.0,
                 w_accel=0.5,
                 max_vx=0.55, max_vy=0.35, max_ax=1.0, max_ay=1.0,
                 footprint_half_length=0.35,
                 footprint_half_width=0.20,
                 footprint_drift_margin=0.08,
                 prediction_enabled=False,
                 prediction_horizon=1,
                 prediction_dt=0.0,
                 w_smooth=0.2,
                 w_pred=20.0,
                 laplacian_ref=None,
                 K_accel=4.0,
                 Kd_accel=2.0,
                 emergency_brake_time=0.20,
                 slack_lambda=1e4,
                 slack_warn_threshold=0.05,
                 slack_enabled=True,
                 gamma_vel=1.0,
                 gamma_vel_pair=2.0):
        # ── CBF 參數 ──
        self.gamma_robot = float(gamma_robot)
        self.gamma_obs = float(gamma_obs)
        self.gamma_wall = float(gamma_wall)
        self.gamma_robot_1 = float(
            self.gamma_robot if gamma_robot_1 is None else gamma_robot_1)
        self.gamma_robot_2 = float(
            self.gamma_robot if gamma_robot_2 is None else gamma_robot_2)
        self.gamma_obs_1 = float(
            self.gamma_obs if gamma_obs_1 is None else gamma_obs_1)
        self.gamma_obs_2 = float(
            self.gamma_obs if gamma_obs_2 is None else gamma_obs_2)
        self.gamma_wall_1 = float(
            self.gamma_wall if gamma_wall_1 is None else gamma_wall_1)
        self.gamma_wall_2 = float(
            self.gamma_wall if gamma_wall_2 is None else gamma_wall_2)
        self.d_min = float(d_min)
        self.d_min_sq = self.d_min ** 2
        self.obstacles = []
        self.rect_obstacles = []
        self.walls = []
        self.last_cbf_status = "ok"
        self.lookahead_tau = float(lookahead_tau)

        # ── QP cost / bounds ──
        self.w_path = float(w_path)
        self.w_track = float(w_track)
        self.w_formation = float(w_formation)
        self.w_reg = float(w_reg)
        self.w_accel = float(w_accel)
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_ax = float(max_ax)
        self.max_ay = float(max_ay)
        self.footprint_half_length = max(0.0, float(footprint_half_length))
        self.footprint_half_width = max(0.0, float(footprint_half_width))
        self.footprint_drift_margin = max(0.0, float(footprint_drift_margin))
        self.prediction_requested = bool(prediction_enabled)
        # 二階 multi-step preview 已實作；prediction_enabled=true 時走 horizon QP。
        self.prediction_enabled = bool(prediction_enabled)
        self.prediction_horizon = max(1, int(prediction_horizon))
        self.prediction_dt = max(0.0, float(prediction_dt))
        self.w_smooth = float(w_smooth)
        self.w_pred = float(w_pred)
        self.laplacian_ref = laplacian_ref
        # Upper-layer nominal acceleration PD gains. K_accel is kept as the
        # position gain for backward compatibility with the old P-only version.
        self.K_accel = float(K_accel)
        self.Kd_accel = float(Kd_accel)
        self.emergency_brake_time = max(1e-3, float(emergency_brake_time))
        # ── HOCBF soft constraints (slack) ──
        # 高 λ：可行時 ε=0（等同硬 CBF）；只有原本 infeasible 才鬆，且最小違反。
        self.slack_lambda = float(slack_lambda)
        self.slack_warn_threshold = float(slack_warn_threshold)
        # False = 純硬 CBF(驗證用):h≥0 才是真保證;不可行→emergency brake。
        self.slack_enabled = bool(slack_enabled)
        # Deprecated: kept only for YAML/launch backward compatibility.
        # Pure second-order HOCBF no longer adds the extra velocity-layer CBF.
        self.gamma_vel = float(gamma_vel)
        self.gamma_vel_pair = float(gamma_vel_pair)
        self.last_max_slack = 0.0
        self.last_slack_by_kind = {}
        self.last_min_h_by_kind = {}
        self._last_a_sol = None
        self._last_A_seq = None  # multi-step 暖啟動:上一輪整段加速度序列
        self.last_accel_cmds = {}
        self.last_prediction_paths = {}
        self.last_reference_paths = {}
        self.last_prediction_dt = 0.0
        self._prediction_fallback_warned = False

    def _footprint_support_along(self, normal, yaw):
        normal = np.array(normal[:2], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return self.footprint_drift_margin
        normal = normal / norm
        body_x = np.array([math.cos(yaw), math.sin(yaw)])
        body_y = np.array([-math.sin(yaw), math.cos(yaw)])
        return (
            self.footprint_half_length * abs(float(normal @ body_x))
            + self.footprint_half_width * abs(float(normal @ body_y))
            + self.footprint_drift_margin
        )

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        rospy.loginfo("[TwoOrderCBF] %d obstacles loaded", len(obstacles))

    def set_rect_obstacles(self, rect_obstacles):
        self.rect_obstacles = rect_obstacles
        rospy.loginfo(
            "[TwoOrderCBF] %d rect obstacles loaded", len(rect_obstacles))

    def set_walls(self, walls):
        self.walls = walls
        rospy.loginfo("[TwoOrderCBF] %d walls loaded", len(walls))

    def reset_prediction(self, reason=""):
        self._last_a_sol = None
        self._last_A_seq = None
        self.last_accel_cmds = {}
        self.last_prediction_paths = {}
        self.last_reference_paths = {}
        self.last_prediction_dt = 0.0
        if reason:
            rospy.logdebug("[TwoOrderCBF] reset prediction: %s", reason)

    @staticmethod
    def _hocbf_rhs(h, hdot, hddot_free, gamma_1, gamma_2):
        return (
            -float(hddot_free)
            - (float(gamma_1) + float(gamma_2)) * float(hdot)
            - float(gamma_1) * float(gamma_2) * float(h)
        )

    def _accel_bounds_from_current_velocity(self, u_now, dt, n_dogs):
        lb = np.zeros(2 * n_dogs)
        ub = np.zeros(2 * n_dogs)
        for idx in range(n_dogs):
            x = 2 * idx
            y = x + 1
            lb[x] = max(-self.max_ax, (-self.max_vx - u_now[x]) / dt)
            ub[x] = min(self.max_ax, ( self.max_vx - u_now[x]) / dt)
            lb[y] = max(-self.max_ay, (-self.max_vy - u_now[y]) / dt)
            ub[y] = min(self.max_ay, ( self.max_vy - u_now[y]) / dt)
        return lb, ub

    def _log_hocbf_infeasible_diagnostics(self, A_rows, b_rows, row_meta,
                                          u_now, dt, n_dogs):
        if not A_rows:
            return

        A = np.array(A_rows, dtype=float)
        b = np.array(b_rows, dtype=float)
        lb, ub = self._accel_bounds_from_current_velocity(u_now, dt, n_dogs)

        lhs_max = np.sum(np.where(A >= 0.0, A * ub, A * lb), axis=1)
        gap = b - lhs_max
        impossible = gap > 1e-6

        by_kind = {}
        for k, meta in enumerate(row_meta):
            kind = meta.get("kind", "unknown")
            stats = by_kind.setdefault(
                kind,
                {"count": 0, "impossible": 0, "min_h": float("inf")},
            )
            stats["count"] += 1
            stats["impossible"] += int(impossible[k])
            stats["min_h"] = min(stats["min_h"], float(meta.get("h", 0.0)))

        summary = ", ".join(
            "%s %d/%d min_h=%.3f"
            % (kind, stats["impossible"], stats["count"], stats["min_h"])
            for kind, stats in sorted(by_kind.items())
        )

        worst_idx = int(np.argmax(gap))
        worst_meta = row_meta[worst_idx]
        rospy.logwarn_throttle(
            1.0,
            "[TwoOrderCBF] infeasible diag: %s | worst=%s:%s "
            "gap=%.3f b=%.3f lhs_max=%.3f h=%.3f hdot=%.3f",
            summary,
            worst_meta.get("kind", "unknown"),
            worst_meta.get("name", "?"),
            float(gap[worst_idx]),
            float(b[worst_idx]),
            float(lhs_max[worst_idx]),
            float(worst_meta.get("h", 0.0)),
            float(worst_meta.get("hdot", 0.0)),
        )

    def _current_body_velocity_vector(self, all_dogs, states):
        u_now = np.zeros(2 * len(all_dogs))
        for idx, name in enumerate(all_dogs):
            s = states[name]
            if not s.received:
                continue
            u_now[2 * idx:2 * idx + 2] = rot2d(s.yaw).T @ s.vel_world
        return self._clip_body_velocity_vector(u_now, len(all_dogs))

    def _record_slack(self, eps, row_meta):
        """記錄 HOCBF slack + 各類最小 h(實際安全距離),供 rqt_plot/debug。"""
        mh = {}
        for meta in row_meta:
            k = meta.get("kind", "?")
            if k.endswith("_v"):
                continue
            mh[k] = min(mh.get(k, 1e9), float(meta.get("h", 1e9)))
        self.last_min_h_by_kind = mh
        if eps is None or eps.value is None:
            self.last_max_slack = 0.0
            self.last_slack_by_kind = {}
            return
        eps_val = np.asarray(eps.value, dtype=float).ravel()
        if eps_val.size == 0:
            self.last_max_slack = 0.0
            self.last_slack_by_kind = {}
            return
        self.last_max_slack = float(np.max(eps_val))
        by_kind = {}
        for i, meta in enumerate(row_meta):
            if i >= eps_val.size:
                break
            kind = meta.get("kind", "?")
            by_kind[kind] = max(by_kind.get(kind, 0.0), float(eps_val[i]))
        self.last_slack_by_kind = by_kind
        if self.last_max_slack > self.slack_warn_threshold:
            rospy.logwarn_throttle(
                1.0,
                "[TwoOrderCBF] slack active: max eps=%.3f by kind=%s",
                self.last_max_slack,
                {k: round(v, 3) for k, v in by_kind.items()},
            )

    def _record_slack_array(self, eps_val, row_meta):
        """multi-step 版: eps_val 為 numpy 陣列 (或 None),其餘同 _record_slack。"""
        mh = {}
        for meta in row_meta:
            k = meta.get("kind", "?")
            if k.endswith("_v"):
                continue
            mh[k] = min(mh.get(k, 1e9), float(meta.get("h", 1e9)))
        self.last_min_h_by_kind = mh
        if eps_val is None or len(eps_val) == 0:
            self.last_max_slack = 0.0
            self.last_slack_by_kind = {}
            return
        eps_val = np.asarray(eps_val, dtype=float).ravel()
        self.last_max_slack = float(np.max(eps_val))
        by_kind = {}
        for i, meta in enumerate(row_meta):
            if i >= eps_val.size:
                break
            kind = meta.get("kind", "?")
            by_kind[kind] = max(by_kind.get(kind, 0.0), float(eps_val[i]))
        self.last_slack_by_kind = by_kind
        if self.last_max_slack > self.slack_warn_threshold:
            rospy.logwarn_throttle(
                1.0,
                "[TwoOrderCBF] slack active: max eps=%.3f by kind=%s",
                self.last_max_slack,
                {k: round(v, 3) for k, v in by_kind.items()},
            )

    def _emergency_brake_result(self, all_dogs, u_now, n_dogs):
        a_brake = -np.array(u_now, dtype=float) / self.emergency_brake_time
        for idx in range(n_dogs):
            x = 2 * idx
            y = x + 1
            a_brake[x] = max(-self.max_ax, min(self.max_ax, a_brake[x]))
            a_brake[y] = max(-self.max_ay, min(self.max_ay, a_brake[y]))
        self.last_accel_cmds = {
            name: a_brake[2 * idx:2 * idx + 2].copy()
            for idx, name in enumerate(all_dogs)
        }
        return {name: (0.0, 0.0) for name in all_dogs}

    def _store_one_step_prediction(self, all_dogs, states, u_now, a_body, dt):
        self.last_reference_paths = {}
        self.last_prediction_paths = {}
        self.last_prediction_dt = dt
        for idx, name in enumerate(all_dogs):
            s = states[name]
            if not s.received:
                continue
            sl = slice(2 * idx, 2 * idx + 2)
            R = rot2d(s.yaw)
            dp_world = R @ (u_now[sl] * dt + 0.5 * a_body[sl] * dt * dt)
            self.last_prediction_paths[name] = [
                s.pos.copy(),
                np.array(s.pos + dp_world, dtype=float),
            ]

    def solve(self, all_dogs, states, u_nominal, a_desired=None,
              formation_grad=None, cbf_enabled=True, dt=0.05, yaw_rates=None):
        if self.prediction_enabled and self.prediction_horizon > 1:
            return self._solve_multistep_preview(
                all_dogs, states, u_nominal,
                a_desired=a_desired,
                formation_grad=formation_grad,
                cbf_enabled=cbf_enabled,
                dt=dt,
                yaw_rates=yaw_rates,
            )
        return self._solve_single_step(
            all_dogs,
            states,
            u_nominal,
            a_desired=a_desired,
            formation_grad=formation_grad,
            cbf_enabled=cbf_enabled,
            dt=dt,
            yaw_rates=yaw_rates,
        )

    def _solve_single_step(self, all_dogs, states, u_nominal, a_desired=None,
                           formation_grad=None, cbf_enabled=True, dt=0.05,
                           yaw_rates=None):
        n_dogs = len(all_dogs)
        n_vars = 2 * n_dogs
        dog_idx = {name: i for i, name in enumerate(all_dogs)}
        dt = max(1e-6, float(dt))
        R_dogs = {name: rot2d(states[name].yaw) for name in all_dogs}
        u_now = self._current_body_velocity_vector(all_dogs, states)
        u_nominal = self._clip_body_velocity_vector(u_nominal, n_dogs)

        # Chain rule for body-frame acceleration:
        #   v^W = R(psi) v^B
        #   a^W = R(psi) [a^B + wz * [-vy^B, vx^B]^T]
        # The first term R a^B stays in a_row as the QP unknown. The yaw term
        # is known in the current cycle and is added to hddot_free.
        if yaw_rates is None:
            yaw_rates = {}
        yaw_acc_world = {}
        for idx, name in enumerate(all_dogs):
            s = states[name]
            sl = slice(2 * idx, 2 * idx + 2)
            v_body = u_now[sl]
            wz = float(yaw_rates.get(name, 0.0))
            yaw_term_body = wz * np.array([-v_body[1], v_body[0]], dtype=float)
            yaw_acc_world[name] = R_dogs[name] @ yaw_term_body

        A_rows, b_rows, row_meta = [], [], []

        if cbf_enabled:
            # ── Pairwise robot-robot HOCBF ──
            for ia in range(n_dogs):
                for ib in range(ia + 1, n_dogs):
                    na, nb = all_dogs[ia], all_dogs[ib]
                    sa, sb = states[na], states[nb]
                    if not sa.received or not sb.received:
                        continue
                    dp = sa.pos - sb.pos
                    dv = sa.vel_world - sb.vel_world
                    # 方向感知 dog-dog 間距：兩狗朝彼此方向的 footprint 投影相加。
                    # 並排(過門 V_narrow)→ 小(~0.85，門過得了)；面對/交錯(場中機動)
                    # → 大(身體+腿的真實間距)。base d_min 為下限。
                    d_min_eff = max(
                        self.d_min,
                        self._footprint_support_along(dp, sa.yaw)
                        + self._footprint_support_along(dp, sb.yaw))
                    h = float(dp @ dp) - d_min_eff ** 2
                    hdot = float(2.0 * dp @ dv)
                    hddot_free = float(
                        2.0 * dv @ dv
                        + 2.0 * dp @ (yaw_acc_world[na] - yaw_acc_world[nb]))
                    a_row = np.zeros(n_vars)
                    col_a = 2 * dog_idx[na]
                    a_row[col_a:col_a + 2] = 2.0 * dp @ R_dogs[na]
                    col_b = 2 * dog_idx[nb]
                    a_row[col_b:col_b + 2] = -2.0 * dp @ R_dogs[nb]
                    A_rows.append(a_row)
                    b_rows.append(self._hocbf_rhs(
                        h, hdot, hddot_free,
                        self.gamma_robot_1, self.gamma_robot_2))
                    row_meta.append({
                        "kind": "pair",
                        "name": "%s-%s" % (na, nb),
                        "h": h,
                        "hdot": hdot,
                    })

            # ── Circular obstacle HOCBF ──
            for obs in self.obstacles:
                p_obs = np.array(obs["pos"][:2])
                r_base = float(obs["radius"])
                r_phys = float(obs.get("physical_radius", 0.2))
                for name in all_dogs:
                    s = states[name]
                    if not s.received:
                        continue
                    p_pred = s.pos
                    dp = p_pred - p_obs
                    # CBF 半徑要含「狗朝障礙方向的 footprint 投影」，否則只擋中心、
                    # 狗鼻子(half_length 0.35 > 既有 buffer 0.30)會啃到障礙。
                    # max(...) 確保不低於原本 radius，只在正面接近時加餘量。
                    r_obs = max(
                        r_base,
                        r_phys + self._footprint_support_along(dp, s.yaw))
                    h_obs = float(dp @ dp) - r_obs ** 2
                    hdot = float(2.0 * dp @ s.vel_world)
                    hddot_free = float(
                        2.0 * s.vel_world @ s.vel_world
                        + 2.0 * dp @ yaw_acc_world[name])
                    a_row = np.zeros(n_vars)
                    col = 2 * dog_idx[name]
                    a_row[col:col + 2] = 2.0 * dp @ R_dogs[name]
                    A_rows.append(a_row)
                    b_rows.append(self._hocbf_rhs(
                        h_obs, hdot, hddot_free,
                        self.gamma_obs_1, self.gamma_obs_2))
                    row_meta.append({
                        "kind": "obs",
                        "name": str(obs.get("pos", "?")),
                        "h": h_obs,
                        "hdot": hdot,
                    })

            # ── Rect obstacle HOCBF（有洞口的牆段）──
            for rect in self.rect_obstacles:
                center = np.array(rect["center"][:2], dtype=float)
                size   = np.array(rect["size"][:2], dtype=float)
                d_safe_base = float(rect.get("d_safe", 0.35))
                for name in all_dogs:
                    s = states[name]
                    if not s.received:
                        continue
                    p_pred    = s.pos
                    p_closest = closest_point_on_aabb(p_pred, center, size)
                    dp   = p_pred - p_closest
                    dist = float(np.linalg.norm(dp))
                    if dist < 1e-4:
                        escape = rect.get("escape_dir", None)
                        if escape is not None:
                            dp = np.array(escape[:2], dtype=float)
                            norm = float(np.linalg.norm(dp))
                            if norm < 1e-9:
                                dp = np.zeros(2)
                            else:
                                dp = dp / norm
                        if float(np.linalg.norm(dp)) < 1e-9:
                            away = p_pred - center
                            if abs(away[0]) > abs(away[1]):
                                dp = np.array([math.copysign(1.0, away[0] or 1.0), 0.0])
                            else:
                                dp = np.array([0.0, math.copysign(1.0, away[1] or 1.0)])
                        dist = 0.0
                    # 門牆同樣加 footprint 投影（正面接近才加餘量，側向通過維持原 d_safe）。
                    d_safe = max(
                        d_safe_base,
                        self._footprint_support_along(dp, s.yaw))
                    h_rect = dist ** 2 - d_safe ** 2
                    hdot = float(2.0 * dp @ s.vel_world)
                    hddot_free = float(
                        2.0 * s.vel_world @ s.vel_world
                        + 2.0 * dp @ yaw_acc_world[name])
                    a_row = np.zeros(n_vars)
                    col = 2 * dog_idx[name]
                    a_row[col:col + 2] = 2.0 * dp @ R_dogs[name]
                    A_rows.append(a_row)
                    b_rows.append(self._hocbf_rhs(
                        h_rect, hdot, hddot_free,
                        self.gamma_obs_1, self.gamma_obs_2))
                    row_meta.append({
                        "kind": "rect",
                        "name": str(rect.get("center", "?")),
                        "h": h_rect,
                        "hdot": hdot,
                    })

            # ── Wall HOCBF ──
            for wall in self.walls:
                n_w    = np.array(wall["normal"][:2], dtype=float)
                p_w    = np.array(wall["point"][:2],  dtype=float)
                d_safe_base = float(wall.get("d_safe", 0.4))
                for name in all_dogs:
                    s = states[name]
                    if not s.received:
                        continue
                    d_safe = max(
                        d_safe_base,
                        self._footprint_support_along(n_w, s.yaw),
                    )
                    p_pred = s.pos
                    h_wall = float(n_w @ (p_pred - p_w)) - d_safe
                    hdot = float(n_w @ s.vel_world)
                    a_row  = np.zeros(n_vars)
                    col    = 2 * dog_idx[name]
                    a_row[col:col + 2] = n_w @ R_dogs[name]
                    A_rows.append(a_row)
                    hddot_free = float(n_w @ yaw_acc_world[name])
                    b_rows.append(self._hocbf_rhs(
                        h_wall, hdot, hddot_free,
                        self.gamma_wall_1, self.gamma_wall_2))
                    row_meta.append({
                        "kind": "wall",
                        "name": str(wall.get("normal", "?")),
                        "h": h_wall,
                        "hdot": hdot,
                    })

            # 純二階 HOCBF：不再額外加入速度層 CBF constraint。

        # ═══ 建構 acceleration QP objective ═══
        a = cp.Variable(n_vars)
        u_next = u_now + dt * a

        # a_d: position error -> desired acceleration (body frame)
        if a_desired is not None:
            a_d = np.array(a_desired, dtype=float)
        else:
            # fallback: derive an acceleration reference from velocity error.
            a_d = (u_nominal - u_now) / dt

        cost_tracking = self.w_track * cp.sum_squares(a - a_d)

        # f(p + R_i (u_now dt + 0.5 a_i dt²))
        # ~= const + grad_f_i^T R_i (0.5 a_i dt²).
        cost_formation_linear = 0.0
        if (self.w_formation > 1e-9
                and formation_grad is not None
                and len(formation_grad) == n_dogs):
            g_vec = np.zeros(n_vars)
            for idx, name in enumerate(all_dogs):
                s = states[name]
                if not s.received:
                    continue
                grad_world = np.array(formation_grad[idx], dtype=float)
                g_vec[2 * idx:2 * idx + 2] = (
                    grad_world @ R_dogs[name]) * (0.5 * dt * dt)
            cost_formation_linear = self.w_formation * (g_vec @ a)

        cost_accel = self.w_accel * cp.sum_squares(a)

        objective_terms = [
            cost_tracking,
            cost_formation_linear,
            cost_accel,
        ]
        constraints = []

        for idx in range(n_dogs):
            ax = a[2 * idx]
            ay = a[2 * idx + 1]
            vx_next = u_next[2 * idx]
            vy_next = u_next[2 * idx + 1]
            constraints += [
                ax <= self.max_ax,
                ax >= -self.max_ax,
                ay <= self.max_ay,
                ay >= -self.max_ay,
                vx_next <= self.max_vx,
                vx_next >= -self.max_vx,
                vy_next <= self.max_vy,
                vy_next >= -self.max_vy,
            ]

        eps = None
        if A_rows:
            A = np.array(A_rows)
            b = np.array(b_rows)
            if self.slack_enabled:
                # Soft HOCBF: A·a >= b - ε, ε>=0, 重罰 λ‖ε‖²。
                # 可行時 ε=0 → 嚴格 CBF；窄處 infeasible 時取最小違反，不卡死。
                eps = cp.Variable(A.shape[0], nonneg=True)
                constraints.append(A @ a >= b - eps)
                objective_terms.append(self.slack_lambda * cp.sum_squares(eps))
            else:
                # 硬 CBF（驗證用，無 slack）：不可行 → QP fail → emergency brake。
                # 此模式下 h≥0 才是真正的安全保證,可用 /cbf_debug/h_min_* 確認。
                constraints.append(A @ a >= b)

        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            rospy.logwarn(
                "[TwoOrderCBF] Solver error, emergency braking")
            self.last_cbf_status = "solver_error"
            return self._emergency_brake_result(all_dogs, u_now, n_dogs)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            rospy.logwarn(
                "[TwoOrderCBF] QP status=%s, emergency braking", prob.status)
            self._log_hocbf_infeasible_diagnostics(
                A_rows, b_rows, row_meta, u_now, dt, n_dogs)
            self.last_cbf_status = str(prob.status)
            return self._emergency_brake_result(all_dogs, u_now, n_dogs)

        self.last_cbf_status = str(prob.status)
        self._record_slack(eps, row_meta)

        a_sol = np.array(a.value, dtype=float)
        self._last_a_sol = a_sol.copy()
        u_next_sol = self._clip_body_velocity_vector(
            u_now + dt * a_sol, n_dogs)
        effective_a = (u_next_sol - u_now) / dt
        self.last_accel_cmds = {
            name: effective_a[2 * dog_idx[name]:2 * dog_idx[name] + 2].copy()
            for name in all_dogs
        }
        self._store_one_step_prediction(
            all_dogs, states, u_now, effective_a, dt)

        return {name: (float(u_next_sol[2 * dog_idx[name]]),
                       float(u_next_sol[2 * dog_idx[name] + 1]))
                for name in all_dogs}

    def _prediction_step_dt(self, dt):
        if self.prediction_dt > 1e-9:
            return self.prediction_dt
        return max(1e-6, float(dt))

    def _clip_body_velocity_vector(self, u_vec, n_dogs):
        u = np.array(u_vec, dtype=float).copy()
        for idx in range(n_dogs):
            u[2 * idx] = float(np.clip(u[2 * idx], -self.max_vx, self.max_vx))
            u[2 * idx + 1] = float(
                np.clip(u[2 * idx + 1], -self.max_vy, self.max_vy))
        return u

    def _clip_body_velocity_sequence(self, v_seq, n_dogs, n_steps):
        v_seq = np.array(v_seq, dtype=float).copy()
        n_per_step = 2 * n_dogs
        for k in range(n_steps):
            sl = slice(k * n_per_step, (k + 1) * n_per_step)
            v_seq[sl] = self._clip_body_velocity_vector(v_seq[sl], n_dogs)
        return v_seq

    def _rollout_prediction_paths(self, all_dogs, states, v_seq, R_dogs,
                                  dt, n_steps):
        n_dogs = len(all_dogs)
        n_per_step = 2 * n_dogs
        paths = {name: [states[name].pos.copy()] for name in all_dogs}
        for k in range(n_steps):
            base = k * n_per_step
            for idx, name in enumerate(all_dogs):
                v_body = v_seq[base + 2 * idx:base + 2 * idx + 2]
                next_pos = paths[name][-1] + R_dogs[name] @ v_body * dt
                paths[name].append(np.array(next_pos, dtype=float))
        return paths

    def _solve_multistep_preview(self, all_dogs, states, u_nominal,
                                 a_desired=None, formation_grad=None,
                                 cbf_enabled=True, dt=0.05, yaw_rates=None):
        """二階 multi-step preview HOCBF-QP。

        決策變數: 整段 body-frame 加速度序列 A = [a_0, a_1, ..., a_{N-1}],
        每個 a_k 是 2*n_dogs 維。只發布第一步 a_0。

        參考軌跡 (reference rollout): 用上一輪解 shift 一格當參考,沿參考的
        body 速度做二次位置外推:
            v_k^B = v_{k-1}^B + a_k^ref * dt        (參考加速度)
            p_k^W = p_{k-1}^W + R v_{k-1}^B dt + 1/2 R a_k^ref dt^2
        未來各步的 HOCBF row 在這些參考點 (p_k, v_k) 上線性化:
            hddot + (g1+g2) hdot + g1 g2 h >= 0
        其中決策變數 a_k 只進 hddot 的 a 項 (2 dp^T R a_k),
        hdot/h 用參考量 (已知常數)。w_smooth 綁相鄰 a_k,把未來趨勢回傳 a_0。

        維持無 slack 硬約束 (slack_enabled 控制),不可行則 emergency brake。
        WBC 介面 / cmd_vel 介面不變: 仍只用第一步 a_0。
        """
        n_dogs = len(all_dogs)
        n_per = 2 * n_dogs
        n_steps = max(2, int(self.prediction_horizon))
        n_total = n_per * n_steps
        dog_idx = {name: i for i, name in enumerate(all_dogs)}
        dt = max(1e-6, float(dt))
        dt_pred = self._prediction_step_dt(dt)

        # 冷啟動: 還沒有暖啟動序列時,第一輪先用 single-step 解 (穩且能立即
        # 產生合理 a_0),並把它擴成整段序列存入暖啟動,下一輪起才真正跑
        # horizon preview。避免冷啟動參考軌跡不準導致第一次無謂 brake。
        if self._last_A_seq is None or len(self._last_A_seq) != n_total:
            res = self._solve_single_step(
                all_dogs, states, u_nominal, a_desired=a_desired,
                formation_grad=formation_grad, cbf_enabled=cbf_enabled,
                dt=dt, yaw_rates=yaw_rates)
            if self._last_a_sol is not None and len(self._last_a_sol) == n_per:
                self._last_A_seq = np.tile(self._last_a_sol, n_steps)
            return res

        R_dogs = {name: rot2d(states[name].yaw) for name in all_dogs}
        u_now = self._current_body_velocity_vector(all_dogs, states)
        u_nominal = self._clip_body_velocity_vector(u_nominal, n_dogs)

        if yaw_rates is None:
            yaw_rates = {}

        # a_desired (body frame) for tracking; 沿 horizon 重複使用第一步 ref。
        if a_desired is not None:
            a_d0 = np.array(a_desired, dtype=float)
        else:
            a_d0 = (u_nominal - u_now) / dt_pred

        # ── 參考加速度序列 (暖啟動: 上一輪解 shift 一格, 末步補 a_d0) ──
        # 注意: 第一輪沒有暖啟動時,不能用「固定衝刺加速度」外推,否則參考
        # 軌跡會直直穿進障礙,使遠期 HOCBF row 要求過量煞車而整段 infeasible。
        # fallback 改用「零加速度」(等速滑行) 當保守參考,讓第一輪能解出來,
        # 之後的 cycle 再靠暖啟動 (上一輪會減速的解) 自然收斂。
        if self._last_A_seq is not None and len(self._last_A_seq) == n_total:
            a_ref_seq = np.concatenate([
                self._last_A_seq[n_per:], self._last_A_seq[-n_per:]
            ])
        else:
            a_ref_seq = np.zeros(n_total)

        # ── 沿參考序列 rollout: 每步每狗的參考位置 p_k^W 與參考速度 v_k^W ──
        # p_ref[name] = [p_0, p_1, ..., p_{N-1}] (world)
        # v_ref[name] = [v_0, v_1, ..., v_{N-1}] (world)
        p_ref = {name: [states[name].pos.copy()] for name in all_dogs}
        v_ref_body = {name: [u_now[2 * dog_idx[name]:2 * dog_idx[name] + 2].copy()]
                      for name in all_dogs}
        v_ref_world = {name: [states[name].vel_world.copy()] for name in all_dogs}
        for k in range(1, n_steps):
            for name in all_dogs:
                i = dog_idx[name]
                R = R_dogs[name]
                a_k_body = a_ref_seq[(k - 1) * n_per + 2 * i:
                                     (k - 1) * n_per + 2 * i + 2]
                v_prev_body = v_ref_body[name][-1]
                p_prev = p_ref[name][-1]
                # 二次位置外推 (yaw 固定為當前, body frame 線性化)
                dp_world = R @ (v_prev_body * dt_pred
                                + 0.5 * a_k_body * dt_pred * dt_pred)
                p_new = p_prev + dp_world
                v_new_body = v_prev_body + a_k_body * dt_pred
                v_new_world = R @ v_new_body
                p_ref[name].append(p_new)
                v_ref_body[name].append(v_new_body)
                v_ref_world[name].append(v_new_world)

        A_rows, b_rows, row_meta = [], [], []

        # row 全部建立並以 "step" 標記; 組裝時 k==0 進硬約束 (無 slack 嚴格
        # 保證), k>=1 進軟性 cost (預警, 不製造 infeasible)。
        if cbf_enabled:
            for k in range(n_steps):
                col_off = k * n_per

                # ── Pairwise robot-robot HOCBF ──
                for ia in range(n_dogs):
                    for ib in range(ia + 1, n_dogs):
                        na, nb = all_dogs[ia], all_dogs[ib]
                        sa, sb = states[na], states[nb]
                        if not sa.received or not sb.received:
                            continue
                        dp = p_ref[na][k] - p_ref[nb][k]
                        dv = v_ref_world[na][k] - v_ref_world[nb][k]
                        d_min_eff = max(
                            self.d_min,
                            self._footprint_support_along(dp, sa.yaw)
                            + self._footprint_support_along(dp, sb.yaw))
                        h = float(dp @ dp) - d_min_eff ** 2
                        hdot = float(2.0 * dp @ dv)
                        # hddot_free: 參考速度的 2|dv|^2 (yaw 項在 preview 忽略)
                        hddot_free = float(2.0 * dv @ dv)
                        a_row = np.zeros(n_total)
                        ca = col_off + 2 * dog_idx[na]
                        a_row[ca:ca + 2] = 2.0 * dp @ R_dogs[na]
                        cb = col_off + 2 * dog_idx[nb]
                        a_row[cb:cb + 2] = -2.0 * dp @ R_dogs[nb]
                        A_rows.append(a_row)
                        b_rows.append(self._hocbf_rhs(
                            h, hdot, hddot_free,
                            self.gamma_robot_1, self.gamma_robot_2))
                        row_meta.append({"kind": "pair", "step": k,
                                         "name": "%s-%s" % (na, nb),
                                         "h": h, "hdot": hdot})

                # ── Circular obstacle HOCBF ──
                for obs in self.obstacles:
                    p_obs = np.array(obs["pos"][:2])
                    r_base = float(obs["radius"])
                    r_phys = float(obs.get("physical_radius", 0.2))
                    for name in all_dogs:
                        s = states[name]
                        if not s.received:
                            continue
                        pk = p_ref[name][k]
                        vk = v_ref_world[name][k]
                        dp = pk - p_obs
                        r_obs = max(
                            r_base,
                            r_phys + self._footprint_support_along(dp, s.yaw))
                        h_obs = float(dp @ dp) - r_obs ** 2
                        hdot = float(2.0 * dp @ vk)
                        hddot_free = float(2.0 * vk @ vk)
                        a_row = np.zeros(n_total)
                        col = col_off + 2 * dog_idx[name]
                        a_row[col:col + 2] = 2.0 * dp @ R_dogs[name]
                        A_rows.append(a_row)
                        b_rows.append(self._hocbf_rhs(
                            h_obs, hdot, hddot_free,
                            self.gamma_obs_1, self.gamma_obs_2))
                        row_meta.append({"kind": "obs", "step": k,
                                         "name": str(obs.get("pos", "?")),
                                         "h": h_obs, "hdot": hdot})

                # ── Rect obstacle HOCBF ──
                for rect in self.rect_obstacles:
                    center = np.array(rect["center"][:2], dtype=float)
                    size = np.array(rect["size"][:2], dtype=float)
                    d_safe_base = float(rect.get("d_safe", 0.35))
                    for name in all_dogs:
                        s = states[name]
                        if not s.received:
                            continue
                        pk = p_ref[name][k]
                        vk = v_ref_world[name][k]
                        p_closest = closest_point_on_aabb(pk, center, size)
                        dp = pk - p_closest
                        dist = float(np.linalg.norm(dp))
                        if dist < 1e-4:
                            escape = rect.get("escape_dir", None)
                            if escape is not None:
                                dp = np.array(escape[:2], dtype=float)
                                norm = float(np.linalg.norm(dp))
                                dp = dp / norm if norm > 1e-9 else np.zeros(2)
                            if float(np.linalg.norm(dp)) < 1e-9:
                                away = pk - center
                                if abs(away[0]) > abs(away[1]):
                                    dp = np.array(
                                        [math.copysign(1.0, away[0] or 1.0), 0.0])
                                else:
                                    dp = np.array(
                                        [0.0, math.copysign(1.0, away[1] or 1.0)])
                            dist = 0.0
                        d_safe = max(
                            d_safe_base,
                            self._footprint_support_along(dp, s.yaw))
                        h_rect = dist ** 2 - d_safe ** 2
                        hdot = float(2.0 * dp @ vk)
                        hddot_free = float(2.0 * vk @ vk)
                        a_row = np.zeros(n_total)
                        col = col_off + 2 * dog_idx[name]
                        a_row[col:col + 2] = 2.0 * dp @ R_dogs[name]
                        A_rows.append(a_row)
                        b_rows.append(self._hocbf_rhs(
                            h_rect, hdot, hddot_free,
                            self.gamma_obs_1, self.gamma_obs_2))
                        row_meta.append({"kind": "rect", "step": k,
                                         "name": str(rect.get("center", "?")),
                                         "h": h_rect, "hdot": hdot})

                # ── Wall HOCBF ──
                for wall in self.walls:
                    n_w = np.array(wall["normal"][:2], dtype=float)
                    p_w = np.array(wall["point"][:2], dtype=float)
                    d_safe_base = float(wall.get("d_safe", 0.4))
                    for name in all_dogs:
                        s = states[name]
                        if not s.received:
                            continue
                        pk = p_ref[name][k]
                        vk = v_ref_world[name][k]
                        d_safe = max(
                            d_safe_base,
                            self._footprint_support_along(n_w, s.yaw))
                        h_wall = float(n_w @ (pk - p_w)) - d_safe
                        hdot = float(n_w @ vk)
                        hddot_free = 0.0
                        a_row = np.zeros(n_total)
                        col = col_off + 2 * dog_idx[name]
                        a_row[col:col + 2] = n_w @ R_dogs[name]
                        A_rows.append(a_row)
                        b_rows.append(self._hocbf_rhs(
                            h_wall, hdot, hddot_free,
                            self.gamma_wall_1, self.gamma_wall_2))
                        row_meta.append({"kind": "wall", "step": k,
                                         "name": str(wall.get("normal", "?")),
                                         "h": h_wall, "hdot": hdot})

        # ═══ QP objective ═══
        A = cp.Variable(n_total)
        objective_terms = []

        # tracking: 只對第一步 a_0 追 a_desired (其餘步交給 smooth 連動)
        sl0 = slice(0, n_per)
        objective_terms.append(self.w_track * cp.sum_squares(A[sl0] - a_d0))

        # formation linear cost: 只加在第一步 (與 single-step 一致)
        if (self.w_formation > 1e-9
                and formation_grad is not None
                and len(formation_grad) == n_dogs):
            g_vec = np.zeros(n_per)
            for idx, name in enumerate(all_dogs):
                s = states[name]
                if not s.received:
                    continue
                grad_world = np.array(formation_grad[idx], dtype=float)
                g_vec[2 * idx:2 * idx + 2] = (
                    grad_world @ R_dogs[name]) * (0.5 * dt_pred * dt_pred)
            objective_terms.append(self.w_formation * (g_vec @ A[sl0]))

        # accel regularization: 整段
        objective_terms.append(self.w_accel * cp.sum_squares(A))

        # smooth: 綁相鄰步加速度,把未來趨勢回傳第一步
        if self.w_smooth > 0.0:
            for k in range(n_steps - 1):
                slk = slice(k * n_per, (k + 1) * n_per)
                slk1 = slice((k + 1) * n_per, (k + 2) * n_per)
                objective_terms.append(
                    self.w_smooth * cp.sum_squares(A[slk1] - A[slk]))

        # ═══ bounds: 每步加速度上限 + 累積速度上限 (沿參考速度) ═══
        constraints = []
        for k in range(n_steps):
            for idx in range(n_dogs):
                ax = A[k * n_per + 2 * idx]
                ay = A[k * n_per + 2 * idx + 1]
                constraints += [ax <= self.max_ax, ax >= -self.max_ax,
                                ay <= self.max_ay, ay >= -self.max_ay]
                # 第一步: 速度上限用真實 u_now 約束 v_next
                if k == 0:
                    vx_next = u_now[2 * idx] + dt_pred * ax
                    vy_next = u_now[2 * idx + 1] + dt_pred * ay
                    constraints += [
                        vx_next <= self.max_vx, vx_next >= -self.max_vx,
                        vy_next <= self.max_vy, vy_next >= -self.max_vy]

        # ═══ CBF row 依 step 分流 ═══
        #   k==0 (當前真實狀態): 硬約束 A0·a >= b0
        #       - slack_enabled=False → 純硬 CBF, 嚴格保證 (教授要的)
        #       - slack_enabled=True  → 軟化 (相容舊行為)
        #   k>=1 (預測步): 軟性 cost w_pred * relu(b - A·a)^2
        #       未來約束滿足時不罰; 違反時溫和拉回, 把減速趨勢回傳第一步。
        #       cost 永遠可行, 不會像硬約束那樣製造 infeasible。
        eps = None
        hard_idx = [i for i, m in enumerate(row_meta)
                    if m.get("step", 0) == 0]
        soft_idx = [i for i, m in enumerate(row_meta)
                    if m.get("step", 0) != 0]

        if hard_idx:
            A_hard = np.array([A_rows[i] for i in hard_idx])
            b_hard = np.array([b_rows[i] for i in hard_idx])
            if self.slack_enabled:
                eps = cp.Variable(A_hard.shape[0], nonneg=True)
                constraints.append(A_hard @ A >= b_hard - eps)
                objective_terms.append(
                    self.slack_lambda * cp.sum_squares(eps))
            else:
                constraints.append(A_hard @ A >= b_hard)

        if soft_idx:
            A_soft = np.array([A_rows[i] for i in soft_idx])
            b_soft = np.array([b_rows[i] for i in soft_idx])
            # 違反量 viol = max(0, b - A·a); cost = w_pred * sum(viol^2)
            # 用溫和權重 (非 slack_lambda 的 1e4): 太大會讓遠期軟約束變回硬性、
            # 重新製造 infeasible 的傾向; 太小則前瞻減速不夠。w_pred~5~50 合適。
            w_pred = float(getattr(self, "w_pred", 20.0))
            viol = cp.pos(b_soft - A_soft @ A)
            objective_terms.append(w_pred * cp.sum_squares(viol))

        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            rospy.logwarn("[TwoOrderCBF] Solver error (multistep), braking")
            self.last_cbf_status = "solver_error"
            self.reset_prediction("solver error")
            return self._emergency_brake_result(all_dogs, u_now, n_dogs)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            rospy.logwarn(
                "[TwoOrderCBF] QP status=%s (multistep), braking", prob.status)
            # 用 step==0 標記精確抓第一步 row、只取前 n_per 欄,維度對得上 u_now
            first_idx = [i for i, m in enumerate(row_meta)
                         if m.get("step", 0) == 0]
            if first_idx:
                A_first = [A_rows[i][:n_per] for i in first_idx]
                b_first = [b_rows[i] for i in first_idx]
                m_first = [row_meta[i] for i in first_idx]
                self._log_hocbf_infeasible_diagnostics(
                    A_first, b_first, m_first, u_now, dt_pred, n_dogs)
            self.last_cbf_status = str(prob.status)
            self.reset_prediction("non-optimal status")
            return self._emergency_brake_result(all_dogs, u_now, n_dogs)

        self.last_cbf_status = str(prob.status)
        A_sol = np.array(A.value, dtype=float)
        self._last_A_seq = A_sol.copy()

        # 只取第一步
        a_first = A_sol[:n_per]
        self._last_a_sol = a_first.copy()

        # slack 記錄: 只看第一步那段 row (step==0)
        first_idx = [i for i, m in enumerate(row_meta)
                     if m.get("step", 0) == 0]
        m_first = [row_meta[i] for i in first_idx]
        if eps is not None and eps.value is not None:
            eps_arr = np.asarray(eps.value, dtype=float).ravel()
            eps_first = [eps_arr[i] for i in first_idx if i < eps_arr.size]
            self._record_slack_array(np.array(eps_first), m_first)
        else:
            self._record_slack_array(None, m_first)

        u_next_sol = self._clip_body_velocity_vector(
            u_now + dt_pred * a_first, n_dogs)
        effective_a = (u_next_sol - u_now) / dt_pred
        self.last_accel_cmds = {
            name: effective_a[2 * dog_idx[name]:2 * dog_idx[name] + 2].copy()
            for name in all_dogs
        }
        self._store_one_step_prediction(
            all_dogs, states, u_now, effective_a, dt_pred)

        return {name: (float(u_next_sol[2 * dog_idx[name]]),
                       float(u_next_sol[2 * dog_idx[name] + 1]))
                for name in all_dogs}


UnifiedQPController = TwoOrderCBFQPController


# ═══════════════════════════════════════════════════════════════
# Module 5: VelocityLimiter（不動）
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
# Module 6: CmdVelPublisher（不動）
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


class AccelCmdPublisher:
    def __init__(self, all_dog_names, scale=1.0):
        self._pubs = {
            name: rospy.Publisher(f"/{name}/accel_cmd", Vector3, queue_size=1)
            for name in all_dog_names
        }
        # Scales the upper-layer acceleration correction fed to the WBC base task.
        # The published value should be Δa = a_QP* - a_nom, not the full a_QP*.
        self._scale = float(scale)

    def publish(self, name, ax, ay):
        msg = Vector3()
        msg.x = float(ax) * self._scale
        msg.y = float(ay) * self._scale
        msg.z = 0.0
        self._pubs[name].publish(msg)

    def publish_zero(self, names):
        zero = Vector3()
        for name in names:
            self._pubs[name].publish(zero)


class CbfDebugPublisher:
    """發布 CBF 監看量到 /cbf_debug/*（Float32），用 rqt_plot 即時看。
    h_min_<kind>: 各類約束最小 h（實際安全距離，<0 = 撞）
    slack_<kind>: ψ2 被違反量（含速度層 _v）
    vcmd_<dog> / vact_<dog>: 命令 vs 實際速度大小（執行落差）
    """

    _KINDS = ("obs", "wall", "pair", "rect")

    def __init__(self, dog_names):
        self._pubs = {}
        for k in self._KINDS:
            self._pubs["h_" + k] = rospy.Publisher(
                "/cbf_debug/h_min_" + k, Float32, queue_size=1)
            self._pubs["s_" + k] = rospy.Publisher(
                "/cbf_debug/slack_" + k, Float32, queue_size=1)
        for name in dog_names:
            self._pubs["vc_" + name] = rospy.Publisher(
                "/cbf_debug/vcmd_" + name, Float32, queue_size=1)
            self._pubs["va_" + name] = rospy.Publisher(
                "/cbf_debug/vact_" + name, Float32, queue_size=1)

    def publish(self, min_h, slack, vcmd, vact):
        for k in self._KINDS:
            self._pubs["h_" + k].publish(Float32(float(min_h.get(k, 9.0))))
            # 取 base 與速度層(_v)的較大 slack
            s = max(float(slack.get(k, 0.0)), float(slack.get(k + "_v", 0.0)))
            self._pubs["s_" + k].publish(Float32(s))
        for name, v in vcmd.items():
            self._pubs["vc_" + name].publish(Float32(float(v)))
        for name, v in vact.items():
            self._pubs["va_" + name].publish(Float32(float(v)))


# ═══════════════════════════════════════════════════════════════
# 主控: FleetManagerUQP
# ═══════════════════════════════════════════════════════════════

class FleetManagerUQP:
    """
    統一 QP 主控流程（每個 cycle @ rate_hz）:

        1. StateCollector → 讀取三隻狗位置
        2. LeaderNavigator → virtual center 的 u_ref (A*+PP or joystick)
        3. FormationSwitcher → 可能切換 L̂_des（自動偵測窄門）
        4. LaplacianFormation → 計算 f 和 ∂f/∂p → 轉成 g (body frame)
        5. TwoOrderCBFQPController → 解 acceleration QP → u_safe × 3
        6. Follower wz: P control 對齊 leader yaw
        7. VelocityLimiter + CmdVelPublisher → /dogN/cmd_vel
    """

    @staticmethod
    def _forbidden_zones_to_rect_obstacles(forbidden_zones, default_d_safe):
        rects = []
        for zone in forbidden_zones or []:
            try:
                center = [float(zone["center"][0]), float(zone["center"][1])]
                size = [float(zone["size"][0]), float(zone["size"][1])]
            except (KeyError, TypeError, ValueError, IndexError):
                rospy.logwarn("[FleetManagerUQP] skip malformed forbidden zone: %s",
                              zone)
                continue
            rects.append({
                "name": zone.get("name", "forbidden_zone"),
                "center": center,
                "size": size,
                "d_safe": float(zone.get("d_safe", default_d_safe)),
                "escape_dir": zone.get("escape_dir", None),
                "virtual_forbidden": True,
            })
        return rects

    def __init__(self):
        rospy.init_node("fleet_manager_uqp")

        # ── 基本參數 ──
        self.leader_name = rospy.get_param(
            "~leader_name", _CFG.get("leader_name", "dog1"))
        self.follower_names = list(rospy.get_param(
            "~follower_names", _CFG.get("follower_names", ["dog2", "dog3"])))
        self.all_dogs = [self.leader_name] + self.follower_names
        self.rate_hz = rospy.get_param("~rate", _CFG.get("rate", 20.0))
        self.stop_without_leader = rospy.get_param("~stop_without_leader", True)
        self.goal_topic = rospy.get_param(
            "~goal_topic", _CFG.get("goal_topic", "/formation/goal"))
        self.cmd_vel_raw_topic = rospy.get_param(
            "~cmd_vel_raw_topic", _CFG.get("cmd_vel_raw_topic", "/formation/cmd_vel_raw"))

        # ── Module 1: StateCollector ──
        self.state_collector = StateCollector(self.all_dogs)

        # ── Module A: AStarPlanner ──
        obstacles = rospy.get_param("~obstacles", _CFG.get("obstacles", []))
        rect_obstacles = rospy.get_param("~rect_obstacles",
                                          _CFG.get("rect_obstacles", []))
        self.forbidden_zones_enabled = bool(rospy.get_param(
            "~forbidden_zones_enabled",
            _CFG.get("forbidden_zones_enabled", True)))
        forbidden_zones = rospy.get_param("~astar_forbidden_zones",
                                           _CFG.get("astar_forbidden_zones", []))
        if not self.forbidden_zones_enabled:
            forbidden_zones = []
        self.astar = AStarPlanner(
            resolution=rospy.get_param(
                "~astar_resolution", _CFG.get("astar_resolution", 0.1)),
            robot_radius=rospy.get_param(
                "~astar_robot_radius", _CFG.get("astar_robot_radius", 0.15)),
            obstacles=obstacles,
            x_min=rospy.get_param("~map_x_min", _CFG.get("map_x_min", 0.0)),
            x_max=rospy.get_param("~map_x_max", _CFG.get("map_x_max", 10.0)),
            y_min=rospy.get_param("~map_y_min", _CFG.get("map_y_min", -5.0)),
            y_max=rospy.get_param("~map_y_max", _CFG.get("map_y_max",  5.0)),
            boundary_margin=rospy.get_param(
                "~astar_boundary_margin", _CFG.get("astar_boundary_margin", 0.45)),
            rect_obstacles=rect_obstacles,
            forbidden_zones=forbidden_zones,
        )
        self.astar_block_narrow_wall_gaps = bool(rospy.get_param(
            "~astar_block_narrow_wall_gaps", _CFG.get("astar_block_narrow_wall_gaps", True)))
        self.astar_wall_gap_min_width = float(rospy.get_param(
            "~astar_wall_gap_min_width", _CFG.get("astar_wall_gap_min_width", 1.10)))
        self.astar_wall_gap_lateral_margin = float(rospy.get_param(
            "~astar_wall_gap_lateral_margin", _CFG.get("astar_wall_gap_lateral_margin", 0.20)))
        self.astar_wall_gap_obstacle_indices = rospy.get_param(
            "~astar_wall_gap_obstacle_indices",
            _CFG.get("astar_wall_gap_obstacle_indices", [2, 3]))
        self.astar.set_narrow_wall_gap_blocking(
            self.astar_block_narrow_wall_gaps,
            min_width=self.astar_wall_gap_min_width,
            lateral_margin=self.astar_wall_gap_lateral_margin,
            obstacle_indices=self.astar_wall_gap_obstacle_indices,
        )
        self.astar_goal_min_obstacle_clearance = float(rospy.get_param(
            "~astar_goal_min_obstacle_clearance",
            _CFG.get("astar_goal_min_obstacle_clearance", 0.20)))
        self.astar_goal_min_wall_clearance = float(rospy.get_param(
            "~astar_goal_min_wall_clearance",
            _CFG.get("astar_goal_min_wall_clearance", 0.65)))
        self.astar.set_goal_candidate_clearance(
            min_obstacle_clearance=self.astar_goal_min_obstacle_clearance,
            min_wall_clearance=self.astar_goal_min_wall_clearance,
        )

        # ── Module B: PurePursuitController ──
        self.pursuer = PurePursuitController(
            look_ahead=rospy.get_param("~pp_look_ahead",
                                        _CFG.get("pp_look_ahead", 0.8)),
            v_cruise=rospy.get_param("~pp_v_cruise",
                                      _CFG.get("pp_v_cruise", 0.22)),
            kp_yaw=rospy.get_param("~pp_kp_yaw",
                                    _CFG.get("pp_kp_yaw", 0.6)),
            goal_tol=rospy.get_param("~pp_goal_tol",
                                      _CFG.get("pp_goal_tol", 0.3)),
            astar=self.astar,
        )
        self.dog_pursuers = {
            name: PurePursuitController(
                look_ahead=rospy.get_param("~pp_look_ahead",
                                            _CFG.get("pp_look_ahead", 0.8)),
                v_cruise=rospy.get_param("~pp_v_cruise",
                                          _CFG.get("pp_v_cruise", 0.22)),
                kp_yaw=rospy.get_param("~pp_kp_yaw",
                                        _CFG.get("pp_kp_yaw", 0.6)),
                goal_tol=rospy.get_param("~pp_goal_tol",
                                          _CFG.get("pp_goal_tol", 0.3)),
                astar=self.astar,
            )
            for name in self.all_dogs
        }

        # ── Module C: LeaderNavigator ──
        self.astar_goal_adjust_max_dist = float(rospy.get_param(
            "~astar_goal_adjust_max_dist",
            _CFG.get("astar_goal_adjust_max_dist", 0.75)))
        self.navigator = LeaderNavigator(
            self.goal_topic, self.cmd_vel_raw_topic, self.astar, self.pursuer,
            max_goal_adjust_dist=self.astar_goal_adjust_max_dist)

        # ── Module NEW-1: LaplacianFormation ──
        # 從 YAML 讀取隊形定義
        default_formations = {
            "V":    [[0.67, 0.0], [-0.33, 1.0], [-0.33, -1.0]],
            "V_narrow": [[0.80, 0.0], [-0.20, 0.475], [-0.60, -0.475]],
            "line": [[1.2, 0.0], [0.0, 0.0], [-1.2, 0.0]],
        }
        formation_configs = rospy.get_param(
            "~formations", _CFG.get("formations", default_formations))
        # 轉換成 numpy arrays
        formation_np = {}
        for name, pts in formation_configs.items():
            formation_np[name] = [np.array(p, dtype=float) for p in pts]
        self.laplacian = LaplacianFormation(formation_np)

        default_formation = rospy.get_param(
            "~default_formation", _CFG.get("default_formation", "V"))
        self.laplacian.set_formation(default_formation)

        # ── Module NEW-2: FormationSwitcher ──
        door_enabled = rospy.get_param(
            "~door_mode_enabled", _CFG.get("door_mode_enabled", True))
        self.switcher = FormationSwitcher(
            self.laplacian,
            door_x=rospy.get_param("~door_x", _CFG.get("door_x", 6.0)),
            default_formation=default_formation,
            passage_formation=rospy.get_param(
                "~door_passage_formation",
                rospy.get_param(
                    "~door_line_formation",
                    _CFG.get(
                        "door_passage_formation",
                        _CFG.get("door_line_formation", "line")))),
            trigger_dist=rospy.get_param(
                "~door_trigger_dist", _CFG.get("door_trigger_dist", 3.0)),
            release_dist=rospy.get_param(
                "~door_release_dist", _CFG.get("door_release_dist", 2.0)),
        )
        self._door_enabled = door_enabled

        # ── Module 4': TwoOrderCBFQPController ──
        self.cbf_enabled = rospy.get_param(
            "~cbf_enabled", _CFG.get("cbf_enabled", True))
        cbf_gamma = rospy.get_param(
            "~cbf_gamma", _CFG.get("cbf_gamma", 1.0))
        cbf_gamma_obs = rospy.get_param(
            "~cbf_gamma_obs", _CFG.get("cbf_gamma_obs", 0.5))
        cbf_gamma_wall = rospy.get_param(
            "~cbf_gamma_wall", _CFG.get("cbf_gamma_wall", 1.0))
        cbf_gamma1 = rospy.get_param(
            "~cbf_gamma1", _CFG.get("cbf_gamma1", cbf_gamma))
        cbf_gamma2 = rospy.get_param(
            "~cbf_gamma2", _CFG.get("cbf_gamma2", cbf_gamma))
        cbf_gamma_obs_1 = rospy.get_param(
            "~cbf_gamma_obs_1", _CFG.get("cbf_gamma_obs_1", cbf_gamma_obs))
        cbf_gamma_obs_2 = rospy.get_param(
            "~cbf_gamma_obs_2", _CFG.get("cbf_gamma_obs_2", cbf_gamma_obs))
        cbf_gamma_wall_1 = rospy.get_param(
            "~cbf_gamma_wall_1", _CFG.get("cbf_gamma_wall_1", cbf_gamma_wall))
        cbf_gamma_wall_2 = rospy.get_param(
            "~cbf_gamma_wall_2", _CFG.get("cbf_gamma_wall_2", cbf_gamma_wall))
        self.qp = TwoOrderCBFQPController(
            gamma_robot=cbf_gamma,
            gamma_robot_1=rospy.get_param(
                "~cbf_gamma_robot_1",
                _CFG.get("cbf_gamma_robot_1", cbf_gamma1)),
            gamma_robot_2=rospy.get_param(
                "~cbf_gamma_robot_2",
                _CFG.get("cbf_gamma_robot_2", cbf_gamma2)),
            d_min=rospy.get_param(
                "~cbf_d_min", _CFG.get("cbf_d_min", 1.0)),
            gamma_obs=cbf_gamma_obs,
            gamma_obs_1=cbf_gamma_obs_1,
            gamma_obs_2=cbf_gamma_obs_2,
            gamma_wall=cbf_gamma_wall,
            gamma_wall_1=cbf_gamma_wall_1,
            gamma_wall_2=cbf_gamma_wall_2,
            lookahead_tau=rospy.get_param(
                "~cbf_lookahead_tau", _CFG.get("cbf_lookahead_tau", 0.30)),
            w_path=rospy.get_param(
                "~w_path", _CFG.get("w_path", 1.0)),
            w_track=rospy.get_param(
                "~w_track",
                _CFG.get("w_track", 5.0)),
            w_formation=rospy.get_param(
                "~w_formation",
                _CFG.get("w_formation", 0.0)),
            w_reg=rospy.get_param(
                "~w_reg", _CFG.get("w_reg", 0.0)),
            w_accel=rospy.get_param(
                "~w_accel", _CFG.get("w_accel", 0.5)),
            max_vx=rospy.get_param("~max_vx", _CFG.get("max_vx", 0.55)),
            max_vy=rospy.get_param("~max_vy", _CFG.get("max_vy", 0.35)),
            max_ax=rospy.get_param("~max_ax", _CFG.get("max_ax", 1.0)),
            max_ay=rospy.get_param("~max_ay", _CFG.get("max_ay", 1.0)),
            footprint_half_length=rospy.get_param(
                "~robot_footprint_half_length",
                _CFG.get("robot_footprint_half_length", 0.35)),
            footprint_half_width=rospy.get_param(
                "~robot_footprint_half_width",
                _CFG.get("robot_footprint_half_width", 0.20)),
            footprint_drift_margin=rospy.get_param(
                "~robot_footprint_drift_margin",
                _CFG.get("robot_footprint_drift_margin", 0.08)),
            prediction_enabled=rospy.get_param(
                "~prediction_enabled",
                _CFG.get("prediction_enabled", False)),
            prediction_horizon=rospy.get_param(
                "~prediction_horizon",
                _CFG.get("prediction_horizon",
                         _CFG.get("horizon_N", 1))),
            prediction_dt=rospy.get_param(
                "~prediction_dt",
                _CFG.get("prediction_dt", 0.0)),
            w_smooth=rospy.get_param(
                "~w_smooth",
                _CFG.get("w_smooth", 0.2)),
            laplacian_ref=self.laplacian,
            K_accel=rospy.get_param(
                "~K_accel", _CFG.get("K_accel", 4.0)),
            Kd_accel=rospy.get_param(
                "~Kd_accel", _CFG.get("Kd_accel", 2.0)),
            emergency_brake_time=rospy.get_param(
                "~emergency_brake_time",
                _CFG.get("emergency_brake_time", 0.20)),
            slack_lambda=rospy.get_param(
                "~slack_lambda", _CFG.get("slack_lambda", 1e4)),
            slack_warn_threshold=rospy.get_param(
                "~slack_warn_threshold",
                _CFG.get("slack_warn_threshold", 0.05)),
            slack_enabled=rospy.get_param(
                "~cbf_slack_enabled", _CFG.get("cbf_slack_enabled", True)),
            gamma_vel=rospy.get_param(
                "~cbf_gamma_vel", _CFG.get("cbf_gamma_vel", 0.0)),
            gamma_vel_pair=rospy.get_param(
                "~cbf_gamma_vel_pair", _CFG.get("cbf_gamma_vel_pair", 0.0)),
        )
        if obstacles:
            self.qp.set_obstacles(obstacles)
        self.forbidden_zone_d_safe = float(rospy.get_param(
            "~forbidden_zone_d_safe",
            _CFG.get("forbidden_zone_d_safe", 0.15)))
        virtual_forbidden_rects = self._forbidden_zones_to_rect_obstacles(
            forbidden_zones, self.forbidden_zone_d_safe)
        qp_rect_obstacles = list(rect_obstacles or []) + virtual_forbidden_rects
        if qp_rect_obstacles:
            self.qp.set_rect_obstacles(qp_rect_obstacles)
        walls = rospy.get_param("~walls", _CFG.get("walls", []))
        if walls:
            self.qp.set_walls(walls)

        # ── Module 5: VelocityLimiter ──
        self.limiter = VelocityLimiter(
            max_vx=rospy.get_param("~max_vx", _CFG.get("max_vx", 0.55)),
            max_vy=rospy.get_param("~max_vy", _CFG.get("max_vy", 0.35)),
            max_wz=rospy.get_param("~max_wz", _CFG.get("max_wz", 0.8)),
        )

        # ── Module 6: CmdVelPublisher ──
        self.cmd_pub = CmdVelPublisher(self.all_dogs)
        self.accel_pub = AccelCmdPublisher(
            self.all_dogs,
            scale=rospy.get_param(
                "~accel_injection_scale",
                _CFG.get("accel_injection_scale", 1.0)),
        )
        # CBF 監看 topic（/cbf_debug/*，用 rqt_plot 即時看 h / slack / 速度落差）
        self.cbf_debug_pub = CbfDebugPublisher(self.all_dogs)

        # ── Debug topics for RViz / external visualizers ──
        self.debug_publish_enabled = bool(rospy.get_param(
            "~debug_publish_enabled", _CFG.get("debug_publish_enabled", True)))
        self.debug_frame_id = rospy.get_param(
            "~debug_frame_id", _CFG.get("debug_frame_id", "map"))
        self._debug_path_pubs = {
            name: rospy.Publisher(
                "/formation/%s_astar_path" % name,
                Path,
                queue_size=1,
                latch=True,
            )
            for name in self.all_dogs
        }
        self._debug_projected_goals_pub = rospy.Publisher(
            "/formation/projected_goals", PoseArray, queue_size=1, latch=True)
        self._debug_formation_pub = rospy.Publisher(
            "/formation/current_formation", String, queue_size=1, latch=True)
        self.debug_prediction_enabled = bool(rospy.get_param(
            "~debug_prediction_enabled",
            _CFG.get("debug_prediction_enabled", True)))
        self._debug_prediction_pub = rospy.Publisher(
            "/formation/prediction_markers",
            MarkerArray,
            queue_size=1,
        )

        # ── Follower yaw tracking ──
        self.kp_yaw_follower = float(rospy.get_param(
            "~kp_yaw", _CFG.get("kp_yaw", 0.6)))
        self.kp_pos_follower = float(rospy.get_param(
            "~kp_pos_follower", _CFG.get("kp_pos_follower", 0.8)))
        self.target_projection_margin = float(rospy.get_param(
            "~target_projection_margin", _CFG.get("target_projection_margin", 0.10)))
        self.target_projection_max_shift = float(rospy.get_param(
            "~target_projection_max_shift", _CFG.get("target_projection_max_shift", 0.45)))
        self.formation_guard_slow_error = float(rospy.get_param(
            "~formation_guard_slow_error", _CFG.get("formation_guard_slow_error", 0.65)))
        self.formation_guard_stop_error = float(rospy.get_param(
            "~formation_guard_stop_error", _CFG.get("formation_guard_stop_error", 1.10)))
        self.formation_guard_min_scale = float(rospy.get_param(
            "~formation_guard_min_scale", _CFG.get("formation_guard_min_scale", 0.15)))
        self.dynamic_slot_assignment = bool(rospy.get_param(
            "~dynamic_slot_assignment", _CFG.get("dynamic_slot_assignment", True)))
        self.slot_switch_hysteresis = float(rospy.get_param(
            "~slot_switch_hysteresis", _CFG.get("slot_switch_hysteresis", 0.15)))
        self.slot_switch_cooldown_seconds = float(rospy.get_param(
            "~slot_switch_cooldown_seconds", _CFG.get("slot_switch_cooldown_seconds", 1.0)))
        self.slot_freeze_projection_threshold = float(rospy.get_param(
            "~slot_freeze_projection_threshold", _CFG.get("slot_freeze_projection_threshold", 0.08)))
        self._slot_switch_cooldown_cycles = max(
            0, int(round(self.slot_switch_cooldown_seconds * self.rate_hz)))
        self._slot_switch_cooldown_counter = 0
        self._last_max_projection_shift = 0.0
        self._slot_assignment = None
        self._slot_assignment_formation = None

        # ── Per-dog A* front-end + soft formation（類 ZJU 架構的過渡版）──
        self.per_dog_astar_enabled = bool(rospy.get_param(
            "~per_dog_astar_enabled", _CFG.get("per_dog_astar_enabled", True)))
        self.per_dog_fail_stop = bool(rospy.get_param(
            "~per_dog_fail_stop", _CFG.get("per_dog_fail_stop", True)))
        self.per_dog_goal_tol = float(rospy.get_param(
            "~per_dog_goal_tol", _CFG.get("per_dog_goal_tol", 0.35)))
        self.per_dog_goal_unlatch_tol = float(rospy.get_param(
            "~per_dog_goal_unlatch_tol",
            _CFG.get("per_dog_goal_unlatch_tol",
                     self.per_dog_goal_tol + 0.12)))
        self.per_dog_goal_slow_radius = float(rospy.get_param(
            "~per_dog_goal_slow_radius",
            _CFG.get("per_dog_goal_slow_radius", 0.80)))
        self.per_dog_final_approach_radius = float(rospy.get_param(
            "~per_dog_final_approach_radius",
            _CFG.get("per_dog_final_approach_radius", 0.55)))
        self.per_dog_final_approach_kp = float(rospy.get_param(
            "~per_dog_final_approach_kp",
            _CFG.get("per_dog_final_approach_kp", 0.90)))
        self.per_dog_goal_projection_max_shift = float(rospy.get_param(
            "~per_dog_goal_projection_max_shift",
            _CFG.get("per_dog_goal_projection_max_shift",
                     self.astar_goal_adjust_max_dist)))
        self.per_dog_replan_interval = float(rospy.get_param(
            "~per_dog_replan_interval",
            _CFG.get("per_dog_replan_interval", 1.5)))
        self.per_dog_replan_accept_goal_shift = float(rospy.get_param(
            "~per_dog_replan_accept_goal_shift",
            _CFG.get("per_dog_replan_accept_goal_shift", 0.30)))
        self.per_dog_start_adjust_max_dist = float(rospy.get_param(
            "~per_dog_start_adjust_max_dist",
            _CFG.get("per_dog_start_adjust_max_dist", 0.45)))
        self.kp_formation_soft = float(rospy.get_param(
            "~kp_formation_soft", _CFG.get("kp_formation_soft", 0.20)))
        self.formation_soft_max_speed = float(rospy.get_param(
            "~formation_soft_max_speed", _CFG.get("formation_soft_max_speed", 0.12)))
        self._dog_paths = {name: [] for name in self.all_dogs}
        self._dog_path_goals = {name: None for name in self.all_dogs}
        self._dog_goal_latched = {name: False for name in self.all_dogs}
        self._dog_path_goal_key = None
        self._dog_path_assignment = None
        self._dog_path_goal_yaw = 0.0
        self._dog_path_last_plan_time = 0.0
        # nominal 速度時間平滑 (EMA) 的上一輪值與係數
        self._last_u_nominal = None
        self.nominal_smooth_alpha = float(rospy.get_param(
            "~nominal_smooth_alpha",
            _CFG.get("nominal_smooth_alpha", 0.4)))

        # ── Formation-aware A* planning margin ──
        self.astar_formation_margin_v = float(rospy.get_param(
            "~astar_formation_margin_v", _CFG.get("astar_formation_margin_v", self.astar.robot_radius)))
        self.astar_formation_margin_line = float(rospy.get_param(
            "~astar_formation_margin_line", _CFG.get("astar_formation_margin_line", self.astar.robot_radius)))
        self.astar_fallback_to_base_map = bool(rospy.get_param(
            "~astar_fallback_to_base_map", _CFG.get("astar_fallback_to_base_map", True)))

        # ── Stuck detection（保留）──
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
        self._consec_replans = 0
        self._last_formation_yaw = 0.0
        self.goal_hold_seconds = float(rospy.get_param(
            "~goal_hold_seconds", _CFG.get("goal_hold_seconds", 2.0)))
        self._goal_hold_counter = 0
        self.cmd_vel_raw_timeout = float(rospy.get_param(
            "~cmd_vel_raw_timeout", _CFG.get("cmd_vel_raw_timeout", 0.4)))
        self.cmd_vel_raw_deadband = float(rospy.get_param(
            "~cmd_vel_raw_deadband", _CFG.get("cmd_vel_raw_deadband", 1e-3)))
        self._idle_after_goal = False
        self._cmd_vel_bootstrap_key = None

        rospy.sleep(1.0)
        rospy.loginfo("=" * 65)
        rospy.loginfo("[FleetManagerUQP] READY  (Two-order CBF QP)")
        rospy.loginfo("  virtual_nav = formation centroid")
        rospy.loginfo("  dog order   = %s", self.all_dogs)
        rospy.loginfo("  goal_topic  = %s", self.goal_topic)
        rospy.loginfo("  cmd_raw     = %s", self.cmd_vel_raw_topic)
        rospy.loginfo("  navigator   = KEYBOARD by default")
        rospy.loginfo("    → publish %s to switch AUTO", self.goal_topic)
        rospy.loginfo("  A*          res=%.2fm, robot_r=%.2fm",
                      self.astar.res, self.astar.robot_radius)
        rospy.loginfo("  PurePursuit la=%.2fm, v=%.2fm/s",
                      self.pursuer.look_ahead, self.pursuer.v_cruise)
        rospy.loginfo("  formation   = '%s' (centroid-relative)",
                      self.laplacian.current_formation)
        rospy.loginfo("  door_switch = %s at x=%.1f (trigger=%.1fm, release=%.1fm)",
                      self._door_enabled, self.switcher.door_x,
                      self.switcher._trigger_dist, self.switcher._release_dist)
        rospy.loginfo(
            "  QP weights  tracking=%.1f, formation=%.2f, accel_reg=%.2f (vel_reg removed)",
            self.qp.w_track, self.qp.w_formation, self.qp.w_accel)
        rospy.loginfo(
            "  upper accel nominal PD: Kp=%.2f, Kd=%.2f",
            self.qp.K_accel, self.qp.Kd_accel)
        rospy.loginfo("  accel limit ax=%.2fm/s^2, ay=%.2fm/s^2",
                      self.qp.max_ax, self.qp.max_ay)
        rospy.loginfo("  prediction  requested=%s, active=%s, horizon=%d, dt=%.3fs",
                      self.qp.prediction_requested,
                      self.qp.prediction_enabled,
                      self.qp.prediction_horizon,
                      self.qp.prediction_dt)
        rospy.loginfo("  tracking    kp_pos=%.2f, kp_yaw=%.2f",
                      self.kp_pos_follower, self.kp_yaw_follower)
        rospy.loginfo("  per-dog A*  = %s (goal_tol=%.2f, fail_stop=%s)",
                      self.per_dog_astar_enabled,
                      self.per_dog_goal_tol,
                      self.per_dog_fail_stop)
        rospy.loginfo("    final approach radius=%.2f, kp=%.2f, unlatch=%.2f",
                      self.per_dog_final_approach_radius,
                      self.per_dog_final_approach_kp,
                      self.per_dog_goal_unlatch_tol)
        rospy.loginfo("    replan_interval=%.2fs, accept_goal_shift=%.2fm, start_adjust=%.2fm",
                      self.per_dog_replan_interval,
                      self.per_dog_replan_accept_goal_shift,
                      self.per_dog_start_adjust_max_dist)
        rospy.loginfo("  soft_form   kp=%.2f, max_speed=%.2fm/s",
                      self.kp_formation_soft,
                      self.formation_soft_max_speed)
        rospy.loginfo("  target_proj margin=%.2f, max_shift=%.2f",
                      self.target_projection_margin,
                      self.target_projection_max_shift)
        rospy.loginfo("  form_guard  slow=%.2f, stop=%.2f, min_scale=%.2f",
                      self.formation_guard_slow_error,
                      self.formation_guard_stop_error,
                      self.formation_guard_min_scale)
        rospy.loginfo("  slots       dynamic=%s, hysteresis=%.2f",
                      self.dynamic_slot_assignment, self.slot_switch_hysteresis)
        rospy.loginfo("    cooldown=%.1fs, freeze_proj=%.2f",
                      self.slot_switch_cooldown_seconds,
                      self.slot_freeze_projection_threshold)
        rospy.loginfo("  A* margin   V=%.2fm, line=%.2fm, fallback=%s",
                      self.astar_formation_margin_v,
                      self.astar_formation_margin_line,
                      self.astar_fallback_to_base_map)
        rospy.loginfo("    goal_adjust_max=%.2fm",
                      self.astar_goal_adjust_max_dist)
        rospy.loginfo("    adjusted_goal_clearance obs=%.2fm, wall=%.2fm",
                      self.astar_goal_min_obstacle_clearance,
                      self.astar_goal_min_wall_clearance)
        rospy.loginfo("  A* gaps     block=%s, width=%.2fm, lateral=%.2fm",
                      self.astar_block_narrow_wall_gaps,
                      self.astar_wall_gap_min_width,
                      self.astar_wall_gap_lateral_margin)
        rospy.loginfo("    obstacles=%s", self.astar_wall_gap_obstacle_indices)
        rospy.loginfo("  goal_hold   %.1fs after AUTO goal reached",
                      self.goal_hold_seconds)
        rospy.loginfo("  idle_after_goal until new goal or raw cmd "
                      "(timeout=%.2fs, deadband=%.4f)",
                      self.cmd_vel_raw_timeout, self.cmd_vel_raw_deadband)
        rospy.loginfo(
            "  HOCBF       = %s (robot γ=(%.2f, %.2f), obs γ=(%.2f, %.2f), wall γ=(%.2f, %.2f))",
            self.cbf_enabled,
            self.qp.gamma_robot_1, self.qp.gamma_robot_2,
            self.qp.gamma_obs_1, self.qp.gamma_obs_2,
            self.qp.gamma_wall_1, self.qp.gamma_wall_2)
        rospy.loginfo("    d_min=%.2f, slack=%s (λ=%.0f, warn>%.2f), τ=%.2fs",
                      self.qp.d_min,
                      "ON(soft)" if self.qp.slack_enabled else "OFF(HARD-verify)",
                      self.qp.slack_lambda,
                      self.qp.slack_warn_threshold, self.qp.lookahead_tau)
        rospy.loginfo("  obstacles   = %d,  walls = %d,  rects = %d",
                      len(self.qp.obstacles), len(self.qp.walls),
                      len(self.qp.rect_obstacles))
        rospy.loginfo("  A* forbidden zones = %d, QP virtual forbidden rects = %d (d_safe=%.2f)",
                      len(self.astar.forbidden_zones),
                      len(virtual_forbidden_rects),
                      self.forbidden_zone_d_safe)
        rospy.loginfo("    forbidden_zones_enabled = %s",
                      self.forbidden_zones_enabled)
        rospy.loginfo("  rate        = %.0f Hz", self.rate_hz)
        rospy.loginfo("=" * 65)

    def _formation_heading(self, center_state, center_nom_world):
        goal = self.navigator.tracking_goal
        if goal is not None:
            to_goal = np.array([goal[0] - center_state.x, goal[1] - center_state.y])
            if float(np.linalg.norm(to_goal)) > 0.25:
                return math.atan2(to_goal[1], to_goal[0])
        if float(np.linalg.norm(center_nom_world)) > 0.05:
            return math.atan2(center_nom_world[1], center_nom_world[0])
        return center_state.yaw

    def _formation_center_state(self, states):
        positions = np.array([states[name].pos for name in self.all_dogs], dtype=float)
        velocities = np.array([states[name].vel_world for name in self.all_dogs], dtype=float)
        center = np.mean(positions, axis=0)
        velocity = np.mean(velocities, axis=0)

        center_state = RobotState()
        center_state.x = float(center[0])
        center_state.y = float(center[1])
        if float(np.linalg.norm(velocity)) > 0.05:
            center_state.yaw = math.atan2(velocity[1], velocity[0])
        else:
            center_state.yaw = states[self.leader_name].yaw
        center_state.vx_world = float(velocity[0])
        center_state.vy_world = float(velocity[1])
        center_state.received = True
        return center_state

    def _update_astar_planning_margin(self):
        formation = self.laplacian.current_formation
        if formation == "line":
            margin = self.astar_formation_margin_line
        else:
            margin = self.astar_formation_margin_v
        self.astar.set_planning_margin(
            margin,
            label=formation or "formation",
            fallback_to_base=self.astar_fallback_to_base_map,
        )

    def _should_use_door_recovery(self, center_state):
        goal = self.navigator.current_goal
        if goal is None:
            return False
        door_x = self.switcher.door_x
        center_side = -1 if center_state.x < door_x - 0.15 else (
            1 if center_state.x > door_x + 0.15 else 0)
        goal_side = -1 if goal[0] < door_x - 0.15 else (
            1 if goal[0] > door_x + 0.15 else 0)
        if center_side == 0:
            return True
        needs_crossing = (goal_side != 0 and center_side != goal_side)
        if not needs_crossing:
            return False
        return abs(center_state.x - door_x) < (self.switcher._trigger_dist + 0.8)

    def _assignment_cost(self, positions, target_positions, assignment):
        cost = 0.0
        for dog_idx, slot_idx in enumerate(assignment):
            err = positions[dog_idx] - target_positions[slot_idx]
            cost += float(err @ err)
        return cost

    def _best_slot_assignment(self, positions, target_positions):
        n_dogs = len(self.all_dogs)
        identity = tuple(range(n_dogs))
        if not self.dynamic_slot_assignment:
            return identity, self._assignment_cost(
                positions, target_positions, identity)

        best_assignment = identity
        best_cost = float("inf")
        for assignment in itertools.permutations(range(n_dogs)):
            cost = self._assignment_cost(positions, target_positions, assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
        return best_assignment, best_cost

    def _select_slot_assignment(self, positions, target_positions):
        n_dogs = len(self.all_dogs)
        identity = tuple(range(n_dogs))
        formation_name = self.laplacian.current_formation

        if not self.dynamic_slot_assignment:
            self._slot_assignment = identity
            self._slot_assignment_formation = formation_name
            return identity

        best_assignment, best_cost = self._best_slot_assignment(
            positions, target_positions)

        reset_assignment = (
            self._slot_assignment is None
            or self._slot_assignment_formation != formation_name
        )
        if reset_assignment:
            self._slot_assignment = best_assignment
            self._slot_assignment_formation = formation_name
            self._slot_switch_cooldown_counter = self._slot_switch_cooldown_cycles
            rospy.loginfo("[FleetManagerUQP] slot assignment (%s): %s",
                          formation_name, self._format_slot_assignment(best_assignment))
            return best_assignment

        if self._slot_switch_cooldown_counter > 0:
            return self._slot_assignment

        if self._last_max_projection_shift >= self.slot_freeze_projection_threshold:
            return self._slot_assignment

        current_cost = self._assignment_cost(
            positions, target_positions, self._slot_assignment)
        if best_cost + self.slot_switch_hysteresis < current_cost:
            self._slot_assignment = best_assignment
            self._slot_switch_cooldown_counter = self._slot_switch_cooldown_cycles
            rospy.loginfo("[FleetManagerUQP] slot assignment (%s): %s",
                          formation_name, self._format_slot_assignment(best_assignment))

        return self._slot_assignment

    def _format_slot_assignment(self, assignment):
        return ", ".join(
            "%s->slot%d" % (name, assignment[idx])
            for idx, name in enumerate(self.all_dogs)
        )

    def _formation_guard_scale(self, max_slot_error):
        slow = max(0.0, self.formation_guard_slow_error)
        stop = max(slow + 1e-3, self.formation_guard_stop_error)
        min_scale = max(0.0, min(1.0, self.formation_guard_min_scale))

        if max_slot_error <= slow:
            return 1.0
        if max_slot_error >= stop:
            return min_scale

        alpha = (max_slot_error - slow) / (stop - slow)
        return 1.0 - alpha * (1.0 - min_scale)

    def _cap_projection_shift(self, original, projected, max_shift=None):
        if max_shift is None:
            max_shift = self.target_projection_max_shift
        max_shift = max(0.0, float(max_shift))
        delta = projected - original
        shift = float(np.linalg.norm(delta))
        if shift <= max_shift or shift < 1e-9:
            return projected
        return original + delta * (max_shift / shift)

    def _rect_clearance_correction(self, point, rect, min_dist):
        center = np.array(rect["center"][:2], dtype=float)
        size = np.array(rect["size"][:2], dtype=float)
        half = 0.5 * size
        closest = closest_point_on_aabb(point, center, size)
        dp = point - closest
        dist = float(np.linalg.norm(dp))

        if dist > 1e-6:
            if dist >= min_dist:
                return np.zeros(2), False
            return dp / dist * (min_dist - dist), True

        escape = rect.get("escape_dir", None)
        if escape is not None:
            direction = np.array(escape[:2], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                direction = direction / norm
                if abs(direction[0]) >= abs(direction[1]):
                    surface = center[0] + math.copysign(half[0], direction[0])
                    dist_to_surface = abs(surface - point[0])
                else:
                    surface = center[1] + math.copysign(half[1], direction[1])
                    dist_to_surface = abs(surface - point[1])
                return direction * (max(0.0, dist_to_surface) + min_dist), True

        rel = point - center
        clearances = np.array([
            half[0] - rel[0],
            half[0] + rel[0],
            half[1] - rel[1],
            half[1] + rel[1],
        ])
        side = int(np.argmin(clearances))
        if side == 0:
            direction = np.array([1.0, 0.0])
            dist_to_surface = clearances[side]
        elif side == 1:
            direction = np.array([-1.0, 0.0])
            dist_to_surface = clearances[side]
        elif side == 2:
            direction = np.array([0.0, 1.0])
            dist_to_surface = clearances[side]
        else:
            direction = np.array([0.0, -1.0])
            dist_to_surface = clearances[side]

        return direction * (max(0.0, dist_to_surface) + min_dist), True

    def _project_formation_target(self, target_pos, max_shift=None):
        original = np.array(target_pos, dtype=float)
        projected = original.copy()
        active = set()
        margin = max(0.0, self.target_projection_margin)

        # A few sequential passes handle corners where several safety bands overlap.
        for _ in range(3):
            before = projected.copy()

            for obs_idx, obs in enumerate(self.qp.obstacles):
                center = np.array(obs["pos"][:2], dtype=float)
                min_dist = float(obs.get("d_safe", obs["radius"])) + margin
                dp = projected - center
                dist = float(np.linalg.norm(dp))
                if dist >= min_dist:
                    continue
                if dist < 1e-6:
                    dp = original - center
                    dist = float(np.linalg.norm(dp))
                if dist < 1e-6:
                    dp = np.array([1.0, 0.0])
                    dist = 1.0
                projected = projected + dp / dist * (min_dist - dist)
                projected = self._cap_projection_shift(original, projected, max_shift)
                active.add("obs%d" % obs_idx)

            for rect_idx, rect in enumerate(self.qp.rect_obstacles):
                min_dist = float(rect.get("d_safe", 0.35)) + margin
                correction, is_active = self._rect_clearance_correction(
                    projected, rect, min_dist)
                if is_active:
                    projected = projected + correction
                    projected = self._cap_projection_shift(original, projected, max_shift)
                    active.add("rect%d" % rect_idx)

            for wall_idx, wall in enumerate(self.qp.walls):
                normal = np.array(wall["normal"][:2], dtype=float)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-9:
                    continue
                normal = normal / norm
                point = np.array(wall["point"][:2], dtype=float)
                min_dist = float(wall.get("d_safe", 0.4)) + margin
                clearance = float(normal @ (projected - point))
                if clearance < min_dist:
                    projected = projected + normal * (min_dist - clearance)
                    projected = self._cap_projection_shift(original, projected, max_shift)
                    active.add("wall%d" % wall_idx)

            if float(np.linalg.norm(projected - before)) < 1e-4:
                break

        shift = float(np.linalg.norm(projected - original))
        return projected, shift, sorted(active)

    @staticmethod
    def _limit_vector_norm(vec, max_norm):
        max_norm = max(0.0, float(max_norm))
        vec = np.array(vec, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm <= max_norm or norm < 1e-9:
            return vec
        return vec * (max_norm / norm)

    def _per_dog_auto_active(self):
        return (self.per_dog_astar_enabled
                and self.navigator.current_mode == LeaderNavigator.MODE_AUTO
                and self.navigator.has_goal)

    def _reset_per_dog_paths(self, reason=""):
        self._dog_paths = {name: [] for name in self.all_dogs}
        self._dog_path_goals = {name: None for name in self.all_dogs}
        self._dog_goal_latched = {name: False for name in self.all_dogs}
        self._dog_path_goal_key = None
        self._dog_path_assignment = None
        self._dog_path_last_plan_time = 0.0
        if hasattr(self, "qp"):
            self.qp.reset_prediction(reason or "per-dog path reset")
        if reason:
            rospy.loginfo("[PerDogA*] reset paths: %s", reason)

    def _goal_formation_yaw(self, center_state, goal):
        delta = np.array([goal[0] - center_state.x, goal[1] - center_state.y])
        if float(np.linalg.norm(delta)) > 0.25:
            return math.atan2(delta[1], delta[0])
        return self._last_formation_yaw

    def _goal_slot_targets(self, goal, center_state):
        offsets = self.laplacian.current_offsets
        if offsets is None or len(offsets) != len(self.all_dogs):
            return None, None
        goal_yaw = self._goal_formation_yaw(center_state, goal)
        R_goal = rot2d(goal_yaw)
        goal_center = np.array(goal, dtype=float)
        targets = [goal_center + R_goal @ offset for offset in offsets]
        return targets, goal_yaw

    def _ensure_per_dog_paths(self, states, center_state):
        goal = self.navigator.current_goal
        if goal is None:
            self._reset_per_dog_paths("AUTO goal cleared")
            return False

        formation_name = self.laplacian.current_formation
        goal_key = (
            round(float(goal[0]), 3),
            round(float(goal[1]), 3),
            formation_name,
        )
        paths_missing = any(not self._dog_paths.get(name)
                            for name in self.all_dogs)
        cached_paths_valid = (
            self._dog_path_goal_key == goal_key and not paths_missing)
        now = rospy.get_time()
        replan_interval = max(0.0, float(self.per_dog_replan_interval))
        periodic_replan_due = (
            cached_paths_valid
            and (
                replan_interval <= 1e-6
                or now - self._dog_path_last_plan_time >= replan_interval
            )
        )
        if cached_paths_valid and not periodic_replan_due:
            return True
        if periodic_replan_due:
            rospy.loginfo_throttle(
                2.0,
                "[PerDogA*] periodic replan after %.2fs",
                now - self._dog_path_last_plan_time,
            )

        def keep_cached_or_fail(message, *args):
            if periodic_replan_due:
                self._dog_path_last_plan_time = now
                rospy.logwarn_throttle(1.0, message + "; keeping cached path",
                                       *args)
                return True
            rospy.logwarn(message, *args)
            return False

        target_positions, goal_yaw = self._goal_slot_targets(goal, center_state)
        if target_positions is None:
            return keep_cached_or_fail(
                "[PerDogA*] invalid formation offsets; cannot plan per-dog paths",
            )

        if periodic_replan_due and self._dog_path_assignment is not None:
            assignment = self._dog_path_assignment
        else:
            dog_positions = [states[name].pos for name in self.all_dogs]
            assignment, _ = self._best_slot_assignment(
                dog_positions, target_positions)

        new_paths = {}
        new_goals = {}
        old_goals = dict(self._dog_path_goals)
        old_latched = dict(self._dog_goal_latched)
        for idx, name in enumerate(self.all_dogs):
            raw_goal = target_positions[assignment[idx]]
            projected_goal, goal_shift, active = self._project_formation_target(
                raw_goal,
                max_shift=self.per_dog_goal_projection_max_shift,
            )
            if goal_shift > 1e-3:
                rospy.loginfo(
                    "[PerDogA*] %s final slot projected %.2fm (%s)",
                    name, goal_shift, ",".join(active) or "clear",
            )
            safe_start = self.astar.nearest_free(
                tuple(states[name].pos),
                max_dist=self.per_dog_start_adjust_max_dist,
                label="%s start" % name,
            )
            if safe_start is None:
                return keep_cached_or_fail(
                    "[PerDogA*] %s start has no nearby free cell", name)
            safe_goal, path = self.astar.find_reachable_goal(
                safe_start, tuple(projected_goal),
                max_dist=self.astar_goal_adjust_max_dist)
            if safe_goal is None or not path:
                return keep_cached_or_fail(
                    "[PerDogA*] %s failed path to slot%d goal=(%.2f,%.2f)",
                    name, assignment[idx], projected_goal[0], projected_goal[1])
            new_paths[name] = path
            new_goals[name] = np.array(safe_goal, dtype=float)

        if periodic_replan_due:
            max_goal_jump = 0.0
            for name in self.all_dogs:
                old_goal = old_goals.get(name)
                if old_goal is None:
                    continue
                max_goal_jump = max(
                    max_goal_jump,
                    float(np.linalg.norm(new_goals[name] - old_goal)),
                )
            accept_shift = max(
                0.0, float(self.per_dog_replan_accept_goal_shift))
            if max_goal_jump > accept_shift:
                self._dog_path_last_plan_time = now
                rospy.logwarn_throttle(
                    1.0,
                    "[PerDogA*] reject periodic replan: goal jump %.2fm > %.2fm; keeping cached path",
                    max_goal_jump,
                    accept_shift,
                )
                return True

        self._dog_paths = new_paths
        self._dog_path_goals = new_goals
        self._dog_goal_latched = {}
        for name in self.all_dogs:
            old_goal = old_goals.get(name)
            keep_latched = (
                periodic_replan_due
                and old_latched.get(name, False)
                and old_goal is not None
                and float(np.linalg.norm(new_goals[name] - old_goal))
                <= self.per_dog_goal_unlatch_tol
            )
            self._dog_goal_latched[name] = bool(keep_latched)
        self._dog_path_goal_key = goal_key
        self._dog_path_assignment = assignment
        self._dog_path_goal_yaw = goal_yaw
        self._dog_path_last_plan_time = now
        rospy.loginfo(
            "[PerDogA*] planned %s | %s",
            formation_name,
            self._format_slot_assignment(assignment),
        )
        for name in self.all_dogs:
            rospy.loginfo(
                "[PerDogA*]   %s path=%d goal=(%.2f,%.2f)",
                name, len(self._dog_paths[name]),
                self._dog_path_goals[name][0],
                self._dog_path_goals[name][1],
            )
        return True

    def _per_dog_goals_reached(self, states):
        if not self._per_dog_auto_active():
            return False
        if all(self._dog_goal_latched.get(name, False)
               for name in self.all_dogs):
            return True
        for name in self.all_dogs:
            goal = self._dog_path_goals.get(name)
            if goal is None:
                return False
            if float(np.linalg.norm(states[name].pos - goal)) > self.per_dog_goal_tol:
                return False
        return True

    def _obstacle_approach_scale(self, pos, vel_world):
        """接近障礙/牆時的 nominal 減速係數 ∈ [min_scale, 1.0]。

        只懲罰「朝障礙方向的前進」: 若 nominal 速度方向指向某個障礙/牆,
        且很近, 就把速度壓低, 讓狗溫和靠近並停, 而不是全速撞上被 CBF 彈回
        (造成前後搖晃)。側向通過 / 遠離障礙不受影響。

        slow_dist: 開始減速的距離 (m); stop_dist: 壓到 min_scale 的距離。
        距離用「沿前進方向的有號接近距離」: 障礙在正前方才減速。
        """
        speed = float(np.linalg.norm(vel_world))
        if speed < 1e-6:
            return 1.0
        heading = vel_world / speed
        slow_dist = float(getattr(self, "approach_slow_dist", 1.2))
        stop_dist = float(getattr(self, "approach_stop_dist", 0.5))
        min_scale = float(getattr(self, "approach_min_scale", 0.12))
        scale = 1.0

        def _apply(surface_dist):
            # surface_dist: 狗中心到障礙安全邊界的距離 (已扣 radius/d_safe)
            if surface_dist >= slow_dist:
                return 1.0
            if surface_dist <= stop_dist:
                return min_scale
            t = (surface_dist - stop_dist) / max(1e-6, slow_dist - stop_dist)
            return min_scale + (1.0 - min_scale) * t

        # 圓形障礙
        for obs in self.qp.obstacles:
            p_obs = np.array(obs["pos"][:2], dtype=float)
            r = max(float(obs["radius"]),
                    float(obs.get("physical_radius", 0.2))
                    + self.qp._footprint_support_along(pos - p_obs, 0.0))
            d = pos - p_obs
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                continue
            # 只在「朝障礙前進」時減速 (heading 與 d 反向 → 朝障礙)
            approaching = float(heading @ (d / dist)) < -0.2
            if approaching:
                scale = min(scale, _apply(dist - r))

        # 牆
        for wall in self.qp.walls:
            n_w = np.array(wall["normal"][:2], dtype=float)
            p_w = np.array(wall["point"][:2], dtype=float)
            d_safe = max(float(wall.get("d_safe", 0.4)),
                         self.qp._footprint_support_along(n_w, 0.0))
            signed = float(n_w @ (pos - p_w)) - d_safe
            # 朝牆前進 (heading 與 n_w 反向)
            if float(heading @ n_w) < -0.2:
                scale = min(scale, _apply(signed))

        # rect (門牆)
        for rect in self.qp.rect_obstacles:
            center = np.array(rect["center"][:2], dtype=float)
            size = np.array(rect["size"][:2], dtype=float)
            d_safe = float(rect.get("d_safe", 0.35))
            closest = closest_point_on_aabb(pos, center, size)
            d = pos - closest
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                continue
            approaching = float(heading @ (d / dist)) < -0.2
            if approaching:
                scale = min(scale, _apply(dist - d_safe))

        return max(min_scale, scale)

    def _build_per_dog_nominal_velocity(self, states, center_state,
                                        formation_grad=None):
        n_vars = 2 * len(self.all_dogs)
        u_nominal = np.zeros(n_vars)
        if not self._ensure_per_dog_paths(states, center_state):
            return u_nominal, {
                "guard_scale": 0.0 if self.per_dog_fail_stop else 1.0,
                "max_slot_error": 0.0,
                "max_projection_shift": 0.0,
                "projected_targets": ["per-dog A* failed"],
                "path_mode": True,
                "path_ready": False,
                "max_goal_error": 0.0,
                "max_path_speed": 0.0,
                "max_form_speed": 0.0,
            }

        if formation_grad is None or len(formation_grad) != len(self.all_dogs):
            formation_grad = [np.zeros(2) for _ in self.all_dogs]

        path_speeds = []
        formation_speeds = []
        goal_errors = []

        for idx, name in enumerate(self.all_dogs):
            s = states[name]
            path = self._dog_paths.get(name, [])
            goal = self._dog_path_goals.get(name)
            goal_error = None
            if goal is not None:
                goal_error = float(np.linalg.norm(s.pos - goal))
                goal_errors.append(goal_error)
                if (self._dog_goal_latched.get(name, False)
                        and goal_error > self.per_dog_goal_unlatch_tol):
                    rospy.loginfo(
                        "[PerDogA*] %s re-acquire goal (err=%.2fm)",
                        name, goal_error)
                    self._dog_goal_latched[name] = False
                elif goal_error <= self.per_dog_goal_tol:
                    if not self._dog_goal_latched.get(name, False):
                        rospy.loginfo(
                            "[PerDogA*] %s settled at projected goal (err=%.2fm)",
                            name, goal_error)
                    self._dog_goal_latched[name] = True

            if self._dog_goal_latched.get(name, False):
                path_speeds.append(0.0)
                formation_speeds.append(0.0)
                continue

            if (goal is not None
                    and goal_error is not None
                    and goal_error <= self.per_dog_final_approach_radius):
                v_path_world = self.per_dog_final_approach_kp * (goal - s.pos)
            else:
                v_path_body, _ = self.dog_pursuers[name].compute(s, path)
                v_path_world = rot2d(s.yaw) @ np.array(v_path_body, dtype=float)
            if goal_error is not None:
                slow_radius = max(self.per_dog_goal_tol + 1e-3,
                                  self.per_dog_goal_slow_radius)
                speed_scale = min(1.0, max(0.0, goal_error / slow_radius))
                v_path_world *= speed_scale

            if self.qp.w_formation > 1e-9:
                # Direct Laplacian formation descent is handled in the QP cost.
                v_form_world = np.zeros(2)
            else:
                v_form_world = -self.kp_formation_soft * np.array(
                    formation_grad[idx], dtype=float)
                if goal_error is not None:
                    denom = max(1e-3, self.per_dog_goal_slow_radius - self.per_dog_goal_tol)
                    form_scale = min(1.0, max(0.0,
                                              (goal_error - self.per_dog_goal_tol) / denom))
                    v_form_world *= form_scale
                v_form_world = self._limit_vector_norm(
                    v_form_world, self.formation_soft_max_speed)
            formation_speeds.append(float(np.linalg.norm(v_form_world)))

            v_world = v_path_world + v_form_world
            # 接近障礙/牆時主動減速: 避免 nominal 全速硬推向過不去的地方,
            # 與 CBF 形成「推-擋-推-擋」的前後搖晃。讓狗溫和靠近並停穩。
            approach_scale = self._obstacle_approach_scale(s.pos, v_world)
            v_world = v_world * approach_scale
            v_body = rot2d(s.yaw).T @ v_world
            u_nominal[2 * idx:2 * idx + 2] = v_body
            path_speeds.append(float(np.linalg.norm(v_path_world)))

        # ── nominal 速度時間平滑 (EMA 低通) ──
        # 每 cycle 重算的 path/formation/approach 合成速度容易在方向/大小上突跳,
        # 造成單狗搖晃、yaw 抖動、側走。用指數移動平均把相鄰 cycle 接起來:
        #   u_smoothed = (1-a) * u_prev + a * u_new
        # alpha 越小越平滑但越鈍; 0.3~0.5 兼顧平滑與反應。
        smooth_alpha = float(getattr(self, "nominal_smooth_alpha", 0.4))
        if (getattr(self, "_last_u_nominal", None) is not None
                and len(self._last_u_nominal) == len(u_nominal)):
            u_nominal = ((1.0 - smooth_alpha) * self._last_u_nominal
                         + smooth_alpha * u_nominal)
        self._last_u_nominal = u_nominal.copy()

        mean_path_world = np.zeros(2)
        for idx, name in enumerate(self.all_dogs):
            s = states[name]
            v_body = u_nominal[2 * idx:2 * idx + 2]
            mean_path_world += rot2d(s.yaw) @ v_body
        mean_path_world /= max(1, len(self.all_dogs))
        if float(np.linalg.norm(mean_path_world)) > 0.05:
            self._last_formation_yaw = math.atan2(
                mean_path_world[1], mean_path_world[0])
        else:
            self._last_formation_yaw = self._dog_path_goal_yaw

        return u_nominal, {
            "guard_scale": 1.0,
            "max_slot_error": max(goal_errors) if goal_errors else 0.0,
            "max_projection_shift": max(formation_speeds) if formation_speeds else 0.0,
            "projected_targets": [
                "path_v=%.2f form_v=%.2f" % (
                    max(path_speeds) if path_speeds else 0.0,
                    max(formation_speeds) if formation_speeds else 0.0,
                )
            ],
            "path_mode": True,
            "path_ready": True,
            "max_goal_error": max(goal_errors) if goal_errors else 0.0,
            "max_path_speed": max(path_speeds) if path_speeds else 0.0,
            "max_form_speed": max(formation_speeds) if formation_speeds else 0.0,
        }

    def _build_nominal_velocity(self, states, center_state, center_nom,
                                formation_grad=None):
        if self._per_dog_auto_active():
            per_dog_u, per_dog_diag = self._build_per_dog_nominal_velocity(
                states, center_state, formation_grad=formation_grad)
            if per_dog_diag.get("path_ready", False) or self.per_dog_fail_stop:
                return per_dog_u, per_dog_diag

        n_vars = 2 * len(self.all_dogs)
        u_nominal = np.zeros(n_vars)
        center_nom = np.array(center_nom, dtype=float)
        center_nom_world = rot2d(center_state.yaw) @ center_nom

        offsets = self.laplacian.current_offsets
        if offsets is None or len(offsets) != len(self.all_dogs):
            rospy.logwarn_throttle(
                1.0,
                "[FleetManagerUQP] invalid formation offsets, followers hold position",
            )
            return u_nominal, {
                "guard_scale": 1.0,
                "max_slot_error": 0.0,
                "max_projection_shift": 0.0,
                "projected_targets": [],
            }

        formation_yaw = self._formation_heading(center_state, center_nom_world)
        self._last_formation_yaw = formation_yaw
        R_form = rot2d(formation_yaw)
        center_pos = center_state.pos
        target_positions = [center_pos + R_form @ offset for offset in offsets]
        dog_positions = [states[name].pos for name in self.all_dogs]
        assignment = self._select_slot_assignment(dog_positions, target_positions)
        slot_errors = [
            float(np.linalg.norm(target_positions[assignment[idx]] - dog_positions[idx]))
            for idx in range(len(self.all_dogs))
        ]
        max_slot_error = max(slot_errors) if slot_errors else 0.0
        guard_scale = self._formation_guard_scale(max_slot_error)
        guarded_center_nom_world = center_nom_world * guard_scale

        max_projection_shift = 0.0
        projected_targets = []

        for idx, name in enumerate(self.all_dogs):
            s = states[name]
            target_pos = target_positions[assignment[idx]]
            target_pos, projection_shift, active = self._project_formation_target(target_pos)
            max_projection_shift = max(max_projection_shift, projection_shift)
            if projection_shift > 1e-3:
                projected_targets.append("%s:%.2f/%s" % (
                    name, projection_shift, ",".join(active) or "clear"))
            pos_error = target_pos - s.pos
            v_world = guarded_center_nom_world + self.kp_pos_follower * pos_error
            v_body = rot2d(s.yaw).T @ v_world
            u_nominal[2 * idx:2 * idx + 2] = v_body

        return u_nominal, {
            "guard_scale": guard_scale,
            "max_slot_error": max_slot_error,
            "max_projection_shift": max_projection_shift,
            "projected_targets": projected_targets,
        }

    def _publish_debug_topics(self):
        if not self.debug_publish_enabled:
            return

        stamp = rospy.Time.now()
        for name in self.all_dogs:
            msg = Path()
            msg.header.stamp = stamp
            msg.header.frame_id = self.debug_frame_id
            for x, y in self._dog_paths.get(name, []):
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.03
                pose.pose.orientation.w = 1.0
                msg.poses.append(pose)
            self._debug_path_pubs[name].publish(msg)

        goals = PoseArray()
        goals.header.stamp = stamp
        goals.header.frame_id = self.debug_frame_id
        for name in self.all_dogs:
            goal = self._dog_path_goals.get(name)
            if goal is None:
                continue
            pose = PoseStamped().pose
            pose.position.x = float(goal[0])
            pose.position.y = float(goal[1])
            pose.position.z = 0.08
            pose.orientation.w = 1.0
            goals.poses.append(pose)
        self._debug_projected_goals_pub.publish(goals)
        formation_msg = String()
        formation_msg.data = self.laplacian.current_formation or ""
        self._debug_formation_pub.publish(formation_msg)

    @staticmethod
    def _debug_dog_color(idx, alpha=1.0):
        colors = [
            (1.0, 0.08, 0.05),
            (0.0, 0.85, 0.25),
            (0.1, 0.45, 1.0),
        ]
        r, g, b = colors[idx % len(colors)]
        return r, g, b, alpha

    def _prediction_delete_all_marker(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.debug_frame_id
        marker.action = Marker.DELETEALL
        return marker

    def _prediction_point(self, pos, z=0.12):
        point = Point()
        point.x = float(pos[0])
        point.y = float(pos[1])
        point.z = float(z)
        return point

    def _publish_prediction_markers(self):
        if not self.debug_publish_enabled or not self.debug_prediction_enabled:
            return

        msg = MarkerArray()
        msg.markers.append(self._prediction_delete_all_marker())
        paths = getattr(self.qp, "last_prediction_paths", {}) or {}
        if not paths:
            self._debug_prediction_pub.publish(msg)
            return

        stamp = rospy.Time.now()
        for idx, name in enumerate(self.all_dogs):
            points = paths.get(name, [])
            if len(points) < 2:
                continue
            r, g, b, a_line = self._debug_dog_color(idx, alpha=0.85)

            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = self.debug_frame_id
            line.ns = "prediction_solution_lines"
            line.id = idx
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.035
            line.color.r = r
            line.color.g = g
            line.color.b = b
            line.color.a = a_line
            line.points = [self._prediction_point(p, z=0.12) for p in points]
            msg.markers.append(line)

            beads = Marker()
            beads.header.stamp = stamp
            beads.header.frame_id = self.debug_frame_id
            beads.ns = "prediction_solution_points"
            beads.id = 100 + idx
            beads.type = Marker.SPHERE_LIST
            beads.action = Marker.ADD
            beads.pose.orientation.w = 1.0
            beads.scale.x = 0.12
            beads.scale.y = 0.12
            beads.scale.z = 0.12
            beads.color.r = r
            beads.color.g = g
            beads.color.b = b
            beads.color.a = 0.75
            beads.points = [self._prediction_point(p, z=0.14) for p in points]
            msg.markers.append(beads)

        self._debug_prediction_pub.publish(msg)

    def _publish_safety_hold(self, states):
        """Hold nominally still while still allowing CBF to push away from hazards."""
        u_nominal = np.zeros(2 * len(self.all_dogs))
        u_safe = self.qp.solve(
            self.all_dogs,
            states,
            u_nominal,
            cbf_enabled=self.cbf_enabled,
            dt=1.0 / max(1e-3, float(self.rate_hz)),
        )
        self._publish_prediction_markers()
        for name in self.all_dogs:
            accel = self.qp.last_accel_cmds.get(name, np.zeros(2))
            self.accel_pub.publish(name, accel[0], accel[1])
            vx, vy = u_safe[name]
            vx, vy, wz = self.limiter.clamp(vx, vy, 0.0)
            self.cmd_pub.publish(name, vx, vy, wz)

    def _bootstrap_is_safe_now(self, states):
        pair_margin = 0.15
        obstacle_margin = 0.15
        wall_margin = 0.12

        for ia in range(len(self.all_dogs)):
            for ib in range(ia + 1, len(self.all_dogs)):
                sa = states[self.all_dogs[ia]]
                sb = states[self.all_dogs[ib]]
                if not sa.received or not sb.received:
                    return False
                if float(np.linalg.norm(sa.pos - sb.pos)) < self.qp.d_min + pair_margin:
                    return False

        for name in self.all_dogs:
            s = states[name]
            if not s.received:
                return False

            for obs in self.qp.obstacles:
                center = np.array(obs["pos"][:2], dtype=float)
                radius = float(obs["radius"])
                if float(np.linalg.norm(s.pos - center)) < radius + obstacle_margin:
                    return False

            for rect in self.qp.rect_obstacles:
                center = np.array(rect["center"][:2], dtype=float)
                size = np.array(rect["size"][:2], dtype=float)
                d_safe = float(rect.get("d_safe", 0.35))
                closest = closest_point_on_aabb(s.pos, center, size)
                if float(np.linalg.norm(s.pos - closest)) < d_safe + obstacle_margin:
                    return False

            for wall in self.qp.walls:
                normal = np.array(wall["normal"][:2], dtype=float)
                point = np.array(wall["point"][:2], dtype=float)
                d_safe = max(
                    float(wall.get("d_safe", 0.4)),
                    self.qp._footprint_support_along(normal, s.yaw),
                )
                if float(normal @ (s.pos - point)) < d_safe + wall_margin:
                    return False

        return True

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        states = self.state_collector.states
        _last_logged_mode = [None]

        while not rospy.is_shutdown():
            # ── 0. 前置檢查：等待全部 robot 狀態，避免 centroid 被未初始化座標污染 ──
            if self.stop_without_leader and not self.state_collector.all_received(self.all_dogs):
                self.accel_pub.publish_zero(self.all_dogs)
                self.cmd_pub.publish_zero(self.all_dogs)
                rate.sleep()
                continue

            center_state = self._formation_center_state(states)
            self._update_astar_planning_margin()

            if self.navigator.unreachable_hold:
                if self.navigator.manual_command_active(
                        self.cmd_vel_raw_timeout, self.cmd_vel_raw_deadband):
                    self.navigator.clear_unreachable_hold("manual command")
                else:
                    self.accel_pub.publish_zero(self.all_dogs)
                    self.cmd_pub.publish_zero(self.all_dogs)
                    rate.sleep()
                    continue

            # ── 1. Virtual-center nominal (AUTO or KEYBOARD) ──
            mode_before = self.navigator.current_mode
            if (self.per_dog_astar_enabled
                    and mode_before == LeaderNavigator.MODE_AUTO
                    and self.navigator.has_goal):
                center_nom = (0.0, 0.0)
                if self._per_dog_goals_reached(states):
                    self.navigator.finish_auto_goal("per-dog goals")
                    self._reset_per_dog_paths("goal reached")
                mode_after = self.navigator.current_mode
            else:
                center_nom, _ = self.navigator.get_nominal(center_state)
                mode_after = self.navigator.current_mode
            if (mode_before == LeaderNavigator.MODE_AUTO
                    and mode_after == LeaderNavigator.MODE_KEYBOARD
                    and not self.navigator.has_goal):
                self._idle_after_goal = True
                self._goal_hold_counter = max(
                    1, int(round(self.goal_hold_seconds * self.rate_hz)))
                rospy.loginfo("[FleetManagerUQP] Goal hold for %.1fs",
                              self.goal_hold_seconds)

            if self._goal_hold_counter > 0 and not self.navigator.has_goal:
                self.accel_pub.publish_zero(self.all_dogs)
                self.cmd_pub.publish_zero(self.all_dogs)
                self._goal_hold_counter -= 1
                rate.sleep()
                continue
            if self.navigator.has_goal:
                self._goal_hold_counter = 0
                self._idle_after_goal = False
            elif self._dog_path_goal_key is not None:
                self._reset_per_dog_paths("no active AUTO goal")
                self._cmd_vel_bootstrap_key = None
            if (self._idle_after_goal
                    and not self.navigator.has_goal
                    and not self.navigator.manual_command_active(
                        self.cmd_vel_raw_timeout, self.cmd_vel_raw_deadband)):
                self.accel_pub.publish_zero(self.all_dogs)
                self.cmd_pub.publish_zero(self.all_dogs)
                rate.sleep()
                continue
            manual_command_active = self.navigator.manual_command_active(
                self.cmd_vel_raw_timeout, self.cmd_vel_raw_deadband)
            if (mode_after == LeaderNavigator.MODE_KEYBOARD
                    and not self.navigator.has_goal
                    and not manual_command_active):
                self.accel_pub.publish_zero(self.all_dogs)
                self.cmd_pub.publish_zero(self.all_dogs)
                rate.sleep()
                continue
            if manual_command_active:
                self._idle_after_goal = False

            mode = mode_after
            if mode != _last_logged_mode[0]:
                rospy.loginfo("[FleetManagerUQP] Virtual-center mode → %s", mode)
                _last_logged_mode[0] = mode

            # ── 2. FormationSwitcher: 偵測是否需要切換隊形 ──
            if self._door_enabled:
                formation_changed = self.switcher.update(
                    center_state, self.navigator.current_goal, states)
                if formation_changed:
                    self.qp.reset_prediction(
                        "formation switched to %s" %
                        self.laplacian.current_formation)
                if formation_changed and self._per_dog_auto_active():
                    self._reset_per_dog_paths(
                        "formation switched to %s" %
                        self.laplacian.current_formation)

            # ── 3. Formation diagnostics + nominal velocity ──
            positions = [states[name].pos.copy() for name in self.all_dogs]
            f_cost, formation_grad = self.laplacian.compute(positions)
            u_nominal, target_diag = self._build_nominal_velocity(
                states, center_state, center_nom,
                formation_grad=formation_grad)
            if target_diag.get("path_mode", False):
                self._last_max_projection_shift = 0.0
            else:
                self._last_max_projection_shift = target_diag["max_projection_shift"]
            self._publish_debug_topics()
            if (target_diag.get("path_mode", False)
                    and not target_diag.get("path_ready", True)
                    and self.per_dog_fail_stop):
                reason = "; ".join(target_diag.get("projected_targets", []))
                self.navigator.hold_unreachable_goal(
                    reason or "per-dog A* failed")
                self._reset_per_dog_paths(reason or "per-dog A* failed")
                self.accel_pub.publish_zero(self.all_dogs)
                self.cmd_pub.publish_zero(self.all_dogs)
                rate.sleep()
                continue

            # ── 4. yaw tracking：統一朝編隊前進方向 ──
            # (先前試過各自朝移動方向, 但三隻狗朝向各異看起來凌亂; 改回統一,
            #  三隻狗朝同一方向比較像一個整體編隊。)
            wz_all = {}
            for name in self.all_dogs:
                s = states[name]
                e_yaw = wrap_to_pi(self._last_formation_yaw - s.yaw)
                wz_all[name] = self.kp_yaw_follower * e_yaw

            # ── 5. 算 a_nom：上層二階 PD nominal acceleration，先 world 再轉 body ──
            # a_nom^W = K_accel (p_d - p) + Kd_accel (v_d - v)
            # a_nom^B = R(psi)^T a_nom^W
            #
            # 設計（與 _build_per_dog_nominal_velocity / u_nominal 完全一致）：
            #   v_d  ← 直接複用 u_nominal 的成品（已含 final-approach 切換、
            #          speed_scale 衰減、obstacle approach 減速、EMA 平滑），
            #          不在此重算，確保 a_nom 的期望速度與 u_nom 永遠同步。
            #   p_d  ← 兩段式（與 u_nom 同樣的 final_approach_radius 切換條件）：
            #            靠近 goal → p_d = 固定 slot goal（真正的平衡點，會歸零）
            #            趕路       → p_d = lookahead（沿 A* 路徑，給前進方向）
            #   latched → 純阻尼 a_nom = -Kd_accel·v（吃掉到點殘速 + trot 噪聲，
            #             乾淨停住；不再用 lookahead 持續往前推 → 根除拉扯）。
            #   latch 狀態只讀不重算（步驟 3 的 u_nom 已更新好，避免雙重判定）。
            #
            # 守衛：latched 是 per-dog AUTO 專屬狀態，只在 per-dog AUTO 模式才
            #       信任它。非 per-dog（keyboard/centroid）模式即使殘留舊 latch
            #       也不走純阻尼分支，避免與手動/centroid 速度指令打架。
            per_dog_active = self._per_dog_auto_active()
            a_desired = np.zeros(2 * len(self.all_dogs))
            for idx, name in enumerate(self.all_dogs):
                s = states[name]
                if not s.received:
                    continue

                # latched：純阻尼煞停（乙案），不施加前進期望
                if per_dog_active and self._dog_goal_latched.get(name, False):
                    a_d_world = -self.qp.Kd_accel * s.vel_world
                    a_d_body = rot2d(s.yaw).T @ a_d_world
                    a_desired[2 * idx:2 * idx + 2] = a_d_body
                    continue

                # v_d：直接取 u_nominal 成品（body frame）→ 轉 world
                v_d_body = u_nominal[2 * idx:2 * idx + 2]
                v_d_world = rot2d(s.yaw) @ np.array(v_d_body, dtype=float)

                # p_d：與 u_nom 相同的兩段式切換
                path = self._dog_paths.get(name, [])
                goal = self._dog_path_goals.get(name)
                goal_error = (float(np.linalg.norm(s.pos - goal))
                              if goal is not None else None)
                if (goal is not None
                        and goal_error is not None
                        and goal_error <= self.per_dog_final_approach_radius):
                    p_d = np.array(goal, dtype=float)          # 靠近：固定 goal
                elif path:
                    p_d = np.array(
                        self.dog_pursuers[name]._find_lookahead(s.pos, path),
                        dtype=float)                            # 趕路：lookahead
                else:
                    p_d = s.pos.copy()

                a_d_world = (
                    self.qp.K_accel * (p_d - s.pos)
                    + self.qp.Kd_accel * (v_d_world - s.vel_world))
                a_d_body = rot2d(s.yaw).T @ a_d_world
                a_desired[2 * idx:2 * idx + 2] = a_d_body

            # ── 6. 解 pure second-order HOCBF acceleration QP ──
            # cbf_enabled 只控制安全 constraints；formation/path objective 仍會保留。
            control_dt = 1.0 / max(1e-3, float(self.rate_hz))
            u_safe = self.qp.solve(
                self.all_dogs, states, u_nominal,
                a_desired=a_desired,
                formation_grad=formation_grad,
                cbf_enabled=self.cbf_enabled,
                dt=control_dt,
                yaw_rates=wz_all,
            )
            self._publish_prediction_markers()

            # ── CBF 監看：發布 /cbf_debug/*（rqt_plot 即時看 h / slack / 速度落差）──
            vcmd = {n: float(np.hypot(*u_safe[n])) for n in self.all_dogs}
            vact = {n: float(np.linalg.norm(states[n].vel_world))
                    for n in self.all_dogs}
            self.cbf_debug_pub.publish(
                self.qp.last_min_h_by_kind, self.qp.last_slack_by_kind,
                vcmd, vact)

            # ── 7. Velocity limiter + publish ──
            bootstrap_key = (
                self._dog_path_goal_key if self._per_dog_auto_active()
                else None)
            bootstrap_cmd_vel = (
                bootstrap_key is not None
                and self._cmd_vel_bootstrap_key != bootstrap_key
                and self._bootstrap_is_safe_now(states))
            for name in self.all_dogs:
                idx = self.all_dogs.index(name)
                safe_accel = self.qp.last_accel_cmds.get(name, np.zeros(2))
                self.accel_pub.publish(name, safe_accel[0], safe_accel[1])
                if bootstrap_cmd_vel:
                    vx = float(u_nominal[2 * idx] + control_dt * safe_accel[0])
                    vy = float(u_nominal[2 * idx + 1] + control_dt * safe_accel[1])
                else:
                    vx, vy = u_safe[name]
                wz = wz_all[name]
                vx, vy, wz = self.limiter.clamp(vx, vy, wz)
                self.cmd_pub.publish(name, vx, vy, wz)
            if bootstrap_cmd_vel:
                self._cmd_vel_bootstrap_key = bootstrap_key

            # ── 8. Stuck detection（保留）──
            center_u_safe = np.mean(
                np.array([u_safe[name] for name in self.all_dogs], dtype=float),
                axis=0,
            )
            self._update_stuck_state(center_u_safe, states, center_state)

            # ── 9. 診斷 log（每 2 秒一次）──
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 0
            if self._log_counter % (int(self.rate_hz) * 2) == 0:
                rospy.loginfo_throttle(
                    2.0,
                    "[UQP] f=%.4f | form='%s' | hocbf=%s | center=(%.2f,%.2f)",
                    f_cost,
                    self.laplacian.current_formation,
                    self.qp.last_cbf_status,
                    center_state.x, center_state.y,
                )
                if target_diag.get("path_mode", False):
                    rospy.loginfo_throttle(
                        2.0,
                        "[UQP-path] ready=%s | goal_err=%.2f | path_v=%.2f | form_v=%.2f",
                        target_diag.get("path_ready", False),
                        target_diag.get("max_goal_error", 0.0),
                        target_diag.get("max_path_speed", 0.0),
                        target_diag.get("max_form_speed", 0.0),
                    )
                elif (target_diag["guard_scale"] < 0.999
                      or target_diag["max_projection_shift"] > 1e-3):
                    rospy.loginfo_throttle(
                        2.0,
                        "[UQP-target] guard=%.2f | slot_err=%.2f | proj=%.2f | %s",
                        target_diag["guard_scale"],
                        target_diag["max_slot_error"],
                        target_diag["max_projection_shift"],
                        "; ".join(target_diag["projected_targets"]) or "none",
                    )

            if self._slot_switch_cooldown_counter > 0:
                self._slot_switch_cooldown_counter -= 1

            rate.sleep()

    def _update_stuck_state(self, center_u_safe, states, center_state):
        """Stuck detection（從舊版完整保留，recovery waypoint 改用 FormationSwitcher）"""
        if not self.navigator.has_goal:
            self._stuck_counter = 0
            self._consec_replans = 0
            return
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return
        speed = math.hypot(center_u_safe[0], center_u_safe[1])
        if speed < self.stuck_speed_threshold:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
            self._consec_replans = 0
            return
        if self._stuck_counter < self.stuck_replan_cycles:
            return
        if self._consec_replans >= self.stuck_max_replans:
            self.navigator.abort_to_keyboard(
                "%d replans, formation center still stuck" % self.stuck_max_replans)
            self._consec_replans = 0
            self._stuck_counter = 0
            self._cooldown_counter = 0
            return
        reason = "formation center speed≈0 for %.2fs (hard HOCBF status=%s)" % (
            self._stuck_counter / self.rate_hz, self.qp.last_cbf_status)
        if self.per_dog_astar_enabled:
            self._reset_per_dog_paths(reason)
            self._consec_replans += 1
            self._stuck_counter = 0
            self._cooldown_counter = self.stuck_replan_cooldown
            return
        if self._door_enabled and self._should_use_door_recovery(center_state):
            via = self.switcher.recovery_waypoint(center_state)
            if self.navigator.force_via_waypoint(tuple(center_state.pos), via, reason):
                self._consec_replans += 1
                self._stuck_counter = 0
                self._cooldown_counter = self.stuck_replan_cooldown
                return
        if self.navigator.force_replan(reason):
            self._consec_replans += 1
            self._stuck_counter = 0
            self._cooldown_counter = self.stuck_replan_cooldown


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        manager = FleetManagerUQP()
        manager.spin()
    except rospy.ROSInterruptException:
        pass