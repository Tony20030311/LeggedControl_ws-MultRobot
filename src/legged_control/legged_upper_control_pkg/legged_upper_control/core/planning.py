"""
A* 路徑規劃器 — 在 2D 格子地圖上搜尋最短路徑。
地圖在 __init__ 時一次性建立，支援多種膨脹半徑快取。
"""

import math
import heapq
import numpy as np
import rospy


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


