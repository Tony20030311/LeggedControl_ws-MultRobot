"""
導航模組 — PurePursuit 路徑追蹤 + Leader 指令管理。
"""

import math
import threading
import numpy as np
import rospy
from geometry_msgs.msg import Twist, PoseStamped

from .geometry import wrap_to_pi


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

