"""
Fleet Manager — 多機編隊上層控制的主迴圈。
組裝所有模組並在 spin() 中按順序呼叫。
"""

import math
import os
import threading
import itertools
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ..core.config import get_config, _CFG, _CONFIG_PATH, _load_config
from ..core.geometry import rot2d, wrap_to_pi, quaternion_to_yaw, closest_point_on_aabb
from ..core.state import RobotState, StateCollector
from ..core.planning import AStarPlanner
from ..core.navigation import PurePursuitController, LeaderCmdRelay, LeaderNavigator
from ..core.formation import LaplacianFormation, FormationSwitcher
from ..core.io import VelocityLimiter, CmdVelPublisher, AccelCmdPublisher, CbfDebugPublisher
from ..controllers.accel_hocbf import TwoOrderCBFQPController
from ..controllers.velocity_qp import UnifiedQPController


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
        rospy.init_node("fleet_manager")

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
            w_pred=rospy.get_param(
                "~w_pred", _CFG.get("w_pred", 20.0)),
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
        # per-dog 路徑朝向：各狗追自己 A* 路徑方向(由自身 nominal 速度導出),
        # 取代「全隊追同一 formation yaw」。避免後狗被迫朝共同 heading、正面對到
        # 前方障礙而被 CBF footprint 放大誤煞。keyboard/centroid 模式仍用 formation yaw。
        self._dog_path_yaw = {name: 0.0 for name in self.all_dogs}
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
            v_world = rot2d(s.yaw) @ v_body
            mean_path_world += v_world
            # 各狗自己的路徑朝向 = 自身 nominal world 速度方向。
            # 速度趨近 0(近 goal/latched)時不更新 → 凍結上一個朝向,避免末端抖動。
            if float(np.linalg.norm(v_world)) > 0.05:
                self._dog_path_yaw[name] = math.atan2(v_world[1], v_world[0])
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

    def spin(self): # 主啟動迴圈 
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
                center_nom = (0.0, 0.0) # 不給虛擬中心速度 
                if self._per_dog_goals_reached(states):
                    self.navigator.finish_auto_goal("per-dog goals")
                    self._reset_per_dog_paths("goal reached")
                mode_after = self.navigator.current_mode
            else:
                center_nom, _ = self.navigator.get_nominal(center_state) # 否則給虛擬中心速度 
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
            f_cost, formation_grad = self.laplacian.compute(positions) # 計算f cost + gradient 
            u_nominal, target_diag = self._build_nominal_velocity( # 計算 追A* Waypoint 的 pure persuit 速度 gain x error
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

            # ── 4. yaw tracking：per-dog AUTO 各狗追自己的 A* 路徑朝向；
            #      keyboard/centroid 模式沒有 per-dog path,維持共同 formation yaw。
            wz_all = {}
            per_dog_yaw_active = self._per_dog_auto_active()
            for name in self.all_dogs:
                s = states[name]
                if per_dog_yaw_active:
                    desired_yaw = self._dog_path_yaw.get(
                        name, self._last_formation_yaw)
                else:
                    desired_yaw = self._last_formation_yaw
                e_yaw = wrap_to_pi(desired_yaw - s.yaw)
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