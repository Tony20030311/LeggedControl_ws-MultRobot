"""
一階 CBF QP 控制器 — velocity-level 安全過濾。
決策變數為 body frame 速度，約束為一階 CBF 不等式。
"""

import math
import itertools
import numpy as np
import cvxpy as cp
import rospy

from .base import CBFControllerBase
from ..core.geometry import rot2d, wrap_to_pi, closest_point_on_aabb


class UnifiedQPController(CBFControllerBase):
    """
    統一上層 QP：nominal velocity tracking + CBF 安全約束。

    與舊版 CBFSafetyFilter 的差異:
      - objective 從 ‖u - u_nom‖² 升級為多項 cost
      - constraint 建構邏輯完全保留（robot-robot + obstacle + rect + wall + predictive）
      - nominal velocity 由 centroid-relative offset tracking 產生

    決策變數: u = [vx1, vy1, vx2, vy2, vx3, vy3] (body frame)
    wz pass-through 不進 QP。

    QP 公式:
        min  w_track · Σ_i ‖u_i − u_nom_i‖²
           + w_formation · Σ_i ∂f/∂p_i · R_i u_i dt ← Laplacian formation cost
           + w_reg  · ‖u‖²                      ← 正則項（防發散）
           + λ      · ‖ε‖²                      ← slack 懲罰

        s.t. A_cbf · u ≥ b_cbf − ε              ← 所有 CBF 約束（不動）
             ε ≥ 0
    """

    def __init__(self, gamma_robot, d_min, gamma_obs=1.0, gamma_wall=1.0,
                 slack_lambda=1e4, slack_warn_threshold=0.05,
                 lookahead_tau=0.15,
                 w_path=1.0, w_track=5.0, w_formation=0.0, w_reg=0.1,
                 max_vx=0.55, max_vy=0.35,
                 footprint_half_length=0.35,
                 footprint_half_width=0.20,
                 footprint_drift_margin=0.08,
                 prediction_enabled=False,
                 prediction_horizon=1,
                 prediction_dt=0.0,
                 w_smooth=0.2,
                 laplacian_ref=None):
        # ── CBF 參數（從舊版完整保留）──
        self.gamma_robot = gamma_robot
        self.d_min       = d_min
        self.d_min_sq    = d_min ** 2
        self.gamma_obs   = gamma_obs
        self.gamma_wall  = gamma_wall
        self.obstacles   = []
        self.rect_obstacles = []
        self.walls       = []
        self.slack_lambda = float(slack_lambda)
        self.slack_warn_threshold = float(slack_warn_threshold)
        self.last_max_slack = 0.0
        self.lookahead_tau = float(lookahead_tau)

        # ── 新增: QP cost 權重 ──
        self.w_path      = float(w_path)
        self.w_track     = float(w_track)
        self.w_formation = float(w_formation)
        self.w_reg       = float(w_reg)
        self.max_vx      = float(max_vx)
        self.max_vy      = float(max_vy)
        self.footprint_half_length = max(0.0, float(footprint_half_length))
        self.footprint_half_width = max(0.0, float(footprint_half_width))
        self.footprint_drift_margin = max(0.0, float(footprint_drift_margin))
        self.prediction_enabled = bool(prediction_enabled)
        self.prediction_horizon = max(1, int(prediction_horizon))
        self.prediction_dt = max(0.0, float(prediction_dt))
        self.w_smooth = float(w_smooth)
        self.laplacian_ref = laplacian_ref
        self._last_V_sol = None
        self.last_prediction_paths = {}
        self.last_reference_paths = {}
        self.last_prediction_dt = 0.0

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
        rospy.loginfo("[UnifiedQP] %d obstacles loaded", len(obstacles))

    def set_rect_obstacles(self, rect_obstacles):
        self.rect_obstacles = rect_obstacles
        rospy.loginfo("[UnifiedQP] %d rect obstacles loaded", len(rect_obstacles))

    def set_walls(self, walls):
        self.walls = walls
        rospy.loginfo("[UnifiedQP] %d walls loaded", len(walls))

    def reset_prediction(self, reason=""):
        self._last_V_sol = None
        self.last_prediction_paths = {}
        self.last_reference_paths = {}
        self.last_prediction_dt = 0.0
        if reason:
            rospy.logdebug("[UnifiedQP] reset prediction: %s", reason)

    def solve(self, all_dogs, states, u_nominal, formation_grad=None,
              cbf_enabled=True, dt=0.05):
        if not self.prediction_enabled or self.prediction_horizon <= 1:
            self.reset_prediction()
            return self._solve_single_step(
                all_dogs,
                states,
                u_nominal,
                formation_grad=formation_grad,
                cbf_enabled=cbf_enabled,
                dt=dt,
            )
        return self._solve_multistep_preview(
            all_dogs,
            states,
            u_nominal,
            formation_grad=formation_grad,
            cbf_enabled=cbf_enabled,
            dt=dt,
        )

    def _solve_single_step(self, all_dogs, states, u_nominal, formation_grad=None,
                           cbf_enabled=True, dt=0.05):
        """
        Parameters
        ----------
        all_dogs      : list[str]         狗名列表
        states        : dict[str, RobotState]
        u_nominal     : np.ndarray(2N,)   每隻狗的 nominal body-frame 速度
        formation_grad: list[np.ndarray]  ∂f/∂p_i，world frame
        dt            : float             QP single-step prediction horizon

        Returns
        -------
        dict[str, (vx, vy)]               每隻狗的 body frame 安全速度
        """
        n_dogs = len(all_dogs)
        n_vars = 2 * n_dogs
        dog_idx = {name: i for i, name in enumerate(all_dogs)}
        dt = max(0.0, float(dt))

        A_rows, b_rows = [], []

        if cbf_enabled:
            # ═══ 建構 CBF constraint（從舊版完整複製）═══
            # ── Pairwise robot-robot CBF ──
            for ia in range(n_dogs):
                for ib in range(ia + 1, n_dogs):
                    na, nb = all_dogs[ia], all_dogs[ib]
                    sa, sb = states[na], states[nb]
                    if not sa.received or not sb.received:
                        continue
                    dp = sa.pos - sb.pos
                    h  = float(dp @ dp) - self.d_min_sq
                    a_row = np.zeros(n_vars)
                    col_a = 2 * dog_idx[na]
                    a_row[col_a:col_a + 2] = 2.0 * dp @ rot2d(sa.yaw)
                    col_b = 2 * dog_idx[nb]
                    a_row[col_b:col_b + 2] = -2.0 * dp @ rot2d(sb.yaw)
                    A_rows.append(a_row)
                    b_rows.append(-self.gamma_robot * h)

            # ── Obstacle CBF（predictive）──
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

            # ── Rect obstacle CBF（有洞口的牆段）──
            for rect in self.rect_obstacles:
                center = np.array(rect["center"][:2], dtype=float)
                size   = np.array(rect["size"][:2], dtype=float)
                d_safe = float(rect.get("d_safe", 0.35))
                for name in all_dogs:
                    s = states[name]
                    if not s.received:
                        continue
                    p_pred    = s.pos + s.vel_world * self.lookahead_tau
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
                    h_rect = dist ** 2 - d_safe ** 2
                    a_row = np.zeros(n_vars)
                    col = 2 * dog_idx[name]
                    a_row[col:col + 2] = 2.0 * dp @ rot2d(s.yaw)
                    A_rows.append(a_row)
                    b_rows.append(-self.gamma_obs * h_rect)

            # ── Wall CBF（predictive）──
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
                    p_pred = s.pos + s.vel_world * self.lookahead_tau
                    h_wall = float(n_w @ (p_pred - p_w)) - d_safe
                    a_row  = np.zeros(n_vars)
                    col    = 2 * dog_idx[name]
                    a_row[col:col + 2] = n_w @ rot2d(s.yaw)
                    A_rows.append(a_row)
                    b_rows.append(-self.gamma_wall * h_wall)

        # ═══ 建構 QP objective（升級部分）═══
        u = cp.Variable(n_vars)

        # (a) Nominal tracking: 每隻狗追 virtual-center formation target
        u_nominal = np.array(u_nominal, dtype=float)

        tracking_terms = []
        for idx, _ in enumerate(all_dogs):
            sl = slice(2 * idx, 2 * idx + 2)
            tracking_terms.append(cp.sum_squares(u[sl] - u_nominal[sl]))
        cost_tracking = self.w_track * (
            sum(tracking_terms) if tracking_terms else 0.0
        )

        # (b) Laplacian formation cost from first-order Taylor expansion:
        # f(p + R_i u_i dt) ~= f(p) + grad_f_i^T R_i u_i dt.
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
                    grad_world @ rot2d(s.yaw)) * dt
            cost_formation_linear = self.w_formation * (g_vec @ u)

        # (c) 正則項: w_reg · ‖u‖²（防止線性項讓速度發散）
        cost_reg = self.w_reg * cp.sum_squares(u)

        # (d) Slack（從舊版完整保留）
        objective_terms = [cost_tracking, cost_formation_linear, cost_reg]
        constraints = []
        eps = None

        for idx in range(n_dogs):
            constraints += [
                u[2 * idx]     <= self.max_vx,
                u[2 * idx]     >= -self.max_vx,
                u[2 * idx + 1] <= self.max_vy,
                u[2 * idx + 1] >= -self.max_vy,
            ]

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
            rospy.logwarn("[UnifiedQP] Solver error, returning zero")
            self.last_max_slack = float('inf')
            return {name: (0.0, 0.0) for name in all_dogs}

        if prob.status not in ("optimal", "optimal_inaccurate"):
            rospy.logwarn("[UnifiedQP] QP status=%s, returning zero", prob.status)
            self.last_max_slack = float('inf')
            return {name: (0.0, 0.0) for name in all_dogs}

        if eps is not None and eps.value is not None:
            self.last_max_slack = float(np.max(eps.value))
            if self.last_max_slack > self.slack_warn_threshold:
                rospy.logwarn_throttle(
                    1.0,
                    "[UnifiedQP] slack active, max ε=%.3f",
                    self.last_max_slack,
                )
        else:
            self.last_max_slack = 0.0

        u_sol = u.value
        return {name: (float(u_sol[2 * dog_idx[name]]),
                       float(u_sol[2 * dog_idx[name] + 1]))
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
                                 formation_grad=None, cbf_enabled=True,
                                 dt=0.05):
        """
        Preview multi-step CBF-QP.

        Future CBF/Laplacian rows are built around a fixed reference rollout
        from the previous horizon solution. The QP still decides the whole
        velocity sequence and publishes only the first step.
        """
        n_dogs = len(all_dogs)
        n_per_step = 2 * n_dogs
        n_steps = max(2, int(self.prediction_horizon))
        n_total = n_per_step * n_steps
        dog_idx = {name: i for i, name in enumerate(all_dogs)}
        dt_pred = self._prediction_step_dt(dt)

        R_dogs = {name: rot2d(states[name].yaw) for name in all_dogs}
        u_nominal = self._clip_body_velocity_vector(u_nominal, n_dogs)

        if self._last_V_sol is not None and len(self._last_V_sol) == n_total:
            nom_seq = np.concatenate([
                self._last_V_sol[n_per_step:], u_nominal
            ])
        else:
            nom_seq = np.tile(u_nominal, n_steps)
        nom_seq = self._clip_body_velocity_sequence(
            nom_seq, n_dogs, n_steps)
        p_nom = self._rollout_prediction_paths(
            all_dogs, states, nom_seq, R_dogs, dt_pred, n_steps)

        A_rows, b_rows = [], []

        if cbf_enabled:
            for k in range(n_steps):
                col_offset = k * n_per_step

                # Pairwise robot-robot CBF.
                for ia in range(n_dogs):
                    for ib in range(ia + 1, n_dogs):
                        na, nb = all_dogs[ia], all_dogs[ib]
                        sa, sb = states[na], states[nb]
                        if not sa.received or not sb.received:
                            continue
                        dp = p_nom[na][k] - p_nom[nb][k]
                        h = float(dp @ dp) - self.d_min_sq
                        a_row = np.zeros(n_total)
                        col_a = col_offset + 2 * dog_idx[na]
                        a_row[col_a:col_a + 2] = 2.0 * dp @ R_dogs[na]
                        col_b = col_offset + 2 * dog_idx[nb]
                        a_row[col_b:col_b + 2] = -2.0 * dp @ R_dogs[nb]
                        A_rows.append(a_row)
                        b_rows.append(-self.gamma_robot * h)

                # Circular obstacle CBF.
                for obs in self.obstacles:
                    p_obs = np.array(obs["pos"][:2])
                    r_obs = float(obs["radius"])
                    for name in all_dogs:
                        s = states[name]
                        if not s.received:
                            continue
                        pk = p_nom[name][k].copy()
                        if k == 0:
                            pk = pk + s.vel_world * self.lookahead_tau
                        dp = pk - p_obs
                        h_obs = float(dp @ dp) - r_obs ** 2
                        a_row = np.zeros(n_total)
                        col = col_offset + 2 * dog_idx[name]
                        a_row[col:col + 2] = 2.0 * dp @ R_dogs[name]
                        A_rows.append(a_row)
                        b_rows.append(-self.gamma_obs * h_obs)

                # Rect obstacle CBF.
                for rect in self.rect_obstacles:
                    center = np.array(rect["center"][:2], dtype=float)
                    size = np.array(rect["size"][:2], dtype=float)
                    d_safe = float(rect.get("d_safe", 0.35))
                    for name in all_dogs:
                        s = states[name]
                        if not s.received:
                            continue
                        pk = p_nom[name][k].copy()
                        if k == 0:
                            pk = pk + s.vel_world * self.lookahead_tau
                        p_closest = closest_point_on_aabb(pk, center, size)
                        dp = pk - p_closest
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
                                away = pk - center
                                if abs(away[0]) > abs(away[1]):
                                    dp = np.array([
                                        math.copysign(1.0, away[0] or 1.0),
                                        0.0,
                                    ])
                                else:
                                    dp = np.array([
                                        0.0,
                                        math.copysign(1.0, away[1] or 1.0),
                                    ])
                            dist = 0.0
                        h_rect = dist ** 2 - d_safe ** 2
                        a_row = np.zeros(n_total)
                        col = col_offset + 2 * dog_idx[name]
                        a_row[col:col + 2] = 2.0 * dp @ R_dogs[name]
                        A_rows.append(a_row)
                        b_rows.append(-self.gamma_obs * h_rect)

                # Wall CBF.
                for wall in self.walls:
                    n_w = np.array(wall["normal"][:2], dtype=float)
                    p_w = np.array(wall["point"][:2], dtype=float)
                    d_safe_base = float(wall.get("d_safe", 0.4))
                    for name in all_dogs:
                        s = states[name]
                        if not s.received:
                            continue
                        d_safe = max(
                            d_safe_base,
                            self._footprint_support_along(n_w, s.yaw),
                        )
                        pk = p_nom[name][k].copy()
                        if k == 0:
                            pk = pk + s.vel_world * self.lookahead_tau
                        h_wall = float(n_w @ (pk - p_w)) - d_safe
                        a_row = np.zeros(n_total)
                        col = col_offset + 2 * dog_idx[name]
                        a_row[col:col + 2] = n_w @ R_dogs[name]
                        A_rows.append(a_row)
                        b_rows.append(-self.gamma_wall * h_wall)

        V = cp.Variable(n_total)
        objective_terms = []

        tracking_terms = []
        for idx in range(n_dogs):
            sl = slice(2 * idx, 2 * idx + 2)
            tracking_terms.append(cp.sum_squares(V[sl] - u_nominal[sl]))
        objective_terms.append(
            self.w_track * (sum(tracking_terms) if tracking_terms else 0.0)
        )

        if (self.w_formation > 1e-9
                and formation_grad is not None
                and len(formation_grad) == n_dogs):
            for k in range(n_steps):
                if k == 0 or self.laplacian_ref is None:
                    grad_k = formation_grad
                else:
                    positions_k = [p_nom[name][k] for name in all_dogs]
                    _, grad_k = self.laplacian_ref.compute(positions_k)

                g_vec = np.zeros(n_per_step)
                for idx, name in enumerate(all_dogs):
                    s = states[name]
                    if not s.received:
                        continue
                    grad_world = np.array(grad_k[idx], dtype=float)
                    g_vec[2 * idx:2 * idx + 2] = (
                        grad_world @ R_dogs[name]) * dt_pred

                sl_k = slice(k * n_per_step, (k + 1) * n_per_step)
                objective_terms.append(self.w_formation * (g_vec @ V[sl_k]))

        objective_terms.append(self.w_reg * cp.sum_squares(V))

        if n_steps > 1 and self.w_smooth > 0.0:
            for k in range(n_steps - 1):
                sl_k = slice(k * n_per_step, (k + 1) * n_per_step)
                sl_k1 = slice((k + 1) * n_per_step,
                              (k + 2) * n_per_step)
                objective_terms.append(
                    self.w_smooth * cp.sum_squares(V[sl_k1] - V[sl_k])
                )

        constraints = []
        for k in range(n_steps):
            for idx in range(n_dogs):
                base = k * n_per_step + 2 * idx
                constraints += [
                    V[base] <= self.max_vx,
                    V[base] >= -self.max_vx,
                    V[base + 1] <= self.max_vy,
                    V[base + 1] >= -self.max_vy,
                ]

        eps = None
        if A_rows:
            A = np.array(A_rows)
            b = np.array(b_rows)
            eps = cp.Variable(A.shape[0], nonneg=True)
            constraints.append(A @ V >= b - eps)
            objective_terms.append(self.slack_lambda * cp.sum_squares(eps))

        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            rospy.logwarn("[UnifiedQP] Solver error, returning zero")
            self.last_max_slack = float('inf')
            self.reset_prediction("solver error")
            return {name: (0.0, 0.0) for name in all_dogs}

        if prob.status not in ("optimal", "optimal_inaccurate"):
            rospy.logwarn("[UnifiedQP] QP status=%s, returning zero",
                          prob.status)
            self.last_max_slack = float('inf')
            self.reset_prediction("non-optimal status")
            return {name: (0.0, 0.0) for name in all_dogs}

        if eps is not None and eps.value is not None:
            self.last_max_slack = float(np.max(eps.value))
            if self.last_max_slack > self.slack_warn_threshold:
                rospy.logwarn_throttle(
                    1.0,
                    "[UnifiedQP] slack active, max ε=%.3f (N=%d)",
                    self.last_max_slack,
                    n_steps,
                )
        else:
            self.last_max_slack = 0.0

        V_sol = np.array(V.value, dtype=float)
        self._last_V_sol = V_sol.copy()
        self.last_reference_paths = p_nom
        self.last_prediction_paths = self._rollout_prediction_paths(
            all_dogs, states, V_sol, R_dogs, dt_pred, n_steps)
        self.last_prediction_dt = dt_pred

        v0 = V_sol[:n_per_step]
        return {name: (float(v0[2 * dog_idx[name]]),
                       float(v0[2 * dog_idx[name] + 1]))
                for name in all_dogs}


# ═══════════════════════════════════════════════════════════════
# Module 5: VelocityLimiter（不動）
# ═══════════════════════════════════════════════════════════════

