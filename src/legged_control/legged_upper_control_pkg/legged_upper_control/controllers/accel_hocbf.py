"""
二階 CBF QP 控制器 — HOCBF acceleration-level 安全過濾。
決策變數為 body frame 加速度，約束為 HOCBF 不等式。
"""

import math
import itertools
import numpy as np
import cvxpy as cp
import rospy

from .base import CBFControllerBase
from ..core.geometry import rot2d, wrap_to_pi, closest_point_on_aabb


class TwoOrderCBFQPController(CBFControllerBase):
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
            u_now[2 * idx:2 * idx + 2] = rot2d(s.yaw).T @ s.vel_world  # .T 是轉置 旋轉矩陣轉置x世界座標速度 = body frame速度 
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
        dog_idx = {name: i for i, name in enumerate(all_dogs)} # enumerate 的意思是 自動幫你編號  {"dog1": 0, "dog2": 1, "dog3": 2} 
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
            sl = slice(2 * idx, 2 * idx + 2) # slice = 存取成變數 v_body = u_now[sl] = u_now[0:2]
            v_body = u_now[sl]
            wz = float(yaw_rates.get(name, 0.0))
            yaw_term_body = wz * np.array([-v_body[1], v_body[0]], dtype=float) # a^W = R · a^B (決策變數) + R · ωz·[-vy^B, vx^B] (已知常數) 
            #                                                                                               ↑ 這一項來自 Ṙ·v^B（旋轉矩陣對時間微分）
            yaw_acc_world[name] = R_dogs[name] @ yaw_term_body # 這邊在處理h''的微分 因為都是在world frame  

        A_rows, b_rows, row_meta = [], [], []

        if cbf_enabled:   # all_dogs 是一個 array ["dog1", "dog2", "dog3"] index = 0 1 2 
            # ── Pairwise robot-robot HOCBF ──
            for ia in range(n_dogs): # n_dog=3 故 ia = 0 1 2 -> 選一對狗 
                for ib in range(ia + 1, n_dogs): # ia = 0 ib = range(1 3 ) = 1 2 做配對 dog1 配 dog2 dog2配do3 dog1配dog3
                    na, nb = all_dogs[ia], all_dogs[ib] 
                    sa, sb = states[na], states[nb] # 
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
                    h = float(dp @ dp) - d_min_eff ** 2 # 定義barrier function 
                    hdot = float(2.0 * dp @ dv)
                    hddot_free = float(
                        2.0 * dv @ dv
                        + 2.0 * dp @ (yaw_acc_world[na] - yaw_acc_world[nb])) # 沒有決策變數的部份 = 常數部份 
                    a_row = np.zeros(n_vars)  # 六維 0 向量
                    col_a = 2 * dog_idx[na] # 以dog1為例 cola = 0
                    a_row[col_a:col_a + 2] = 2.0 * dp @ R_dogs[na] # a_row[0:2]= 2xdp x 旋轉矩陣  +2 dp^T R_a
                    col_b = 2 * dog_idx[nb] 
                    a_row[col_b:col_b + 2] = -2.0 * dp @ R_dogs[nb] #  # -2 dp^T R_b 
                    A_rows.append(a_row)
                    b_rows.append(self._hocbf_rhs(
                        h, hdot, hddot_free,
                        self.gamma_robot_1, self.gamma_robot_2)) # b = −hddfree ​−(γ1​+γ2​)h˙−γ1​γ2​⋅h , 
                    row_meta.append({
                        "kind": "pair",
                        "name": "%s-%s" % (na, nb),
                        "h": h,
                        "hdot": hdot,
                    }) # CBF Constraint =>  A·a ≥ b , a ＝ 決策變數 

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
                and len(formation_grad) == n_dogs): # 預設0 然後給fallback weight太小 沒有梯度 or 數量不對 維持0 沒有formation這個cost
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

        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints) # 解QP 

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

