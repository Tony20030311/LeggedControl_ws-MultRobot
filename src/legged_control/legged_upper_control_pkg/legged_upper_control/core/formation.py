"""
編隊模組 — Laplacian formation similarity metric + 自動隊形切換。
"""

import math
import numpy as np
import rospy


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

