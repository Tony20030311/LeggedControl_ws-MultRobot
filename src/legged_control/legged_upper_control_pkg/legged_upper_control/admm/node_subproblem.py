"""Stage 1 -- single-dog node QP (no ADMM, no edge, no consensus, no formation).

World-frame double integrator, trajectory-level multiple shooting. Decision
vector xi = [x_1..x_N (4N), a_0..a_{N-1} (2N)] with x_0 the measured initial
condition bound by the dynamics equality. Safety is the Xiong discrete
second-order HOCBF (node-local obstacle/wall), linearized once per cycle to a
linear inequality on the accelerations.

All coefficients and index formulas come from constants.py (the stage-0 gate) --
nothing is recomputed or hardcoded here. OSQP is used with a *fixed* sparsity
pattern: the pattern is assembled once, and each cycle only the values change
(q, the CBF rows of A, and l/u) via osqp.update(...). Never rebuilt, warm-started.

CBF row (increment-form linearization about operating point (pbar, abar)).
FULL linearization: a_k enters BOTH h_{k+2} (via 3/2 Ts^2) and h_{k+1} (via
1/2 Ts^2) -- dropping the h_{k+1} term under-brakes ~4 cm skirting an obstacle
(verified stage 1). g = d(three-point value)/d(a_k), A-row = -g:

  three-point value  Hbar = hbar_{k+2} + COEF_HK1*hbar_{k+1} + COEF_HK*hbar_k
  obstacle (grad 2e):  g = COEF_HK2*CBF_CONSTR_COEF*ebar_{k+2}
                            + COEF_HK1*CBF_CONSTR_COEF_P1*ebar_{k+1}
  wall (affine, exact): g = (COEF_HK2*A_TO_P2_COEF + COEF_HK1*A_TO_P1_COEF)*n_w
  A-row on a_k = -g ,  u = b_k = Hbar - g . abar_k

hard/soft split: accel-step k=0 is a hard constraint (strict guarantee at the
current state); k=1..N-2 are soft (slack s>=0 penalized by w_pred) so a far
prediction step never makes the QP infeasible, it just bends earlier steps.

cold start (no warm start): the straight reference is a weak linearization point
-- if an obstacle sits within ~2 steps of the start, the k=0 hard row linearized
there has a tiny gradient (no steering authority) and the QP is infeasible. So
cycle 0 first runs a SOFT pass (k=0 hard rows dropped; the k>=1 soft rows still
curve the path around the obstacle), then linearizes the real HARD pass at that
already-curving trajectory -- where the gradient has authority. Steady-state
cycles have a warm start and skip the soft pass (no extra cost). This soft warm-up
doubles as the stage-2 edge cold-start seed (paper 4.6 uncoupled-MPC z0). The k=0
hard row is NEVER softened for the output pass -- if it is still infeasible after
the soft warm-up, that is reported, not hidden.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import osqp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C  # noqa: E402


# --- node-local barrier helpers (also used by the verify gate) ----------------
def h_obstacle(p, p_obs, r_eff):
    e = np.asarray(p, dtype=float)[:2] - np.asarray(p_obs, dtype=float)[:2]
    return float(e @ e) - r_eff * r_eff


def h_wall(p, n_w, p_w, d_safe):
    return float(np.asarray(n_w, dtype=float)[:2]
                 @ (np.asarray(p, dtype=float)[:2]
                    - np.asarray(p_w, dtype=float)[:2])) - d_safe


class NodeSubproblem:
    """Single-dog node QP over the 6N decision vector + soft-CBF slacks."""

    def __init__(self, obstacles=None, walls=None, robot_margin=0.30,
                 q_pos=10.0, q_v=1.0, r_accel=0.5, w_pred=20.0, n=None):
        self.N = C.N if n is None else int(n)
        N = self.N
        self.n_xi = C.xi_dim(N)                       # 6N
        # obstacles: [{"pos": (x,y), "radius": r}]; r_eff = radius + robot_margin
        self.obstacles = list(obstacles or [])
        # walls: [{"normal": (nx,ny), "point": (px,py), "d_safe": d}]
        self.walls = list(walls or [])
        self.robot_margin = float(robot_margin)
        self.q_pos, self.q_v = float(q_pos), float(q_v)
        self.r_accel, self.w_pred = float(r_accel), float(w_pred)

        n_feat = len(self.obstacles) + len(self.walls)
        # CBF accel-steps k = 0..N-2 (need state x_{k+2}); k=0 hard, k>=1 soft.
        self.n_soft_per = N - 2                       # k = 1..N-2
        self.n_slack = n_feat * self.n_soft_per
        self.nvar = self.n_xi + self.n_slack

        # row-block offsets
        self._r_dyn = 0
        self._r_vb = self._r_dyn + 4 * N              # velocity bounds
        self._r_ab = self._r_vb + 2 * N               # accel bounds
        self._r_hard = self._r_ab + 2 * N             # CBF hard (k=0)
        self._r_soft = self._r_hard + n_feat          # CBF soft (k=1..N-2)
        self._r_snn = self._r_soft + self.n_slack     # slack >= 0
        self.nrow = self._r_snn + self.n_slack

        self._build_fixed()
        self._build_cbf_layout()
        self._P = self._build_P()
        self._prob = None

    # ---- feature access ----
    def _features(self):
        """Uniform (kind, geom) list: obstacles first, then walls."""
        feats = []
        for o in self.obstacles:
            feats.append(("obs", np.asarray(o["pos"][:2], float),
                          float(o["radius"]) + self.robot_margin))
        for w in self.walls:
            feats.append(("wall", np.asarray(w["normal"][:2], float),
                          np.asarray(w["point"][:2], float),
                          float(w.get("d_safe", 0.4))))
        return feats

    # ---- fixed part of A (dynamics, bounds, slack -1's, slack>=0) ----
    def _build_fixed(self):
        N = self.N
        rows, cols, vals = [], [], []
        lo = np.zeros(self.nrow)
        up = np.zeros(self.nrow)

        def put(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        Ad, Bd = C.Ad, C.Bd
        # dynamics equality: x_{k+1} = Ad x_k + Bd a_k ; x_0 measured (b_eq set later)
        for k in range(N):
            r0 = self._r_dyn + 4 * k
            for d in range(4):                        # +I4 on x_{k+1}
                put(r0 + d, C.x_index(k + 1, N) + d, 1.0)
            for d in range(4):                        # -Bd on a_k
                for c in range(2):
                    put(r0 + d, C.a_index(k, N) + c, -Bd[d, c])
            if k >= 1:                                # -Ad on x_k
                for d in range(4):
                    for c in range(4):
                        put(r0 + d, C.x_index(k, N) + c, -Ad[d, c])
            # l == u == b_eq (filled per cycle in _lu)

        # velocity bounds on states k=1..N
        for k in range(1, N + 1):
            rvx = self._r_vb + 2 * (k - 1)
            put(rvx, C.vx_index(k, N), 1.0)
            put(rvx + 1, C.vy_index(k, N), 1.0)
            lo[rvx], up[rvx] = -C.MAX_VX, C.MAX_VX
            lo[rvx + 1], up[rvx + 1] = -C.MAX_VY, C.MAX_VY
        # accel bounds on a_k, k=0..N-1
        for k in range(N):
            rax = self._r_ab + 2 * k
            put(rax, C.ax_index(k, N), 1.0)
            put(rax + 1, C.ay_index(k, N), 1.0)
            lo[rax], up[rax] = -C.MAX_AX, C.MAX_AX
            lo[rax + 1], up[rax + 1] = -C.MAX_AY, C.MAX_AY

        # slack >= 0 identity rows + slack column -1 in soft CBF rows
        for j in range(self.n_slack):
            put(self._r_snn + j, self.n_xi + j, 1.0)
            lo[self._r_snn + j], up[self._r_snn + j] = 0.0, np.inf
            put(self._r_soft + j, self.n_xi + j, -1.0)   # -s in soft row

        self._fixed = (np.array(rows), np.array(cols), np.array(vals))
        self._lo_base, self._up_base = lo, up

    # ---- CBF row layout: which (feature, k) -> row, and its 2 a_k columns ----
    def _build_cbf_layout(self):
        N = self.N
        n_feat = len(self.obstacles) + len(self.walls)
        self._cbf = []          # list of dict(feat, k, row, cols(2), is_hard)
        for f in range(n_feat):
            # hard row k=0
            self._cbf.append(dict(feat=f, k=0, row=self._r_hard + f,
                                  cols=(C.a_index(0, N), C.a_index(0, N) + 1),
                                  hard=True))
            # soft rows k=1..N-2
            for kk, k in enumerate(range(1, N - 1)):
                soft_j = f * self.n_soft_per + kk
                self._cbf.append(dict(feat=f, k=k, row=self._r_soft + soft_j,
                                      cols=(C.a_index(k, N), C.a_index(k, N) + 1),
                                      hard=False))

    def _build_P(self):
        """Diagonal Hessian: 2*blkdiag(Q..P_term, R.., w_pred I). PSD."""
        N = self.N
        diag = np.zeros(self.nvar)
        for k in range(1, N):                         # stage states x_1..x_{N-1}
            diag[C.px_index(k, N)] = 2.0 * self.q_pos
            diag[C.py_index(k, N)] = 2.0 * self.q_pos
            # velocity stage weight is 0 (B7)
        # terminal state x_N: P_term = diag{10 q_pos, 10 q_pos, q_v, q_v}
        diag[C.px_index(N, N)] = 2.0 * 10.0 * self.q_pos
        diag[C.py_index(N, N)] = 2.0 * 10.0 * self.q_pos
        diag[C.vx_index(N, N)] = 2.0 * self.q_v
        diag[C.vy_index(N, N)] = 2.0 * self.q_v
        for k in range(N):                            # accels a_0..a_{N-1}
            diag[C.ax_index(k, N)] = 2.0 * self.r_accel
            diag[C.ay_index(k, N)] = 2.0 * self.r_accel
        for j in range(self.n_slack):                 # slacks
            diag[self.n_xi + j] = 2.0 * self.w_pred
        return sp.diags(diag).tocsc()

    # ---- per-cycle assembly ----
    def _q(self, x_des):
        """Linear term: -2 Q x_des (position only), -2 P_term x_des at terminal."""
        N = self.N
        q = np.zeros(self.nvar)
        for k in range(1, N):
            q[C.px_index(k, N)] = -2.0 * self.q_pos * x_des[k - 1, 0]
            q[C.py_index(k, N)] = -2.0 * self.q_pos * x_des[k - 1, 1]
        q[C.px_index(N, N)] = -2.0 * 10.0 * self.q_pos * x_des[N - 1, 0]
        q[C.py_index(N, N)] = -2.0 * 10.0 * self.q_pos * x_des[N - 1, 1]
        # terminal velocity target is 0 -> no linear term
        return q

    def _operating_point(self, x_now, xbar):
        """pbar[0..N] positions (pbar[0]=x_now) and abar[0..N-1] accels."""
        N = self.N
        pbar = np.zeros((N + 1, 2))
        abar = np.zeros((N, 2))
        pbar[0] = np.asarray(x_now, float)[:2]
        if xbar is None:
            return None  # signal: caller supplies pbar via x_des
        for k in range(1, N + 1):
            pbar[k] = xbar[C.px_index(k, N):C.px_index(k, N) + 2]
        for k in range(N):
            abar[k] = xbar[C.ax_index(k, N):C.ax_index(k, N) + 2]
        return pbar, abar

    def _cbf_row(self, feat, k, pbar, abar):
        """Return (coef_vec(2,), b_scalar) for one CBF (feature, accel-step k).

        Full linearization: a_k enters BOTH h_{k+2} (coef 3/2 Ts^2) and h_{k+1}
        (coef 1/2 Ts^2). g = d(three-point value)/d(a_k); A-row = -g;
        b = three-point value - g . abar_k  (increment form). Walls are affine
        so this is exact; obstacles drop only the O(Ts^4) quadratic term.
        """
        kind = feat[0]
        pk, pk1, pk2 = pbar[k], pbar[k + 1], pbar[k + 2]
        if kind == "obs":
            p_obs, r_eff = feat[1], feat[2]
            hb = np.array([h_obstacle(pk, p_obs, r_eff),
                           h_obstacle(pk1, p_obs, r_eff),
                           h_obstacle(pk2, p_obs, r_eff)])
            e1, e2 = pk1 - p_obs, pk2 - p_obs
            g = (C.COEF_HK2 * C.CBF_CONSTR_COEF * e2
                 + C.COEF_HK1 * C.CBF_CONSTR_COEF_P1 * e1)
        else:  # wall (affine, exact)
            n_w, p_w, d_safe = feat[1], feat[2], feat[3]
            hb = np.array([h_wall(pk, n_w, p_w, d_safe),
                           h_wall(pk1, n_w, p_w, d_safe),
                           h_wall(pk2, n_w, p_w, d_safe)])
            g = (C.COEF_HK2 * C.A_TO_P2_COEF + C.COEF_HK1 * C.A_TO_P1_COEF) * n_w
        hbar3 = C.COEF_HK2 * hb[2] + C.COEF_HK1 * hb[1] + C.COEF_HK * hb[0]
        return -g, hbar3 - float(g @ abar[k])

    def _assemble_A(self, pbar, abar):
        """A (csc) with fixed pattern + this cycle's CBF values; and u for CBF."""
        rows_f, cols_f, vals_f = self._fixed
        feats = self._features()
        rows_c, cols_c, vals_c = [], [], []
        up = self._up_base.copy()
        for entry in self._cbf:
            coef, b = self._cbf_row(feats[entry["feat"]], entry["k"], pbar, abar)
            c0, c1 = entry["cols"]
            rows_c += [entry["row"], entry["row"]]
            cols_c += [c0, c1]
            vals_c += [coef[0], coef[1]]
            up[entry["row"]] = b                      # l = -inf (set in _lu)
        rows = np.concatenate([rows_f, np.array(rows_c, dtype=int)])
        cols = np.concatenate([cols_f, np.array(cols_c, dtype=int)])
        vals = np.concatenate([vals_f, np.array(vals_c, dtype=float)])
        A = sp.csc_matrix((vals, (rows, cols)), shape=(self.nrow, self.nvar))
        return A, up

    def _lu(self, x_now, up_cbf):
        lo = self._lo_base.copy()
        up = up_cbf
        # dynamics equality: l = u = b_eq (first block Ad @ x_now, rest 0)
        beq = np.zeros(4 * self.N)
        beq[0:4] = C.Ad @ np.asarray(x_now, float)
        lo[self._r_dyn:self._r_dyn + 4 * self.N] = beq
        up[self._r_dyn:self._r_dyn + 4 * self.N] = beq
        # CBF rows: l = -inf, u = b (already in up)
        for entry in self._cbf:
            lo[entry["row"]] = -np.inf
        return lo, up

    def solve(self, x_now, x_des, xbar=None):
        """One node QP. Returns dict(status, xi, x_pred(N,4), a_pred(N,2), a0).

        Steady state (xbar given): one hard pass linearized at xbar. Cold start
        (xbar is None): a soft warm-up pass (k=0 hard rows dropped) yields a
        curved operating point, then the real hard pass linearizes there.
        """
        N = self.N
        x_now = np.asarray(x_now, dtype=float)
        q = self._q(x_des)

        op = self._operating_point(x_now, xbar)
        if op is None:                                # cold start: soft warm-up
            pbar = np.zeros((N + 1, 2))
            pbar[0] = x_now[:2]
            pbar[1:] = x_des[:, :2]                    # straight reference seed
            abar = np.zeros((N, 2))
            soft = self._solve_pass(x_now, q, pbar, abar, drop_hard=True)
            soft_xi = soft["xi"] if (soft["xi"] is not None
                                     and np.all(np.isfinite(soft["xi"]))) else None
            op2 = self._operating_point(x_now, soft_xi)
            if op2 is not None:                       # linearize hard pass here
                pbar, abar = op2
            # else: soft pass failed too -> hard pass runs at the reference seed
            # and reports the infeasibility honestly (k=0 is never softened).
        else:
            pbar, abar = op

        return self._solve_pass(x_now, q, pbar, abar, drop_hard=False)

    def _solve_pass(self, x_now, q, pbar, abar, drop_hard):
        """Assemble + solve one pass at (pbar, abar). drop_hard=True disables the
        k=0 hard CBF rows (u -> +inf) for the cold-start soft warm-up; the fixed
        sparsity pattern is untouched (only u changes), so it stays update-only."""
        A, up_cbf = self._assemble_A(pbar, abar)
        lo, up = self._lu(x_now, up_cbf)
        if drop_hard:
            for entry in self._cbf:
                if entry["hard"]:
                    up[entry["row"]] = np.inf
        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(P=self._P, q=q, A=A, l=lo, u=up,
                             verbose=False, warm_start=True,
                             eps_abs=1e-6, eps_rel=1e-6, polish=True,
                             max_iter=8000)
            self._A0 = A
        else:
            # same sparsity pattern -> update values only, never rebuild
            self._prob.update(q=q, l=lo, u=up, Ax=A.data)
        return self._out_from_res(self._prob.solve())

    def _out_from_res(self, res):
        N = self.N
        status = res.info.status
        xi = np.asarray(res.x, dtype=float) if res.x is not None else None
        out = dict(status=status, xi=xi)
        if xi is None or not np.all(np.isfinite(xi)):
            out.update(x_pred=None, a_pred=None, a0=None)
            return out
        x_pred = np.array([xi[C.x_index(k, N):C.x_index(k, N) + 4]
                           for k in range(1, N + 1)])
        a_pred = np.array([xi[C.a_index(k, N):C.a_index(k, N) + 2]
                           for k in range(N)])
        out.update(x_pred=x_pred, a_pred=a_pred, a0=a_pred[0].copy())
        return out
