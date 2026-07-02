"""Stage 2 edge subproblem -- the inter-agent (dog-dog) coupling QP of ADMM (20).

One edge {i,j}. Decision  w = [ z^{ij,i} (6N) ; z^{ij,j} (6N) ; s (N-1) ]  (~12N+n_s,
spec detail 4). The copies z are pulled toward the true bodies by two proximal terms;
the ONLY constraints on z are the linearized inter-agent HOCBF (soft, slack per step)
and s >= 0. Dynamics/bounds live in the node QP; z inherits feasibility via consensus.

  cost:   phi(s) + (rho/2)||z^i - xi^i - lam^i||^2 + (rho/2)||z^j - xi^j - lam^j||^2
          phi(s) = SLACK_LAMBDA s^2  (paper Eq.13)
  CBF row (soft, k=0..N-2; coefficients FROZEN by rti_linearizer this cycle):
          -g_k . a^i_k(z^i) + g_k . a^j_k(z^j) - s_k <= u_k
          col_a^i = -g_k , col_a^j = +g_k  (push-pull; a^j endpoint is -g^i = +g),
          g_k has BOTH the h_{k+2} (0.03) and h_{k+1} (0.01) terms (full linearization).

OSQP direct binding (spec E): P and the A sparsity pattern are fixed; set_linearization
refreshes the frozen CBF values (A data + u) once per control cycle; each ADMM iteration
updates only q (proximal targets) and warm-starts. Never cvxpy, never rebuilt per iter.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import osqp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C  # noqa: E402


class EdgeSubproblem:
    def __init__(self, n=None, rho=None, slack_lambda=None, hard_k0=True):
        self.N = C.N if n is None else int(n)
        N = self.N
        self.rho = C.RHO if rho is None else float(rho)
        self.slack_lambda = C.SLACK_LAMBDA if slack_lambda is None else float(slack_lambda)
        # Hard edge k=0 (stage-1-consistent: node & edge both k=0 hard / k>=1 soft).
        # All-soft cannot produce realized avoidance -- the g=3Ts^2*e gradient is too
        # weak, so slack absorbs the violation until lambda_slk ~ 1e6, which then
        # breaks ADMM convergence. Clamping the k=0 slack to 0 forces the immediate
        # inter-agent avoidance while k>=1 stay soft (never infeasible; convergence
        # kept). z is unbounded so the hard k=0 edge QP is always feasible.
        self.hard_k0 = bool(hard_k0)

        self.n_z = C.xi_dim(N)                 # 6N per copy
        self.n_cbf = N - 1                     # CBF accel-steps k = 0..N-2
        self.nvar = 2 * self.n_z + self.n_cbf  # z^i, z^j, s
        self._r_cbf = 0                        # CBF rows first
        self._r_snn = self.n_cbf               # then slack >= 0 rows
        self.nrow = 2 * self.n_cbf

        self._build_fixed()
        self._P = self._build_P()
        self._A = None
        self._u = None
        self._prob = None
        # pre-assemble the A sparsity pattern now (spec E) with zero CBF coefficients;
        # set_linearization() writes the real frozen values each control cycle.
        self.set_linearization({"g": np.zeros((self.n_cbf, 2)),
                                "u": np.zeros(self.n_cbf)})

    # ---- fixed Hessian: rho on both z blocks, 2*lambda_slk on the slack block ----
    def _build_P(self):
        diag = np.empty(self.nvar)
        diag[:2 * self.n_z] = self.rho          # (rho/2)||z-c||^2 -> Hessian rho
        diag[2 * self.n_z:] = 2.0 * self.slack_lambda   # phi(s)=lambda s^2 -> 2 lambda
        return sp.diags(diag).tocsc()

    # ---- fixed part of A (the slack columns) + fixed lower bound -----------------
    def _build_fixed(self):
        rows, cols, vals = [], [], []
        for k in range(self.n_cbf):
            scol = 2 * self.n_z + k
            rows.append(self._r_cbf + k); cols.append(scol); vals.append(-1.0)  # -s_k in CBF row
            rows.append(self._r_snn + k); cols.append(scol); vals.append(1.0)   # +s_k in s>=0 row
        self._fixed = (np.array(rows, dtype=int), np.array(cols, dtype=int),
                       np.array(vals, dtype=float))
        lo = np.full(self.nrow, -np.inf)        # CBF rows: -inf <= . <= u_k
        up_base = np.full(self.nrow, np.inf)     # slack rows: 0 <= s <= +inf
        for k in range(self.n_cbf):
            lo[self._r_snn + k] = 0.0
        if self.hard_k0:                         # clamp k=0 slack to 0 -> hard k=0 row
            up_base[self._r_snn + 0] = 0.0
        self._lo = lo
        self._up_base = up_base

    # ---- freeze the CBF coefficients for this control cycle (outside ADMM loop) --
    def set_linearization(self, frozen):
        """frozen = rti_linearizer.linearize_edge(...) (built from the TRUE-BODY xi).
        Writes g into the CBF rows (col_a^i=-g, col_a^j=+g) and u into the bounds.
        Same sparsity pattern every cycle -> OSQP update, never a rebuild."""
        N = self.N
        g = np.asarray(frozen["g"], dtype=float)
        u = np.asarray(frozen["u"], dtype=float)
        assert g.shape == (self.n_cbf, 2) and u.shape == (self.n_cbf,)
        rows_f, cols_f, vals_f = self._fixed
        rc, cc, vc = [], [], []
        for k in range(self.n_cbf):
            aik = C.a_index(k, N)                # a^i_k column in the z^i block (offset 0)
            ajk = self.n_z + C.a_index(k, N)     # a^j_k column in the z^j block (offset 6N)
            r = self._r_cbf + k
            rc += [r, r, r, r]
            cc += [aik, aik + 1, ajk, ajk + 1]
            vc += [-g[k, 0], -g[k, 1], +g[k, 0], +g[k, 1]]   # push-pull
        rows = np.concatenate([rows_f, np.array(rc, dtype=int)])
        cols = np.concatenate([cols_f, np.array(cc, dtype=int)])
        vals = np.concatenate([vals_f, np.array(vc, dtype=float)])
        self._A = sp.csc_matrix((vals, (rows, cols)), shape=(self.nrow, self.nvar))
        up = self._up_base.copy()
        up[self._r_cbf:self._r_cbf + self.n_cbf] = u
        self._u = up
        if self._prob is not None:              # same pattern -> values-only update
            self._prob.update(Ax=self._A.data, l=self._lo, u=self._u)

    # ---- one edge update (per ADMM iteration): only q changes, warm-started ------
    def solve(self, xi_i, xi_j, lam_i, lam_j):
        """min phi(s) + (rho/2)||z^i-xi^i-lam^i||^2 + (rho/2)||z^j-xi^j-lam^j||^2
        s.t. frozen CBF + s>=0. Returns (z_i(6N), z_j(6N), s(N-1))."""
        assert self._A is not None, "call set_linearization() before solve()"
        n_z = self.n_z
        q = np.zeros(self.nvar)
        q[:n_z] = -self.rho * (np.asarray(xi_i, float) + np.asarray(lam_i, float))
        q[n_z:2 * n_z] = -self.rho * (np.asarray(xi_j, float) + np.asarray(lam_j, float))
        # slack block q = 0 (phi is pure quadratic)
        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(P=self._P, q=q, A=self._A, l=self._lo, u=self._u,
                             verbose=False, warm_start=True,
                             eps_abs=1e-6, eps_rel=1e-6, polish=True, max_iter=8000)
        else:
            self._prob.update(q=q)
        res = self._prob.solve()
        w = np.asarray(res.x, dtype=float) if res.x is not None else None
        if w is None or not np.all(np.isfinite(w)):
            return None, None, None
        return w[:n_z].copy(), w[n_z:2 * n_z].copy(), w[2 * n_z:].copy()
