"""Stage 2 RTI linearizer -- freeze the inter-agent (edge) HOCBF coefficients once
per control cycle, OUTSIDE the ADMM loop (spec B8). The 15-20 ADMM iterations share
this frozen set; the node and edge subproblems both build on it.

============================ OPERATING-POINT CONTRACT ============================
The linearization operating point is the TRUE-BODY xi ONLY -- never the copies z.

  cold start : xibar = the uncoupled MPC solution, which BY CONSTRUCTION equals z0
               (paper 4.6), so true body and copy coincide -- no ambiguity.
  steady state: xibar = the previous control cycle's xi, shifted one step.

  z NEVER enters linearization.

Why (decisive for >= 3 dogs): the collision geometry e = p^i - p^j is a single
GLOBAL truth. The copies z^{ij,i} are edge-local -- a dog on two edges has two
distinct copies (z^{12,i}, z^{13,i}) -- so no copy uniquely represents "dog i's
position". Only the unique true body xi can build e. The node's own obstacle/
formation linearization also uses xibar; edge and node must share this one operating
point, or the frozen set is incoherent and r_prim will not descend cleanly.
=================================================================================

All coefficients come from constants.py. FULL linearization (mirrors stage-1 node):
a_k enters the barrier through BOTH h_{k+2} (via 3/2 Ts^2) and h_{k+1} (via 1/2
Ts^2), so g_k keeps both gradient terms. The a^j endpoint is the exact opposite
(-g_k); the edge QP applies that sign (col_a=-g on a^i, col_b=+g on a^j).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C  # noqa: E402


def edge_h(e):
    """Inter-agent barrier h = ||e||^2 - D_MIN^2, e = p^i - p^j (2-vector)."""
    e = np.asarray(e, dtype=float)[:2]
    return float(e @ e) - C.D_MIN ** 2


def three_point(h0, h1, h2):
    """Xiong discrete second-order HOCBF three-point value; >= 0 is the constraint.
    COEF_HK2 h_{k+2} + COEF_HK1 h_{k+1} + COEF_HK h_k (all from constants)."""
    return C.COEF_HK2 * h2 + C.COEF_HK1 * h1 + C.COEF_HK * h0


def _positions(xibar, xnow, n):
    """p_m for m=0..n:  p_0 = xnow[:2] (measured),  p_m = xibar state position
    (m=1..n). CONTRACT: xibar is the true body xi, never a copy z."""
    p = np.zeros((n + 1, 2))
    p[0] = np.asarray(xnow, dtype=float)[:2]
    for m in range(1, n + 1):
        p[m] = xibar[C.px_index(m, n):C.px_index(m, n) + 2]
    return p


def shift_xi(xi, n=None):
    """Operating point for the next control cycle: the previous true-body solution
    shifted one step (x_k <- x_{k+1}, a_k <- a_{k+1}; last step held). This is the
    steady-state xibar (cold start uses the uncoupled MPC solution instead)."""
    N = C.N if n is None else int(n)
    out = np.zeros(C.xi_dim(N))
    for k in range(1, N + 1):
        src = min(k + 1, N)
        out[C.x_index(k, N):C.x_index(k, N) + 4] = xi[C.x_index(src, N):C.x_index(src, N) + 4]
    for k in range(N):
        src = min(k + 1, N - 1)
        out[C.a_index(k, N):C.a_index(k, N) + 2] = xi[C.a_index(src, N):C.a_index(src, N) + 2]
    return out


def realized_hbar(xi_i, xi_j, xnow_i, xnow_j, n=None):
    """Realized inter-agent three-point HOCBF value per k=0..N-2 at the TRUE-BODY
    positions (spec C5.4 h2_viol). >= 0 means the nonlinear HOCBF holds; the
    coordinator reports h2_viol = max_k (-realized_hbar_k)^+ ."""
    N = C.N if n is None else int(n)
    pi = _positions(xi_i, xnow_i, N)
    pj = _positions(xi_j, xnow_j, N)
    e = pi - pj
    h = np.array([edge_h(e[m]) for m in range(N + 1)])
    return np.array([three_point(h[k], h[k + 1], h[k + 2]) for k in range(N - 1)])


def linearize_edge(xibar_i, xibar_j, xnow_i, xnow_j, n=None):
    """Freeze the edge inter-agent CBF for accel-steps k=0..n-2. Returns a dict:

      g      (n-1, 2)  d(three-point)/d a^i_k  (full linearization)
                       = COEF_HK2*CBF_CONSTR_COEF*e_{k+2}
                         + COEF_HK1*CBF_CONSTR_COEF_P1*e_{k+1}
                       (a^j endpoint uses -g; applied by the edge QP)
      u      (n-1,)    edge-row bound = Hbar_k - g_k . (abar_i_k - abar_j_k)
      Hbar   (n-1,)    three-point value at the operating point
      abar_i,abar_j (n-1, 2)  frozen accels a^i_k, a^j_k
      e      (n+1, 2)  relative positions p^i_m - p^j_m (m=0..n; diagnostics/tests)
      N      int

    Everything is built from the true-body xibar (see the module contract).
    """
    N = C.N if n is None else int(n)
    pi = _positions(xibar_i, xnow_i, N)
    pj = _positions(xibar_j, xnow_j, N)
    e = pi - pj                                              # (N+1, 2)
    h = np.array([edge_h(e[m]) for m in range(N + 1)])       # (N+1,)

    nk = N - 1                                               # k = 0..N-2
    g = np.zeros((nk, 2))
    u = np.zeros(nk)
    Hbar = np.zeros(nk)
    abar_i = np.zeros((nk, 2))
    abar_j = np.zeros((nk, 2))
    for k in range(nk):
        Hbar[k] = three_point(h[k], h[k + 1], h[k + 2])
        g[k] = (C.COEF_HK2 * C.CBF_CONSTR_COEF * e[k + 2]
                + C.COEF_HK1 * C.CBF_CONSTR_COEF_P1 * e[k + 1])
        abar_i[k] = xibar_i[C.ax_index(k, N):C.ax_index(k, N) + 2]
        abar_j[k] = xibar_j[C.ax_index(k, N):C.ax_index(k, N) + 2]
        u[k] = Hbar[k] - float(g[k] @ (abar_i[k] - abar_j[k]))
    return dict(g=g, u=u, Hbar=Hbar, abar_i=abar_i, abar_j=abar_j, e=e, N=N)
