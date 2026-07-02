"""Stage 2 edge-QP matrix-assembly checks (hand-computed CBF row) + a solve smoke.

Checkpoint-2 focus -- the three places the edge CBF row is easiest to get wrong and
where a slip silently breaks realized inter-agent safety:
  1. g_k carries the h_{k+1} symmetric term (g_k = 0.03*e_{k+2} - 0.014*e_{k+1});
  2. a^i and a^j endpoints have OPPOSITE signs (col_a = -g, col_b = +g: push-pull);
  3. e_{k+2}/e_{k+1} come from the true-body xi (via the RTI linearizer), not z.

A varying geometry (e_{k+1} != e_{k+2}) is used so a dropped term / wrong sign /
wrong operating point fails here, not in a silent trajectory.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C            # noqa: E402
import rti_linearizer as rti     # noqa: E402
import edge_subproblem as es     # noqa: E402


def _make_xibar(pos_fn, acc, N=C.N):
    xi = np.zeros(C.xi_dim(N))
    for m in range(1, N + 1):
        xi[C.px_index(m, N)], xi[C.py_index(m, N)] = pos_fn(m)
    for k in range(N):
        xi[C.ax_index(k, N)], xi[C.ay_index(k, N)] = acc
    return xi


def _frozen_varying(N=C.N):
    # dog j fixed at origin, dog i marching +y at 0.1/step -> e_m = (0, 0.1 m).
    xi_i = _make_xibar(lambda m: (0.0, 0.1 * m), (0.0, 0.1), N)
    xi_j = _make_xibar(lambda m: (0.0, 0.0), (0.0, 0.0), N)
    return rti.linearize_edge(xi_i, xi_j, (0.0, 0.0), (0.0, 0.0))


def test_edge_cbf_row_handcalc():
    N = C.N
    fr = _frozen_varying(N)
    edge = es.EdgeSubproblem()
    edge.set_linearization(fr)

    A = np.asarray(edge._A.todense())
    g0 = fr["g"][0]                                      # (0, 0.0046): has h_{k+1} term
    ai0 = C.a_index(0, N)                                # a^i_0 in z^i block (offset 0)
    aj0 = edge.n_z + C.a_index(0, N)                     # a^j_0 in z^j block (offset 6N)
    s0 = 2 * edge.n_z + 0                                # slack_0 column
    row = A[edge._r_cbf + 0]

    # #2 push-pull: a^i side = -g, a^j side = +g (equal magnitude, opposite sign)
    np.testing.assert_allclose(row[ai0:ai0 + 2], -g0, atol=1e-12)
    np.testing.assert_allclose(row[aj0:aj0 + 2], +g0, atol=1e-12)
    assert row[s0] == -1.0                               # -s_k in the soft CBF row

    # #1 the h_{k+1} term really made it in: g0[1] = 0.0046 (not 0.006 if dropped)
    assert abs(row[ai0 + 1] - (-0.0046)) < 1e-12
    assert abs(row[ai0 + 1] - (-0.006)) > 1e-4
    assert abs(row[aj0 + 1] - (+0.0046)) < 1e-12

    # #3 operating point is the true body: u_0 equals the RTI frozen bound
    assert abs(edge._u[edge._r_cbf + 0] - fr["u"][0]) < 1e-12

    # slack >= 0 row: +1 on the slack column
    assert A[edge._r_snn + 0][s0] == 1.0

    # P: rho on the z blocks, 2*lambda_slk on the slack block
    P = np.asarray(edge._P.todense())
    assert abs(P[0, 0] - C.RHO) < 1e-12
    assert abs(P[edge.n_z, edge.n_z] - C.RHO) < 1e-12
    assert abs(P[2 * edge.n_z, 2 * edge.n_z] - 2 * C.SLACK_LAMBDA) < 1e-12


def test_edge_dimensions():
    N = C.N
    edge = es.EdgeSubproblem()
    assert edge.n_z == 6 * N
    assert edge.nvar == 12 * N + (N - 1)                 # z^i, z^j, s
    assert edge._P.shape == (edge.nvar, edge.nvar)
    assert edge._A.shape == (2 * (N - 1), edge.nvar)     # (N-1) CBF + (N-1) slack>=0


def test_edge_k0_is_hard():
    """Hard edge k=0 (stage-1-consistent: k=0 hard, k>=1 soft). The k=0 slack is
    clamped to 0 -- its slack>=0 row upper bound is 0 -- so k=0 is a hard row; the
    k>=1 slacks stay free (upper +inf). The hardened row keeps the FULL symmetric g
    (both endpoints -g/+g, both the h_{k+2} and h_{k+1} terms); hardening only
    removes the slack escape, it does not touch the row coefficients."""
    N = C.N
    fr = _frozen_varying(N)
    edge = es.EdgeSubproblem()                        # hard_k0 defaults True
    edge.set_linearization(fr)
    assert edge._u[edge._r_snn + 0] == 0.0            # k=0 slack clamped to 0 (hard)
    assert np.isinf(edge._u[edge._r_snn + 1])         # k=1 slack free (soft)
    assert np.isinf(edge._u[edge._r_snn + (edge.n_cbf - 1)])   # last k soft
    # the hardened k=0 CBF row is unchanged: full symmetric g with the h_{k+1} term
    A = np.asarray(edge._A.todense())
    g0 = fr["g"][0]
    ai0 = C.a_index(0, N)
    aj0 = edge.n_z + C.a_index(0, N)
    np.testing.assert_allclose(A[edge._r_cbf + 0][ai0:ai0 + 2], -g0, atol=1e-12)
    np.testing.assert_allclose(A[edge._r_cbf + 0][aj0:aj0 + 2], +g0, atol=1e-12)
    assert abs(g0[1] - 0.0046) < 1e-12                # h_{k+1} term still in the hard row


def test_edge_soft_k0_option():
    """hard_k0=False restores the all-soft edge (k=0 slack free) -- keep it explicit."""
    edge = es.EdgeSubproblem(hard_k0=False)
    edge.set_linearization(_frozen_varying())
    assert np.isinf(edge._u[edge._r_snn + 0])


def test_edge_solve_respects_cbf_and_slack():
    """Solve smoke: two dogs on a near-collision frozen geometry. The returned
    copies must satisfy the (soft) CBF rows and s >= 0 (OSQP feasibility), and the
    copies stay near their proximal targets xi+lambda."""
    N = C.N
    # both agents ~head-on close: e_m small so h < 0 -> CBF active
    xi_i = _make_xibar(lambda m: (0.0, +0.15), (0.0, 0.0), N)
    xi_j = _make_xibar(lambda m: (0.0, -0.15), (0.0, 0.0), N)
    fr = rti.linearize_edge(xi_i, xi_j, (0.0, 0.15), (0.0, -0.15))
    edge = es.EdgeSubproblem()
    edge.set_linearization(fr)
    lam_i = np.zeros(C.xi_dim(N))
    lam_j = np.zeros(C.xi_dim(N))
    z_i, z_j, s = edge.solve(xi_i, xi_j, lam_i, lam_j)

    assert z_i.shape == (6 * N,) and z_j.shape == (6 * N,) and s.shape == (N - 1,)
    assert np.all(np.isfinite(z_i)) and np.all(np.isfinite(z_j))
    assert np.all(s >= -1e-6)                            # slack >= 0
    # each CBF row holds at the solution: -g.a^i(z) + g.a^j(z) - s <= u
    for k in range(N - 1):
        ai = z_i[C.a_index(k, N):C.a_index(k, N) + 2]
        aj = z_j[C.a_index(k, N):C.a_index(k, N) + 2]
        lhs = -fr["g"][k] @ ai + fr["g"][k] @ aj - s[k]
        assert lhs <= fr["u"][k] + 1e-5, f"CBF row {k} violated: {lhs} > {fr['u'][k]}"
