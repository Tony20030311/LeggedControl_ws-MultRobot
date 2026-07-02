"""Stage 1 node-QP matrix-assembly checks.

Not the behavioral gate (that's verify_stage1.py). These pin the pieces that are
easy to get subtly wrong: decision-vector dimensions, the dynamics equality
signs, and the two CBF-row derivations (obstacle 0.03, wall 0.015) against a
hand-computed geometry -- so a sign flip fails here, not in a silent trajectory.

Run: pytest legged_upper_control/admm/test_node_qp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C          # noqa: E402
import node_subproblem as nq   # noqa: E402


def test_dimensions():
    node = nq.NodeSubproblem(obstacles=[{"pos": (2.0, 0.4), "radius": 0.35}],
                             walls=[], n=C.N)
    assert node.n_xi == 6 * C.N == 120
    assert node.n_slack == 1 * (C.N - 2)             # 1 feature, k=1..N-2
    assert node.nvar == node.n_xi + node.n_slack
    assert node._P.shape == (node.nvar, node.nvar)
    A, up = node._assemble_A(np.zeros((C.N + 1, 2)), np.zeros((C.N, 2)))
    assert A.shape == (node.nrow, node.nvar)
    assert len(up) == node.nrow


def test_dynamics_equality_roundtrip():
    """A valid Ad/Bd rollout must satisfy A_eq z == b_eq exactly."""
    node = nq.NodeSubproblem(obstacles=[], walls=[], n=C.N)
    N = C.N
    rng = np.random.default_rng(0)
    x_now = np.array([0.1, -0.2, 0.05, 0.0])
    accels = rng.uniform(-0.5, 0.5, size=(N, 2))
    xs = [x_now.copy()]
    for k in range(N):
        xs.append(C.Ad @ xs[-1] + C.Bd @ accels[k])
    xi = np.zeros(node.n_xi)
    for k in range(1, N + 1):
        xi[C.x_index(k, N):C.x_index(k, N) + 4] = xs[k]
    for k in range(N):
        xi[C.a_index(k, N):C.a_index(k, N) + 2] = accels[k]
    z = np.concatenate([xi, np.zeros(node.n_slack)])
    A, up = node._assemble_A(np.zeros((N + 1, 2)), np.zeros((N, 2)))
    lo, up = node._lu(x_now, up)
    Aeq = A[:4 * N, :]
    beq = lo[:4 * N]                                  # equality: l == u
    assert np.allclose(Aeq @ z, beq, atol=1e-9)


def test_cbf_row_obstacle_handcomputed():
    obs = {"pos": (2.0, 0.0), "radius": 0.30}
    node = nq.NodeSubproblem(obstacles=[obs], walls=[], robot_margin=0.20, n=C.N)
    r_eff = 0.30 + 0.20
    p_obs = np.array([2.0, 0.0])
    pbar = np.zeros((C.N + 1, 2))
    pbar[0], pbar[1], pbar[2] = [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]
    abar = np.zeros((C.N, 2))
    coef, b = node._cbf_row(node._features()[0], 0, pbar, abar)

    # full linearization: a_k enters h_{k+2} (0.03) and h_{k+1} (0.01);
    # g = COEF_HK2*0.03*e2 + COEF_HK1*0.01*e1 = 0.03*e2 - 0.014*e1
    e1, e2 = pbar[1] - p_obs, pbar[2] - p_obs
    g = 0.03 * e2 - 0.014 * e1
    assert np.allclose(coef, -g, atol=1e-12), (coef, -g)

    def h(p):
        e = p - p_obs
        return float(e @ e) - r_eff ** 2
    exp_b = h(pbar[2]) + C.COEF_HK1 * h(pbar[1]) + C.COEF_HK * h(pbar[0])
    assert abs(b - exp_b) < 1e-12, (b, exp_b)

    # increment-form fold: nonzero abar_0 shifts b by -g . abar_0
    abar[0] = [0.5, -0.3]
    _, b2 = node._cbf_row(node._features()[0], 0, pbar, abar)
    assert abs((b2 - exp_b) - (-float(g @ abar[0]))) < 1e-12


def test_cbf_row_wall_handcomputed():
    wall = {"normal": (0.0, 1.0), "point": (0.0, -1.0), "d_safe": 0.3}
    node = nq.NodeSubproblem(obstacles=[], walls=[wall], n=C.N)
    n_w, p_w, d = np.array([0.0, 1.0]), np.array([0.0, -1.0]), 0.3
    pbar = np.zeros((C.N + 1, 2))
    pbar[0], pbar[1], pbar[2] = [0.0, -0.5], [0.0, -0.4], [0.0, -0.3]
    abar = np.zeros((C.N, 2))
    coef, b = node._cbf_row(node._features()[0], 0, pbar, abar)
    # full linearization (affine wall, exact): g = (0.015 + (-1.4)*0.005)*n_w
    g = (0.015 - 0.007) * n_w
    assert np.allclose(coef, -g, atol=1e-12)

    def hw(p):
        return float(n_w @ (p - p_w)) - d
    exp_b = hw(pbar[2]) + C.COEF_HK1 * hw(pbar[1]) + C.COEF_HK * hw(pbar[0])
    assert abs(b - exp_b) < 1e-12
