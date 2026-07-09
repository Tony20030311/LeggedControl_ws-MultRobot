"""Stage 2 RTI-linearizer unit checks (hand-computed geometry).

Pins the edge inter-agent HOCBF coefficients that are easy to get subtly wrong and
would silently break realized inter-agent safety:
  - g_k carries BOTH the h_{k+2} (0.03) and the h_{k+1} (0.01) gradient terms
    (full linearization -- dropping h_{k+1} changes g and under-brakes);
  - g_k uses e_{k+2} for the 0.03 term and e_{k+1} for the 0.01 term (right index);
  - e_m is built from the TRUE-BODY xi positions (the contract), not the copies z.

A varying geometry (e_{k+1} != e_{k+2}) is used on purpose so a wrong index or a
dropped term fails here, not in a silent trajectory.

Run (in-container inline runner): call every test_* function.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C          # noqa: E402
# ADMM_IMPL=cpp runs this same gate against the C++ port (admm_core_cpp).
if os.environ.get("ADMM_IMPL", "python") == "cpp":
    from admm_core_cpp import rti  # noqa: E402
else:
    import rti_linearizer as rti   # noqa: E402


def _make_xibar(pos_fn, acc, N=C.N):
    """Flat 6N true-body vector: state positions p_m = pos_fn(m) (m=1..N),
    accels a_k = acc (k=0..N-1); velocities left 0 (unused by the edge CBF)."""
    xi = np.zeros(C.xi_dim(N))
    for m in range(1, N + 1):
        xi[C.px_index(m, N)], xi[C.py_index(m, N)] = pos_fn(m)
    for k in range(N):
        xi[C.ax_index(k, N)], xi[C.ay_index(k, N)] = acc
    return xi


def test_linearize_edge_handcalc():
    N = C.N
    # dog j fixed at origin; dog i marches along +y at 0.1 / step -> e_m = (0, 0.1 m).
    xi_i = _make_xibar(lambda m: (0.0, 0.1 * m), (0.0, 0.1), N)
    xi_j = _make_xibar(lambda m: (0.0, 0.0), (0.0, 0.0), N)
    fr = rti.linearize_edge(xi_i, xi_j, xnow_i=(0.0, 0.0), xnow_j=(0.0, 0.0))

    # e_m from TRUE-BODY positions: e_0 = xnow diff, e_m = p^i_m - p^j_m
    np.testing.assert_allclose(fr["e"][0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fr["e"][1], [0.0, 0.1], atol=1e-12)
    np.testing.assert_allclose(fr["e"][2], [0.0, 0.2], atol=1e-12)

    # g_0 = COEF_HK2*0.03*e_2 + COEF_HK1*0.01*e_1
    #     = 0.03*(0,0.2) + (-1.4*0.01)*(0,0.1) = (0, 0.006 - 0.0014) = (0, 0.0046)
    np.testing.assert_allclose(fr["g"][0], [0.0, 0.0046], atol=1e-12)
    # if the h_{k+1} term were dropped, g_0 would be 0.03*e_2 = (0, 0.006) -- guard it
    assert abs(fr["g"][0][1] - 0.006) > 1e-4, "h_{k+1} term missing from g_k"

    # Hbar_0 = h_2 - 1.4 h_1 + 0.49 h_0 ,  h_m = ||e_m||^2 - D_MIN^2
    def h(m):
        return (0.1 * m) ** 2 - C.D_MIN ** 2
    Hbar0 = C.COEF_HK2 * h(2) + C.COEF_HK1 * h(1) + C.COEF_HK * h(0)
    assert abs(fr["Hbar"][0] - Hbar0) < 1e-12

    # u_0 = Hbar_0 - g_0 . (abar_i_0 - abar_j_0) ,  abar diff = (0, 0.1)
    u0 = Hbar0 - float(np.array([0.0, 0.0046]) @ np.array([0.0, 0.1]))
    assert abs(fr["u"][0] - u0) < 1e-12

    # shapes: k = 0..N-2
    assert fr["g"].shape == (N - 1, 2)
    assert fr["u"].shape == (N - 1,)
    assert fr["e"].shape == (N + 1, 2)


def test_linearize_uses_xnow_at_step0():
    # e_0 must come from the MEASURED state (xnow), not xibar[step 1].
    N = C.N
    xi_i = _make_xibar(lambda m: (1.0, 0.0), (0.0, 0.0), N)   # xibar positions at x=1
    xi_j = _make_xibar(lambda m: (0.0, 0.0), (0.0, 0.0), N)
    fr = rti.linearize_edge(xi_i, xi_j, xnow_i=(0.3, 0.0), xnow_j=(0.0, 0.0))
    np.testing.assert_allclose(fr["e"][0], [0.3, 0.0], atol=1e-12)   # xnow diff
    np.testing.assert_allclose(fr["e"][1], [1.0, 0.0], atol=1e-12)   # xibar diff
