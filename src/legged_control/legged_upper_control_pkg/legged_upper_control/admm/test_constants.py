"""Stage 0 gate — pins the documented coefficients so a sign/discretization slip
fails here instead of silently corrupting behavior downstream.

Values are the binding literals from docs/spec_updates_v2.md details 1, 2, 4.
Run: pytest legged_upper_control/admm/test_constants.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import constants as C  # noqa: E402


# --- detail 2: second-order HOCBF coefficients (gamma1 gamma2 POSITIVE) --------
def test_hocbf_coefficients():
    g1 = g2 = 0.3
    coef_hk = 1 - g1 - g2 + g1 * g2       # positive g1 g2
    coef_hk1 = -(2 - g1 - g2)
    assert abs(coef_hk - 0.49) < 1e-9, f"h_k coef wrong! {coef_hk} should be 0.49 (positive)"
    assert abs(coef_hk1 - (-1.4)) < 1e-9
    # the module must agree with the hand-computed literals
    assert abs(C.COEF_HK - 0.49) < 1e-9
    assert abs(C.COEF_HK1 - (-1.4)) < 1e-9
    assert abs(C.COEF_HK2 - 1.0) < 1e-9


def test_hocbf_positive_sign_trap():
    # the classic bug: a negative g1 g2 gives 0.31, off by 37% — must NOT match
    assert abs(C.COEF_HK - 0.31) > 1e-3
    np.testing.assert_allclose(C.HOCBF_COEFS, [0.49, -1.4, 1.0], atol=1e-9)


# --- detail 1: exact Bd and CBF-constraint scalars ----------------------------
def test_bd_coefficients():
    Ts = 0.1
    assert abs(1.5 * Ts**2 - 0.015) < 1e-9   # a_k -> p_{k+2}
    assert abs(3.0 * Ts**2 - 0.03) < 1e-9    # linearized CBF constraint coef
    assert abs(C.A_TO_P2_COEF - 0.015) < 1e-9
    assert abs(C.CBF_CONSTR_COEF - 0.03) < 1e-9
    # h_{k+1} gradient coeffs (full linearization: a_k also reaches p_{k+1})
    assert abs(C.A_TO_P1_COEF - 0.005) < 1e-9       # a_k -> p_{k+1} = 1/2 Ts^2
    assert abs(C.CBF_CONSTR_COEF_P1 - 0.01) < 1e-9  # obstacle h_{k+1} grad = Ts^2


def test_bd_matrix_exact():
    Ts = C.TS
    expected = np.array([[0.5 * Ts**2, 0.0],
                         [0.0, 0.5 * Ts**2],
                         [Ts, 0.0],
                         [0.0, Ts]])
    np.testing.assert_allclose(C.Bd, expected, atol=1e-12)
    assert abs(C.BD_POS_COEF - 0.005) < 1e-12
    assert abs(C.BD_VEL_COEF - 0.1) < 1e-12


def test_ad_matrix_and_a_to_p2_consistency():
    Ts = C.TS
    expected_ad = np.array([[1, 0, Ts, 0],
                            [0, 1, 0, Ts],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
    np.testing.assert_allclose(C.Ad, expected_ad, atol=1e-12)
    # cross-check: a_k -> p_{k+2} coef IS the position block of Ad @ Bd
    ad_bd = C.Ad @ C.Bd
    np.testing.assert_allclose(np.diag(ad_bd[:2, :]), [C.A_TO_P2_COEF] * 2, atol=1e-12)


# --- detail 4: index formulas use 4*N, never hardcoded offsets -----------------
def test_index_formulas_n20():
    assert C.N == 20 and C.XI_DIM == 120
    assert C.x_index(1) == 0 and C.x_index(20) == 76      # x_1@0-3, x_20@76-79
    assert C.a_index(0) == 80 and C.a_index(19) == 118    # a_0@80-81, a_19@118-119
    assert (C.px_index(5), C.py_index(5), C.vx_index(5), C.vy_index(5)) == (16, 17, 18, 19)
    assert (C.ax_index(3), C.ay_index(3)) == (86, 87)


def test_index_formulas_scale_with_n():
    # N=10 must relocate the accel block — guards against a hardcoded 80
    assert C.a_index(0, n=10) == 40 and C.xi_dim(10) == 60
    assert C.a_index(0, n=20) == 80
