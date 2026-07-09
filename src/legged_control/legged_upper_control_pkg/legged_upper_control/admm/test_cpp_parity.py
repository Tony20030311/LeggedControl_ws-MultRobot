"""C++ port parity gate — bit-identity between the Python golden reference and
admm_core_cpp, stage by stage. Grows as the port advances (C1: constants).

Run (inside the SIL container, devel sourced):
    python3 legged_upper_control/admm/test_cpp_parity.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import constants as PY  # noqa: E402
import rti_linearizer as PY_RTI  # noqa: E402

from admm_core_cpp import constants as CPP  # noqa: E402
from admm_core_cpp import rti as CPP_RTI  # noqa: E402

_SCALARS = [
    "N", "TS", "GAMMA1", "GAMMA2", "N_X", "N_U", "XI_DIM",
    "MAX_VX", "MAX_VY", "MAX_AX", "MAX_AY",
    "BD_POS_COEF", "BD_VEL_COEF",
    "COEF_HK2", "COEF_HK1", "COEF_HK",
    "A_TO_P2_COEF", "CBF_CONSTR_COEF", "A_TO_P1_COEF", "CBF_CONSTR_COEF_P1",
    "RHO", "SLACK_LAMBDA", "D_MIN", "P_ITERS", "K_SEND",
]
_ARRAYS = ["Ad", "Bd", "HOCBF_COEFS"]
_INDEX_FNS = ["xi_dim", "x_index", "px_index", "py_index", "vx_index",
              "vy_index", "a_index", "ax_index", "ay_index"]


def test_c1_scalars_bit_identical():
    for name in _SCALARS:
        p, q = getattr(PY, name), getattr(CPP, name)
        assert p == q, f"{name}: py={p!r} cpp={q!r}"


def test_c1_arrays_bit_identical():
    for name in _ARRAYS:
        p, q = np.asarray(getattr(PY, name)), np.asarray(getattr(CPP, name))
        assert p.shape == q.shape, f"{name} shape: {p.shape} vs {q.shape}"
        assert np.array_equal(p, q), f"{name} differs:\n{p}\n{q}"


def test_c1_index_formulas_identical():
    for n in (10, 12, 20, 40):
        assert PY.xi_dim(n) == CPP.xi_dim(n)
        for k in range(1, n + 1):
            for fn in ("x_index", "px_index", "py_index", "vx_index", "vy_index"):
                assert getattr(PY, fn)(k, n) == getattr(CPP, fn)(k, n), (fn, k, n)
        for k in range(0, n):
            for fn in ("a_index", "ax_index", "ay_index"):
                assert getattr(PY, fn)(k, n) == getattr(CPP, fn)(k, n), (fn, k, n)


def test_c2_rti_scalars_bit_identical():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        e = rng.uniform(-5.0, 5.0, 2)
        assert PY_RTI.edge_h(e) == CPP_RTI.edge_h(e), f"edge_h({e})"
        h0, h1, h2 = rng.uniform(-3.0, 3.0, 3)
        assert PY_RTI.three_point(h0, h1, h2) == CPP_RTI.three_point(h0, h1, h2)


def test_c2_shift_xi_bit_identical():
    rng = np.random.default_rng(7)
    for n in (10, 20):
        for _ in range(200):
            xi = rng.standard_normal(PY.xi_dim(n))
            assert np.array_equal(PY_RTI.shift_xi(xi, n), CPP_RTI.shift_xi(xi, n))
    xi = rng.standard_normal(PY.xi_dim(PY.N))
    assert np.array_equal(PY_RTI.shift_xi(xi), CPP_RTI.shift_xi(xi))  # default n


def _random_edge_inputs(rng, n):
    xi_i = rng.uniform(-4.0, 4.0, PY.xi_dim(n))
    xi_j = rng.uniform(-4.0, 4.0, PY.xi_dim(n))
    xnow_i = rng.uniform(-4.0, 4.0, 2)
    xnow_j = rng.uniform(-4.0, 4.0, 2)
    return xi_i, xi_j, xnow_i, xnow_j


def test_c2_realized_hbar_bit_identical():
    rng = np.random.default_rng(11)
    for n in (10, 20):
        for _ in range(200):
            args = _random_edge_inputs(rng, n)
            assert np.array_equal(PY_RTI.realized_hbar(*args, n=n),
                                  CPP_RTI.realized_hbar(*args, n=n))


def test_c2_linearize_edge_bit_identical():
    rng = np.random.default_rng(13)
    for n in (10, 20):
        for _ in range(200):
            args = _random_edge_inputs(rng, n)
            p = PY_RTI.linearize_edge(*args, n=n)
            q = CPP_RTI.linearize_edge(*args, n=n)
            assert p["N"] == q["N"]
            for key in ("g", "u", "Hbar", "abar_i", "abar_j", "e"):
                pk, qk = np.asarray(p[key]), np.asarray(q[key])
                assert pk.shape == qk.shape, (key, pk.shape, qk.shape)
                assert np.array_equal(pk, qk), f"{key} differs (n={n})"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_") and callable(v)}.items()):
        try:
            fn()
            print(f"   [OK ] {name}")
        except AssertionError as e:
            failed += 1
            print(f"   [FAIL] {name}: {e}")
    print("\n[cpp-parity] " + ("ALL GREEN" if failed == 0 else f"{failed} FAILED"))
    sys.exit(1 if failed else 0)
