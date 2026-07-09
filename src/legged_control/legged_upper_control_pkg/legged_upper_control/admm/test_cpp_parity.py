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
import node_subproblem as PY_NQ  # noqa: E402
import edge_subproblem as PY_EQ  # noqa: E402

import admm_core_cpp  # noqa: E402
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


# ---------------- C3: node / edge QP ----------------

_OBSTACLES = [{"pos": (2.0, 0.15), "radius": 0.30},
              {"pos": (3.1, -0.4), "radius": 0.45}]
_WALLS = [{"normal": (0.0, 1.0), "point": (0.0, -1.5), "d_safe": 0.4}]


def _mk_nodes(n, rho_consensus=0.0, n_neighbors=0, w_form=0.0):
    py_n = PY_NQ.NodeSubproblem(obstacles=_OBSTACLES, walls=_WALLS, n=n,
                                rho_consensus=rho_consensus,
                                n_neighbors=n_neighbors, w_form=w_form)
    cpp_n = admm_core_cpp.NodeSubproblem(obstacles=_OBSTACLES, walls=_WALLS, n=n,
                                         rho_consensus=rho_consensus,
                                         n_neighbors=n_neighbors, w_form=w_form)
    return py_n, cpp_n


def _node_inputs(rng, n):
    x_now = rng.uniform(-0.3, 0.3, 4)
    x_des = np.zeros((n, 4))
    x_des[:, 0] = np.linspace(0.2, 4.0, n) + rng.uniform(-0.05, 0.05, n)
    x_des[:, 1] = rng.uniform(-0.4, 0.4, n)
    xbar = rng.uniform(-2.0, 4.0, PY.xi_dim(n))
    ct = rng.uniform(-1.0, 1.0, PY.xi_dim(n))
    fg = rng.uniform(-0.5, 0.5, (n, 2))
    return x_now, x_des, xbar, ct, fg


def test_c3_node_P_bit_identical():
    for n in (10, 20):
        for cons in ((0.0, 0), (PY.RHO, 2)):
            py_n, cpp_n = _mk_nodes(n, rho_consensus=cons[0], n_neighbors=cons[1])
            P = py_n._P
            p, i, x = cpp_n._debug_P()
            assert np.array_equal(P.indptr, np.asarray(p)), f"P indptr n={n}"
            assert np.array_equal(P.indices, np.asarray(i)), f"P indices n={n}"
            assert np.array_equal(P.data, np.asarray(x)), f"P data n={n}"


def test_c3_node_assembly_bit_identical():
    rng = np.random.default_rng(21)
    for n in (10, 20):
        py_n, cpp_n = _mk_nodes(n, rho_consensus=PY.RHO, n_neighbors=2, w_form=1.5)
        for _ in range(50):
            x_now, x_des, xbar, ct, fg = _node_inputs(rng, n)
            q_py = py_n._q(x_des, consensus_target=ct, formation_grad=fg)
            pbar, abar = py_n._operating_point(x_now, xbar)
            A, up_cbf = py_n._assemble_A(pbar, abar)
            lo, up = py_n._lu(x_now, up_cbf.copy())
            d = cpp_n._debug_pass(x_now, x_des, xbar, consensus_target=ct,
                                  formation_grad=fg)
            ap, ai = cpp_n._debug_A_pattern()
            assert np.array_equal(A.indptr, np.asarray(ap))
            assert np.array_equal(A.indices, np.asarray(ai))
            assert np.array_equal(A.data, np.asarray(d["Ax"])), "A data"
            assert np.array_equal(q_py, np.asarray(d["q"])), "q"
            assert np.array_equal(lo, np.asarray(d["lo"])), "lo"
            assert np.array_equal(up, np.asarray(d["up"])), "up"


def _bits_equal(a, b):
    """Bitwise equality including NaN (an infeasible solve returns NaN-filled x;
    NaN != NaN makes np.array_equal a false negative there)."""
    a, b = np.ascontiguousarray(a, float), np.ascontiguousarray(b, float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def test_c3_node_solve_bit_identical():
    rng = np.random.default_rng(23)
    for n in (10, 20):
        py_n, cpp_n = _mk_nodes(n, rho_consensus=PY.RHO, n_neighbors=2, w_form=1.5)
        x_now, x_des, _, ct, fg = _node_inputs(rng, n)
        # cold start (xbar=None, soft warm-up path), then steady-state chain
        rp = py_n.solve(x_now, x_des)
        rc = cpp_n.solve(x_now, x_des)
        assert rp["status"] == rc["status"], (rp["status"], rc["status"])
        assert _bits_equal(rp["xi"], rc["xi"]), f"cold-start xi differs n={n}"
        for step in range(4):
            xbar = np.asarray(rp["xi"], float)[:PY.xi_dim(n)]
            x_now2 = rng.uniform(-0.3, 0.3, 4)
            rp = py_n.solve(x_now2, x_des, xbar=xbar, consensus_target=ct,
                            formation_grad=fg)
            rc = cpp_n.solve(x_now2, x_des, xbar=xbar, consensus_target=ct,
                             formation_grad=fg)
            assert rp["status"] == rc["status"], f"status step={step}"
            assert _bits_equal(rp["xi"], rc["xi"]), f"steady xi differs step={step}"
            if rp["a0"] is not None or rc["a0"] is not None:
                assert _bits_equal(rp["a0"], rc["a0"])


def _mk_edges(n, hard_through=1):
    return (PY_EQ.EdgeSubproblem(n=n, hard_through=hard_through),
            admm_core_cpp.EdgeSubproblem(n=n, hard_through=hard_through))


def test_c3_edge_static_bit_identical():
    for n in (10, 20):
        for k_hard in (0, 1, 3):
            py_e, cpp_e = _mk_edges(n, hard_through=k_hard)
            p, i, x = cpp_e._debug_P()
            assert np.array_equal(py_e._P.indptr, np.asarray(p))
            assert np.array_equal(py_e._P.indices, np.asarray(i))
            assert np.array_equal(py_e._P.data, np.asarray(x))
            ap, ai, ax = cpp_e._debug_A()
            assert np.array_equal(py_e._A.indptr, np.asarray(ap))
            assert np.array_equal(py_e._A.indices, np.asarray(ai))
            assert np.array_equal(py_e._A.data, np.asarray(ax))
            lo, u = cpp_e._debug_lu()
            assert np.array_equal(py_e._lo, np.asarray(lo))
            assert np.array_equal(py_e._u, np.asarray(u))


def test_c3_edge_solve_bit_identical():
    rng = np.random.default_rng(31)
    for n in (10, 20):
        py_e, cpp_e = _mk_edges(n, hard_through=1)
        for cycle in range(3):  # a fresh linearization per "control cycle"
            args = _random_edge_inputs(rng, n)
            fr = PY_RTI.linearize_edge(*args, n=n)
            py_e.set_linearization(fr)
            cpp_e.set_linearization(fr)
            ap, ai, ax = cpp_e._debug_A()
            assert np.array_equal(py_e._A.data, np.asarray(ax)), "frozen A data"
            assert np.array_equal(py_e._u, np.asarray(cpp_e._debug_lu()[1]))
            for it in range(4):  # ADMM iterations: q-only updates, warm-started
                xi_i = rng.uniform(-2.0, 2.0, PY.xi_dim(n))
                xi_j = rng.uniform(-2.0, 2.0, PY.xi_dim(n))
                lam_i = rng.uniform(-0.5, 0.5, PY.xi_dim(n))
                lam_j = rng.uniform(-0.5, 0.5, PY.xi_dim(n))
                zp = py_e.solve(xi_i, xi_j, lam_i, lam_j)
                zc = cpp_e.solve(xi_i, xi_j, lam_i, lam_j)
                assert (zp[0] is None) == (zc[0] is None)
                if zp[0] is not None:
                    for a, b, name in zip(zp, zc, ("z_i", "z_j", "s")):
                        assert np.array_equal(a, b), \
                            f"{name} differs n={n} cycle={cycle} iter={it}"


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
