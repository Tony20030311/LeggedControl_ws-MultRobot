"""C++ port parity gate — bit-identity between the Python golden reference and
admm_core_cpp, stage by stage. Grows as the port advances (C1: constants).

Run (inside the SIL container, devel sourced):
    python3 legged_upper_control/admm/test_cpp_parity.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import constants as PY  # noqa: E402
import rti_linearizer as PY_RTI  # noqa: E402
import node_subproblem as PY_NQ  # noqa: E402
import edge_subproblem as PY_EQ  # noqa: E402
import admm_coordinator as PY_AC  # noqa: E402
import motion_adapter as PY_MA  # noqa: E402
import reference as PY_REF  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.formation import LaplacianFormation as PyFormation  # noqa: E402
from core.planning import AStarPlanner as PyAStar  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts"))
from ocs2_fleet_publisher import ARENAS as PUB_ARENAS  # noqa: E402
from ocs2_fleet_publisher import (  # noqa: E402
    DEFAULT_GOALS as PUB_DEFAULT_GOALS,
    FORMATIONS as PUB_FORMATIONS,
    centroid_slot_targets as py_centroid_slot_targets,
    min_cost_assignment as py_min_cost_assignment,
    rot2d as py_rot2d,
)

import admm_core_cpp  # noqa: E402
from admm_core_cpp import constants as CPP  # noqa: E402
from admm_core_cpp import rti as CPP_RTI  # noqa: E402
from admm_core_cpp import motion_adapter as CPP_MA  # noqa: E402
from admm_core_cpp import coordinator as CPP_AC  # noqa: E402
from admm_core_cpp import reference as CPP_REF  # noqa: E402

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


# ---------------- C4: formation / motion adapter / coordinator ----------------

_FORMATIONS = {"V": [[0.67, 0.0], [-0.33, 1.0], [-0.33, -1.0]],
               "line": [[1.2, 0.0], [0.0, 0.0], [-1.2, 0.0]]}


def test_c4_formation_bit_identical():
    rng = np.random.default_rng(41)
    py_f = PyFormation({k: [np.array(p) for p in v] for k, v in _FORMATIONS.items()})
    cpp_f = admm_core_cpp.LaplacianFormation(_FORMATIONS)
    for name in ("V", "line"):
        py_f.set_formation(name)
        cpp_f.set_formation(name)
        assert py_f.current_formation == cpp_f.current_formation
        for _ in range(300):
            pos = [rng.uniform(-3.0, 3.0, 2) for _ in range(3)]
            fp, gp = py_f.compute(pos)
            fc, gc = cpp_f.compute(pos)
            assert _bits_equal([fp], [fc]), f"f differs ({name})"
            for a, b in zip(gp, gc):
                assert _bits_equal(a, b), f"grad differs ({name})"


def test_c4_motion_adapter_bit_identical():
    rng = np.random.default_rng(43)
    joints = rng.uniform(-1.0, 1.0, 12)
    kwargs = dict(com_height=0.3, default_joints=joints, max_yaw_rate=1.2)
    py_m = PY_MA.MotionAdapter(**kwargs)
    cpp_m = CPP_MA.MotionAdapter(**kwargs)
    for a in rng.uniform(-20.0, 20.0, 500):  # wrap_to_pi randomized
        assert PY_MA.wrap_to_pi(a) == CPP_MA.wrap_to_pi(a), f"wrap_to_pi({a})"
    for cyc in range(6):  # stateful adapt() chain, both dogs
        xi = rng.uniform(-3.0, 3.0, PY.xi_dim(PY.N))
        for dog in (1, 2):
            tp = py_m.adapt(xi, dog, t0=0.7 * cyc)
            tc = cpp_m.adapt(xi, dog, t0=0.7 * cyc)
            for key in ("times", "states", "inputs"):
                assert _bits_equal(np.asarray(tp[key]), np.asarray(tc[key])), \
                    f"{key} differs cyc={cyc} dog={dog}"
            assert tp["next_seed"] == tc["next_seed"]
    # tangent mode + explicit k_send override
    xi = rng.uniform(-3.0, 3.0, PY.xi_dim(PY.N))
    tp = py_m.build_target(xi, seed_yaw=0.3, yaw_mode="tangent", k_send=7)
    tc = cpp_m.build_target(xi, seed_yaw=0.3, yaw_mode="tangent", k_send=7)
    for key in ("times", "states", "inputs"):
        assert _bits_equal(np.asarray(tp[key]), np.asarray(tc[key])), key
    # safe_prefix_length randomized
    for _ in range(200):
        n = 10
        px_i, py_i = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
        others = [(rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)) for _ in range(2)]
        assert (PY_MA.safe_prefix_length(px_i, py_i, others, 0.6, 10)
                == CPP_MA.safe_prefix_length(px_i, py_i, others, 0.6, 10))


def _coordinator_pair(dogs, edges, with_formation):
    obstacles = [{"pos": (2.0, 0.2), "radius": 0.30}]
    walls = [{"normal": (0.0, 1.0), "point": (0.0, -2.0), "d_safe": 0.4}]
    py_f = cpp_f = None
    if with_formation:
        py_f = PyFormation({k: [np.array(p) for p in v] for k, v in _FORMATIONS.items()})
        py_f.set_formation("V")
        cpp_f = admm_core_cpp.LaplacianFormation(_FORMATIONS)
        cpp_f.set_formation("V")
    py_c = PY_AC.ADMMCoordinator(dogs=dogs, edges=edges, formation=py_f,
                                 w_form=1.0 if with_formation else 0.0,
                                 obstacles=obstacles, walls=walls, hard_through=1)
    cpp_c = CPP_AC.ADMMCoordinator(dogs=dogs, edges=edges, formation=cpp_f,
                                   w_form=1.0 if with_formation else 0.0,
                                   obstacles=obstacles, walls=walls, hard_through=1)
    return py_c, cpp_c


def _coordinator_chain(dogs, edges, with_formation, cycles=3):
    rng = np.random.default_rng(47)
    py_c, cpp_c = _coordinator_pair(dogs, edges, with_formation)
    N = PY.N
    starts = {d: np.array([0.0, 1.0 * i, 0.0, 0.0])
              for i, d in enumerate(dogs)}
    for cyc in range(cycles):
        xnow = {d: starts[d] + np.concatenate([rng.uniform(-0.05, 0.05, 2), [0, 0]])
                for d in dogs}
        xdes = {}
        for i, d in enumerate(dogs):
            xd = np.zeros((N, 4))
            xd[:, 0] = np.linspace(0.2, 4.0, N)
            xd[:, 1] = 1.0 * i
            xdes[d] = xd
        xp, hp = py_c.step(xnow, xdes)
        xc, hc = cpp_c.step(xnow, xdes)
        for d in dogs:
            assert _bits_equal(xp[d], xc[d]), \
                f"xi differs dog={d} cycle={cyc} (formation={with_formation})"
        for key in ("r_prim", "r_dual", "h2_viol"):
            assert _bits_equal(np.asarray(hp[key]), np.asarray(hc[key])), \
                f"hist {key} differs cycle={cyc}"
        assert hp["edge_fail"] == hc["edge_fail"]


def test_c5_parallel_equals_sequential():
    """OpenMP parallel node/edge solves must be bit-identical to sequential —
    each subproblem owns its OSQP workspace, so only wall-clock may change."""
    rng = np.random.default_rng(53)
    dogs, edges = (1, 2, 3), ((1, 2), (1, 3), (2, 3))
    obstacles = [{"pos": (2.0, 0.2), "radius": 0.30}]
    seq = CPP_AC.ADMMCoordinator(dogs=dogs, edges=edges, obstacles=obstacles,
                                 hard_through=1, parallel=False)
    par = CPP_AC.ADMMCoordinator(dogs=dogs, edges=edges, obstacles=obstacles,
                                 hard_through=1, parallel=True)
    N = PY.N
    for cyc in range(3):
        xnow = {d: np.array([rng.uniform(-0.1, 0.1), 1.0 * i, 0.0, 0.0])
                for i, d in enumerate(dogs)}
        xdes = {}
        for i, d in enumerate(dogs):
            xd = np.zeros((N, 4))
            xd[:, 0] = np.linspace(0.2, 4.0, N)
            xd[:, 1] = 1.0 * i
            xdes[d] = xd
        xs, hs = seq.step(xnow, xdes)
        xp, hp = par.step(xnow, xdes)
        for d in dogs:
            assert _bits_equal(xs[d], xp[d]), f"parallel xi differs dog={d} cyc={cyc}"
        for key in ("r_prim", "r_dual", "h2_viol"):
            assert _bits_equal(np.asarray(hs[key]), np.asarray(hp[key])), key
        assert hs["edge_fail"] == hp["edge_fail"]


def test_c4_coordinator_two_dogs_bit_identical():
    _coordinator_chain(dogs=(1, 2), edges=((1, 2),), with_formation=False)


def test_c4_coordinator_three_dogs_formation_bit_identical():
    _coordinator_chain(dogs=(1, 2, 3), edges=((1, 2), (1, 3), (2, 3)),
                       with_formation=True)


# ---------------- C6: reference / A* / fleet config ----------------


def _c6_polylines(rng):
    grid = [[0.0, 0.0]]
    for k in range(1, 12):  # A*-shaped: axis + diagonal 0.15 m grid steps
        p = grid[-1]
        step = (0.15, 0.0) if k % 3 else (0.15, 0.15)
        grid.append([p[0] + step[0], p[1] + step[1]])
    polys = [
        [[0.0, 0.0], [4.0, 0.0]],                           # straight 2-pt
        grid,                                               # grid polyline
        [[0.0, 0.0], [1.0, 0.5], [1.0, 0.5], [2.0, -0.3]],  # duplicate wp (seg<1e-12)
        [[0.2, -0.1], [0.2, -0.1]],                         # fully degenerate, L=0
    ]
    for _ in range(30):  # random jagged polylines
        m = int(rng.integers(2, 9))
        polys.append(np.cumsum(rng.uniform(-0.4, 0.6, (m, 2)), axis=0).tolist())
    return polys


def test_c6_reference_bit_identical():
    rng = np.random.default_rng(61)
    for poly in _c6_polylines(rng):
        wp, cum = PY_REF._cumulative(poly)
        L = cum[-1]
        for s in (-1.0, 0.0, 0.3 * L, L, L + 2.0, *cum.tolist()):
            assert _bits_equal(PY_REF.point_at_arclength(wp, cum, s),
                               CPP_REF.point_at_arclength(wp, cum, s)), \
                f"point_at_arclength s={s}"
        for cur in ([0.0, 0.0], wp[0], wp[-1],
                    rng.uniform(-1.0, 3.0, 2), rng.uniform(-1.0, 3.0, 2)):
            for kw in ({}, {"n": 20, "ts": 0.2},
                       {"v_cruise": 0.15, "d_brake": 0.5},
                       {"d_brake": 0.0}):  # d_brake<=1e-9 branch
                assert _bits_equal(PY_REF.build_reference(cur, poly, **kw),
                                   CPP_REF.build_reference(cur, poly, **kw)), \
                    f"build_reference cur={cur} kw={kw}"


# exact ctor args of the real ADMM-side callers (verify_plum / verify_door / publisher)
_ASTAR_CASES = {
    "plum": dict(resolution=0.15, robot_radius=0.35,
                 obstacles=[{"pos": o["pos"], "radius": o["radius"]}
                            for o in PUB_ARENAS["plum"]["obstacles"]],
                 x_min=0.0, x_max=10.0, y_min=-4.0, y_max=4.0),
    "door": dict(resolution=0.15, robot_radius=0.35,
                 obstacles=[{"pos": o["pos"], "radius": o["radius"]}
                            for o in PUB_ARENAS["door"]["obstacles"]],
                 x_min=0.0, x_max=10.0, y_min=-5.0, y_max=5.0,
                 rect_obstacles=PUB_ARENAS["door"]["rects"]),
}
_ASTAR_START = {1: (2.0, 0.0), 2: (1.0, 1.0), 3: (1.0, -1.0)}


def test_c6_astar_hypot_bit_identical():
    # math.hypot is CPython's scaled compensated sum, NOT libm hypot (~70% of
    # random float pairs differ in the last bit) — the C++ port must mirror it.
    for dx in range(-160, 161):
        for dy in range(-160, 161):
            assert admm_core_cpp.py_hypot(float(dx), float(dy)) == math.hypot(dx, dy)
    rng = np.random.default_rng(67)
    for lo, hi in ((-20.0, 20.0), (-1e6, 1e6), (-1e-6, 1e-6)):
        for _ in range(30000):
            x, y = rng.uniform(lo, hi), rng.uniform(-20.0, 20.0)
            assert admm_core_cpp.py_hypot(x, y) == math.hypot(x, y), (x, y)


def _paths_equal(pp, pc, label):
    assert len(pp) == len(pc), f"{label}: len {len(pp)} vs {len(pc)}"
    if pp:
        assert _bits_equal(np.asarray(pp, float), np.asarray(pc, float)), label


def test_c6_astar_bit_identical():
    rng = np.random.default_rng(71)
    for name, kwargs in _ASTAR_CASES.items():
        py_p, cpp_p = PyAStar(**kwargs), admm_core_cpp.AStarPlanner(**kwargs)
        assert np.array_equal(py_p._omap.astype(np.int32),
                              np.asarray(cpp_p._debug_omap())), f"{name} omap"
        goals = PUB_ARENAS[name]["goals"]
        starts_goals = [( _ASTAR_START[i], goals[i]) for i in (1, 2, 3)]
        for k in range(60):  # fuzz: arbitrary pairs; 1/3 of goals inside obstacles
            s = (rng.uniform(kwargs["x_min"], kwargs["x_max"]),
                 rng.uniform(kwargs["y_min"], kwargs["y_max"]))
            if k % 3 == 0:
                o = kwargs["obstacles"][int(rng.integers(len(kwargs["obstacles"])))]
                g = (o["pos"][0] + rng.uniform(-0.2, 0.2),
                     o["pos"][1] + rng.uniform(-0.2, 0.2))
            else:
                g = (rng.uniform(kwargs["x_min"], kwargs["x_max"]),
                     rng.uniform(kwargs["y_min"], kwargs["y_max"]))
            starts_goals.append((s, g))
        for j, (s, g) in enumerate(starts_goals):
            _paths_equal(py_p.plan(s, g), cpp_p.plan(s, g), f"{name} plan {j}")
            cp, pp = py_p.find_reachable_goal(s, g)
            cc, pc = cpp_p.find_reachable_goal(s, g)
            assert (cp is None) == (cc is None), f"{name} frg cand {j}"
            if cp is not None:
                assert _bits_equal(np.asarray(cp, float), np.asarray(cc, float)), \
                    f"{name} frg cand val {j}"
            _paths_equal(pp, pc, f"{name} frg path {j}")


def test_c6_fleet_data_identical():
    # single source of truth moves to C++ — the py dicts must match EXACTLY
    # (tuple-vs-list shapes included) so verify gates keep working unchanged.
    assert admm_core_cpp.ARENAS == PUB_ARENAS
    assert admm_core_cpp.FORMATIONS == PUB_FORMATIONS
    assert admm_core_cpp.DEFAULT_GOALS == PUB_DEFAULT_GOALS


def test_c6_fleet_math_bit_identical():
    rng = np.random.default_rng(73)
    for _ in range(300):
        yaw = rng.uniform(-8.0, 8.0)
        assert _bits_equal(py_rot2d(yaw), admm_core_cpp.rot2d(yaw)), yaw
        goal_c = rng.uniform(-5.0, 10.0, 2)
        offsets = PUB_FORMATIONS[("V", "column", "V_wide")[int(rng.integers(3))]]
        sp = py_centroid_slot_targets(goal_c, offsets, yaw)
        sc = admm_core_cpp.centroid_slot_targets(goal_c, offsets, yaw)
        assert len(sp) == len(sc)
        for a, b in zip(sp, sc):
            assert _bits_equal(a, b), (goal_c, yaw)
        positions = [rng.uniform(-2.0, 8.0, 2) for _ in range(3)]
        slots = [rng.uniform(-2.0, 8.0, 2) for _ in range(3)]
        assert (py_min_cost_assignment(positions, slots)
                == tuple(admm_core_cpp.min_cost_assignment(positions, slots)))
    # permutation ties: strict < keeps the FIRST (lexicographic) minimum
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    assert (py_min_cost_assignment(pts, pts)
            == tuple(admm_core_cpp.min_cost_assignment(pts, pts)) == (0, 1, 2))
    twin = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([3.0, 3.0])]
    assert (py_min_cost_assignment(pts, twin)
            == tuple(admm_core_cpp.min_cost_assignment(pts, twin)))


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
