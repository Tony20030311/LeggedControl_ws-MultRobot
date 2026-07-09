"""Stage 2 ADMM coordinator -- mechanical correctness (deterministic).

These pin the node-edge wiring that is easy to get wrong; the convergence-curve
behaviour (r_prim / r_dual descent) is the closed-loop gate in verify_stage2.py.

  - consensus_target sums the RIGHT per-edge copies (B1): sum_{j in N(i)} uses
    dog i's own copy on edge (i,j), never another dog's copy;
  - the dual update is SCALED with no rho factor: lambda += (xi - z);
  - z is stored per edge: z[(1,2)][1], z[(1,2)][2].
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admm_impl import constants as C  # noqa: E402
from admm_impl import reference as ref  # noqa: E402
from admm_impl import ac  # noqa: E402


def _two_dog_setup():
    xnow = {1: np.array([-1.0, 0.10, 0.0, 0.0]),
            2: np.array([1.0, -0.10, 0.0, 0.0])}
    xdes = {1: ref.build_reference(xnow[1][:2], [xnow[1][:2], [3.0, 0.10]]),
            2: ref.build_reference(xnow[2][:2], [xnow[2][:2], [-3.0, -0.10]])}
    return xnow, xdes


def test_consensus_target_sums_right_copies():
    """B1: dog 1's consensus target = sum over its edges of ITS OWN copy minus its
    own dual. A 3-dog synthetic makes the wrong-copy bug visible (9.0 = other dogs'
    copies must never be summed in)."""
    n = 6
    z = {(1, 2): {1: np.full(n, 2.0), 2: np.full(n, 9.0)},
         (1, 3): {1: np.full(n, 3.0), 3: np.full(n, 9.0)},
         (2, 3): {2: np.full(n, 9.0), 3: np.full(n, 9.0)}}
    lam = {(1, 2): {1: np.full(n, 0.5), 2: np.zeros(n)},
           (1, 3): {1: np.full(n, 1.0), 3: np.zeros(n)},
           (2, 3): {2: np.zeros(n), 3: np.zeros(n)}}
    ct = ac.consensus_target(z, lam, 1, [2, 3])
    # (2.0 - 0.5) + (3.0 - 1.0) = 1.5 + 2.0 = 3.5, from dog-1 copies on BOTH edges
    np.testing.assert_allclose(ct, np.full(n, 3.5))


def test_dual_update_scaled_no_rho():
    """One cycle (cold start, lambda0=0), one ADMM iter -> lambda = xi - z exactly.
    A stray rho factor (lambda = rho*(xi-z)) would blow this up by 20x."""
    coord = ac.ADMMCoordinator(p_iters=1)
    xnow, xdes = _two_dog_setup()
    xi, hist = coord.step(xnow, xdes)
    for i in (1, 2):
        expected = xi[i] - coord.prev_z[(1, 2)][i]
        np.testing.assert_allclose(coord.prev_lam[(1, 2)][i], expected, atol=1e-9)


def test_z_stored_per_edge():
    coord = ac.ADMMCoordinator(p_iters=2)
    xnow, xdes = _two_dog_setup()
    xi, hist = coord.step(xnow, xdes)
    assert set(coord.prev_z.keys()) == {(1, 2)}
    assert set(coord.prev_z[(1, 2)].keys()) == {1, 2}
    assert coord.prev_z[(1, 2)][1].shape == (C.xi_dim(C.N),)
    assert len(hist["r_prim"]) == 2       # one residual record per ADMM iteration


def test_residual_keys_present():
    coord = ac.ADMMCoordinator(p_iters=3)
    xnow, xdes = _two_dog_setup()
    xi, hist = coord.step(xnow, xdes)
    for k in ("r_prim", "r_dual", "h2_viol"):
        assert k in hist and len(hist[k]) == 3
        assert all(np.isfinite(v) for v in hist[k])


def test_three_dog_wiring_generic():
    """Stage 3 (change A): the complete 3-graph must wire with NO two-dog
    assumption -- each node has exactly 2 neighbors, z/lam are stored per edge with
    both endpoints, and one full ADMM cycle runs over all 3 edges. This is why
    change A needs no structural edit to the coordinator."""
    coord = ac.ADMMCoordinator(dogs=(1, 2, 3),
                               edges=((1, 2), (1, 3), (2, 3)), p_iters=2)
    assert coord.neighbors == {1: [2, 3], 2: [1, 3], 3: [1, 2]}
    xnow = {1: np.array([-1.0, 0.6, 0.0, 0.0]),
            2: np.array([-1.0, -0.6, 0.0, 0.0]),
            3: np.array([-1.7, 0.0, 0.0, 0.0])}
    xdes = {i: ref.build_reference(
                xnow[i][:2], [xnow[i][:2], xnow[i][:2] + np.array([3.0, 0.0])])
            for i in (1, 2, 3)}
    xi, hist = coord.step(xnow, xdes)
    assert set(coord.prev_z.keys()) == {(1, 2), (1, 3), (2, 3)}
    for e in ((1, 2), (1, 3), (2, 3)):
        assert set(coord.prev_z[e].keys()) == set(e)
    assert set(xi.keys()) == {1, 2, 3}
    assert len(hist["r_prim"]) == 2
