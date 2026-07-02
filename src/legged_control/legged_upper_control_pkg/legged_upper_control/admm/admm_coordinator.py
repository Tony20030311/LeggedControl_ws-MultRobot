"""Stage 2 ADMM coordinator -- two dogs, one edge {1,2}, complete graph.

Each control cycle (spec C5.2), node-edge splitting (NOT global consensus):
  1. operating point: cold start -> uncoupled MPC (= z0, paper 4.6); else the
     previous true-body xi shifted one step. Only xi is linearized about (contract
     in rti_linearizer).
  2. linearize the edge inter-agent CBF ONCE, outside the ADMM loop (spec B8).
  3. loop P_ITERS (15-20, no wait-for-convergence):
        node  (parallel): xi^i = argmin J^i + (rho/2) sum_{j in N(i)} ||xi^i - z^{ij,i} + lam^{ij,i}||^2
        edge  (parallel): (z^{ij,i}, z^{ij,j}, s) = edge QP (proximal + inter-agent CBF)
        dual  (scaled, no rho): lam^{ij,i} += xi^i - z^{ij,i}
        measure r_prim, r_dual, h2_viol
  4. store (xi, z, lam) for next-cycle warm start; return xi + residual history.

z is stored PER EDGE (spec B1): z[(1,2)][1], z[(1,2)][2]. The node's consensus sum
Sigma_{j in N(i)} reads dog i's OWN copy on each incident edge (consensus_target).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C            # noqa: E402
import node_subproblem as nq     # noqa: E402
import edge_subproblem as es     # noqa: E402
import rti_linearizer as rti     # noqa: E402


def consensus_target(z, lam, i, neighbors):
    """Node update (19) neighbor sum: sum_{j in N(i)} (z^{ij,i} - lam^{ij,i}).
    z/lam are stored per edge (B1); a dog on multiple edges has an independent copy
    per edge, so this reads dog i's OWN copy on each incident edge (i,j)."""
    total = None
    for j in neighbors:
        e = (i, j) if (i, j) in z else (j, i)
        term = z[e][i] - lam[e][i]
        total = term.copy() if total is None else total + term
    return total


class ADMMCoordinator:
    def __init__(self, p_iters=None, rho=None, dogs=(1, 2), edges=((1, 2),),
                 formation=None, w_form=0.0, obstacles=None, walls=None):
        self.rho = C.RHO if rho is None else float(rho)
        self.P = C.P_ITERS if p_iters is None else int(p_iters)
        self.dogs = list(dogs)
        self.edges = [tuple(e) for e in edges]
        # DI: LaplacianFormation (or None) injected by the verify script so the
        # coordinator stays rospy-free; w_form scales the node formation term.
        self.formation = formation
        self.w_form = float(w_form)
        # node-local obstacle/wall CBF: re-enabled in stage 3 (empty in stage 2).
        self.obstacles = list(obstacles or [])
        self.walls = list(walls or [])
        self.neighbors = {i: [b if a == i else a for (a, b) in self.edges if i in (a, b)]
                          for i in self.dogs}
        # node: obstacle/wall CBF (dormant in stage 2 = empty; fed in stage 3),
        # consensus on, formation weight w_form (0 -> off).
        self.node = {i: nq.NodeSubproblem(obstacles=self.obstacles, walls=self.walls,
                                          rho_consensus=self.rho,
                                          n_neighbors=len(self.neighbors[i]),
                                          w_form=self.w_form)
                     for i in self.dogs}
        self.edge = {e: es.EdgeSubproblem(rho=self.rho) for e in self.edges}
        self.N = self.node[self.dogs[0]].N
        self.nz = C.xi_dim(self.N)
        self.cycle = 0
        self.prev_xi = None
        self.prev_z = None
        self.prev_lam = None

    def _uncoupled(self, xnow, xdes):
        """cycle-0 z0: each dog solves its uncoupled MPC (no consensus, no inter-
        agent) -- exactly the stage-1 node QP. z0 = this, lam0 = 0 (paper 4.6)."""
        xibar = {}
        for i in self.dogs:
            r = self.node[i].solve(xnow[i], xdes[i])       # xbar=None, no consensus
            xibar[i] = np.asarray(r["xi"], float)[:self.nz].copy()
        return xibar

    def _h2_viol(self, xi, xnow):
        """max_k (-realized three-point Hbar)^+ over all edges (spec C5.4/B9).
        DIAGNOSTIC ONLY -- NOT the safety truth. The three-point coefficients sum to
        1-1.4+0.49 = 0.09, so when h is roughly constant Hbar ~= 0.09*h; h2_viol thus
        UNDERSTATES the realized barrier breach ~11x. The safety gate uses the raw
        realized barrier ||p^i-p^j||^2 - D_MIN^2, never h2_viol."""
        worst = 0.0
        for (a, b) in self.edges:
            hb = rti.realized_hbar(xi[a], xi[b], xnow[a], xnow[b], self.N)
            worst = max(worst, float(max(0.0, -hb.min())))
        return worst

    def _formation_grad(self, xibar):
        """Freeze dphi/dp_i for every dog and horizon step (B8: once per cycle,
        OUTSIDE the ADMM loop). Operating point = the true-body xibar; step k
        gathers all dogs' predicted position at k and asks the injected
        core.formation for the per-dog gradient. Returns {i: (N,2)} RAW (no
        w_form here -- the node's _q folds w_form in). Formation is all-to-all:
        every dog's grad depends on all others' positions at step k, which is why
        it is a frozen node-local term, never an edge (spec B4)."""
        N = self.N
        grad = {i: np.zeros((N, 2)) for i in self.dogs}
        for k in range(1, N + 1):
            pos_k = [xibar[i][C.px_index(k, N):C.px_index(k, N) + 2]
                     for i in self.dogs]
            _f, glist = self.formation.compute(pos_k)
            for idx, i in enumerate(self.dogs):
                grad[i][k - 1] = np.asarray(glist[idx], float)
        return grad

    def step(self, xnow, xdes):
        """One control cycle. xnow[i]=state4, xdes[i]=(N,4). Returns (xi, hist)."""
        nz = self.nz

        # 1. operating point (true-body xibar) + init z, lam
        if self.cycle == 0:
            xibar = self._uncoupled(xnow, xdes)
            z = {e: {e[0]: xibar[e[0]].copy(), e[1]: xibar[e[1]].copy()} for e in self.edges}
            lam = {e: {e[0]: np.zeros(nz), e[1]: np.zeros(nz)} for e in self.edges}
        else:
            xibar = {i: rti.shift_xi(self.prev_xi[i], self.N) for i in self.dogs}
            z = {e: {k: self.prev_z[e][k].copy() for k in e} for e in self.edges}       # warm start
            lam = {e: {k: self.prev_lam[e][k].copy() for k in e} for e in self.edges}

        # 2. linearize each edge ONCE (outside the ADMM loop, B8)
        for (a, b) in self.edges:
            frozen = rti.linearize_edge(xibar[a], xibar[b], xnow[a], xnow[b], self.N)
            self.edge[(a, b)].set_linearization(frozen)

        # 2b. freeze the formation gradient ONCE too (B8). cycle 0's xibar is the
        # uncoupled z0 (no formation info) -> skip, formation off (spec detail 3).
        fgrad = (self._formation_grad(xibar)
                 if (self.formation is not None and self.cycle > 0) else None)

        # 3. ADMM loop (fixed P_ITERS, no wait-for-convergence)
        xi = {i: xibar[i].copy() for i in self.dogs}
        hist = {"r_prim": [], "r_dual": [], "h2_viol": []}
        z_prev = {e: {k: z[e][k].copy() for k in e} for e in self.edges}
        for _p in range(self.P):
            # node update (parallel over dogs)
            for i in self.dogs:
                ct = consensus_target(z, lam, i, self.neighbors[i])
                fg = fgrad[i] if fgrad is not None else None
                r = self.node[i].solve(xnow[i], xdes[i], xbar=xibar[i],
                                       consensus_target=ct, formation_grad=fg)
                xi[i] = np.asarray(r["xi"], float)[:nz].copy()
            # edge update (parallel over edges)
            for (a, b) in self.edges:
                zi, zj, _s = self.edge[(a, b)].solve(xi[a], xi[b],
                                                     lam[(a, b)][a], lam[(a, b)][b])
                z[(a, b)][a] = zi
                z[(a, b)][b] = zj
            # dual update (scaled, NO rho factor)
            for (a, b) in self.edges:
                lam[(a, b)][a] = lam[(a, b)][a] + (xi[a] - z[(a, b)][a])
                lam[(a, b)][b] = lam[(a, b)][b] + (xi[b] - z[(a, b)][b])
            # residuals (spec C5.4)
            rp2 = sum(float(np.sum((xi[i] - z[e][i]) ** 2)) for e in self.edges for i in e)
            rd2 = sum(float(np.sum((z[e][i] - z_prev[e][i]) ** 2)) for e in self.edges for i in e)
            hist["r_prim"].append(np.sqrt(rp2))
            hist["r_dual"].append(self.rho * np.sqrt(rd2))
            hist["h2_viol"].append(self._h2_viol(xi, xnow))
            z_prev = {e: {k: z[e][k].copy() for k in e} for e in self.edges}

        # 4. store for next-cycle warm start
        self.prev_xi = xi
        self.prev_z = z
        self.prev_lam = lam
        self.cycle += 1
        return xi, hist
