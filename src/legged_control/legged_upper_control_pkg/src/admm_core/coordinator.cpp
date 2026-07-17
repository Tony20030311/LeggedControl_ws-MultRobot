// ADMMCoordinator implementation — mirrors admm/admm_coordinator.py step by step.
#include "legged_upper_control/admm_coordinator.hpp"

#include <algorithm>
#include <cmath>

namespace admm {

Eigen::VectorXd consensus_target(const EdgeVecs& z, const EdgeVecs& lam, int i,
                                 const std::vector<int>& neighbors) {
    Eigen::VectorXd total;
    bool first = true;
    for (const int j : neighbors) {
        EdgeKey e(i, j);
        if (z.find(e) == z.end()) e = EdgeKey(j, i);
        const Eigen::VectorXd term = z.at(e).at(i) - lam.at(e).at(i);
        if (first) {
            total = term;
            first = false;
        } else {
            total += term;
        }
    }
    return total;
}

ADMMCoordinator::ADMMCoordinator(int p_iters, double rho, std::vector<int> dogs,
                                 std::vector<EdgeKey> edges,
                                 const LaplacianFormation* formation, double w_form,
                                 std::vector<Obstacle> obstacles,
                                 std::vector<Wall> walls, int hard_through,
                                 bool parallel, double robot_margin)
    : rho_(rho),
      P_(p_iters),
      hard_through_(hard_through),
      parallel_(parallel),
      dogs_(std::move(dogs)),
      edges_(std::move(edges)),
      formation_(formation),
      w_form_(w_form),
      obstacles_(obstacles) {
    for (const int i : dogs_) {
        std::vector<int> nb;
        for (const EdgeKey& e : edges_)
            if (e.first == i || e.second == i) nb.push_back(e.first == i ? e.second : e.first);
        neighbors_[i] = nb;
    }
    for (const int i : dogs_)
        node_[i] = std::make_unique<NodeSubproblem>(
            obstacles, walls, robot_margin, /*q_pos=*/10.0, /*q_v=*/1.0,
            /*r_accel=*/0.5, /*w_pred=*/20.0, /*n=*/N, rho_,
            static_cast<int>(neighbors_[i].size()), w_form_);
    for (const EdgeKey& e : edges_)
        edge_[e] = std::make_unique<EdgeSubproblem>(N, rho_, SLACK_LAMBDA, hard_through_);
    N_ = node_[dogs_[0]]->N_;
    nz_ = xi_dim(N_);
}

std::map<int, Eigen::VectorXd> ADMMCoordinator::uncoupled_(
    const std::map<int, Eigen::VectorXd>& xnow,
    const std::map<int, Eigen::MatrixXd>& xdes) {
    std::map<int, Eigen::VectorXd> xibar;
    for (const int i : dogs_) {
        const NodeSubproblem::Out r =
            node_[i]->solve(xnow.at(i), xdes.at(i), nullptr, nullptr, nullptr);
        xibar[i] = r.xi.head(nz_);
    }
    return xibar;
}

double ADMMCoordinator::h2_viol_(const std::map<int, Eigen::VectorXd>& xi,
                                 const std::map<int, Eigen::VectorXd>& xnow) const {
    double worst = 0.0;
    for (const EdgeKey& e : edges_) {
        const Eigen::VectorXd hb =
            realized_hbar(xi.at(e.first), xi.at(e.second), xnow.at(e.first),
                          xnow.at(e.second), N_);
        worst = std::max(worst, std::max(0.0, -hb.minCoeff()));
    }
    return worst;
}

std::map<int, Eigen::MatrixX2d> ADMMCoordinator::formation_grad_(
    const std::map<int, Eigen::VectorXd>& xibar) const {
    const int N = N_;
    std::map<int, Eigen::MatrixX2d> grad;
    for (const int i : dogs_) grad[i] = Eigen::MatrixX2d::Zero(N, 2);
    for (int k = 1; k <= N; ++k) {
        std::vector<Eigen::Vector2d> pos_k;
        pos_k.reserve(dogs_.size());
        for (const int i : dogs_)
            pos_k.emplace_back(xibar.at(i)(px_index(k, N)),
                               xibar.at(i)(px_index(k, N) + 1));
        const auto fg = formation_->compute(pos_k);
        for (std::size_t idx = 0; idx < dogs_.size(); ++idx) {
            grad[dogs_[idx]](k - 1, 0) = fg.second[idx](0);
            grad[dogs_[idx]](k - 1, 1) = fg.second[idx](1);
        }
    }
    return grad;
}

std::pair<std::map<int, Eigen::VectorXd>, ADMMCoordinator::Hist>
ADMMCoordinator::step(const std::map<int, Eigen::VectorXd>& xnow,
                      const std::map<int, Eigen::MatrixXd>& xdes) {
    const int nz = nz_;

    // 1. operating point (true-body xibar) + init z, lam
    std::map<int, Eigen::VectorXd> xibar;
    EdgeVecs z, lam;
    if (cycle_ == 0) {
        xibar = uncoupled_(xnow, xdes);
        for (const EdgeKey& e : edges_) {
            z[e][e.first] = xibar[e.first];
            z[e][e.second] = xibar[e.second];
            lam[e][e.first] = Eigen::VectorXd::Zero(nz);
            lam[e][e.second] = Eigen::VectorXd::Zero(nz);
        }
    } else {
        for (const int i : dogs_) xibar[i] = shift_xi(prev_xi_.at(i), N_);
        z = prev_z_;      // warm start (deep copies via map value semantics)
        lam = prev_lam_;
    }

    // 2. linearize each edge ONCE (outside the ADMM loop, B8)
    for (const EdgeKey& e : edges_) {
        const EdgeLin frozen = linearize_edge(xibar[e.first], xibar[e.second],
                                              xnow.at(e.first), xnow.at(e.second), N_);
        edge_[e]->set_linearization(frozen.g, frozen.u);
    }

    // 2b. freeze the formation gradient ONCE too (B8); cycle 0 skips (spec detail 3)
    const bool use_fgrad = (formation_ != nullptr && cycle_ > 0);
    std::map<int, Eigen::MatrixX2d> fgrad;
    if (use_fgrad) fgrad = formation_grad_(xibar);

    // 3. ADMM loop (fixed P_ITERS, no wait-for-convergence)
    std::map<int, Eigen::VectorXd> xi = xibar;
    Hist hist;
    EdgeVecs z_prev = z;

    // Pre-fetch stable slot pointers so the OpenMP loops never touch std::map
    // (map lookups/inserts are not thread-safe). Every entry already exists.
    const int nd = static_cast<int>(dogs_.size());
    const int ne = static_cast<int>(edges_.size());
    std::vector<NodeSubproblem*> node_p(nd);
    std::vector<const Eigen::VectorXd*> xnow_p(nd);
    std::vector<const Eigen::MatrixXd*> xdes_p(nd);
    std::vector<Eigen::VectorXd*> xibar_p(nd), xi_p(nd);
    std::vector<const Eigen::MatrixX2d*> fg_p(nd, nullptr);
    for (int idx = 0; idx < nd; ++idx) {
        const int i = dogs_[idx];
        node_p[idx] = node_[i].get();
        xnow_p[idx] = &xnow.at(i);
        xdes_p[idx] = &xdes.at(i);
        xibar_p[idx] = &xibar[i];
        xi_p[idx] = &xi[i];
        if (use_fgrad) fg_p[idx] = &fgrad[i];
    }
    std::vector<EdgeSubproblem*> edge_p(ne);
    std::vector<const Eigen::VectorXd*> exi_a(ne), exi_b(ne), elam_a(ne), elam_b(ne);
    std::vector<Eigen::VectorXd*> ez_a(ne), ez_b(ne);
    for (int idx = 0; idx < ne; ++idx) {
        const EdgeKey& e = edges_[idx];
        edge_p[idx] = edge_[e].get();
        exi_a[idx] = &xi[e.first];
        exi_b[idx] = &xi[e.second];
        elam_a[idx] = &lam[e][e.first];
        elam_b[idx] = &lam[e][e.second];
        ez_a[idx] = &z[e][e.first];
        ez_b[idx] = &z[e][e.second];
    }

    for (int p = 0; p < P_; ++p) {
        // node update (parallel over dogs). ct depends only on z/lam, which are
        // frozen during the node phase -> precompute, then solve concurrently.
        std::vector<Eigen::VectorXd> cts(nd);
        for (int idx = 0; idx < nd; ++idx)
            cts[idx] = consensus_target(z, lam, dogs_[idx], neighbors_[dogs_[idx]]);
        int edge_fail_round = 0;
#pragma omp parallel for schedule(static) if (parallel_)
        for (int idx = 0; idx < nd; ++idx) {
            const NodeSubproblem::Out r = node_p[idx]->solve(
                *xnow_p[idx], *xdes_p[idx], xibar_p[idx], &cts[idx], fg_p[idx]);
            *xi_p[idx] = r.xi.head(nz);
        }
        // A non-finite node solve means that node's QP was infeasible (OSQP -> NaN
        // xi). Stop the loop NOW, before the edge solve and dual update turn that NaN
        // into a NaN q that poisons the OSQP workspaces (and before the poisoned
        // solves spin to max_iter). xi stays non-finite -> the finite-guard at the
        // end of step() resets the workspaces and cold-starts next cycle.
        bool node_nan = false;
        for (int idx = 0; idx < nd; ++idx)
            if (!xi_p[idx]->allFinite()) { node_nan = true; break; }
        if (node_nan) break;
        // edge update (parallel over edges)
#pragma omp parallel for schedule(static) reduction(+ : edge_fail_round) if (parallel_)
        for (int idx = 0; idx < ne; ++idx) {
            const EdgeSubproblem::Out o = edge_p[idx]->solve(
                *exi_a[idx], *exi_b[idx], *elam_a[idx], *elam_b[idx]);
            if (!o.ok) {  // dead edge solve: keep the last z, flag it
                edge_fail_round += 1;
            } else {
                *ez_a[idx] = o.z_i;
                *ez_b[idx] = o.z_j;
            }
        }
        hist.edge_fail += edge_fail_round;
        // dual update (scaled, NO rho factor)
        for (const EdgeKey& e : edges_) {
            lam[e][e.first] = lam[e][e.first] + (xi[e.first] - z[e][e.first]);
            lam[e][e.second] = lam[e][e.second] + (xi[e.second] - z[e][e.second]);
        }
        // residuals (spec C5.4) — np.sum uses pairwise summation, mirrored exactly
        double rp2 = 0.0, rd2 = 0.0, pos_max = 0.0, vel_max = 0.0;
        std::vector<double> sq(nz);
        for (const EdgeKey& e : edges_)
            for (const int i : {e.first, e.second}) {
                const Eigen::VectorXd d1 = xi[i] - z[e][i];
                for (int j = 0; j < nz; ++j) sq[j] = d1(j) * d1(j);
                rp2 += numpy_pairwise_sum(sq.data(), sq.size());
                const Eigen::VectorXd d2 = z[e][i] - z_prev[e][i];
                for (int j = 0; j < nz; ++j) sq[j] = d2(j) * d2(j);
                rd2 += numpy_pairwise_sum(sq.data(), sq.size());
                // per-iter plan change, split into max position (m) vs max velocity (m/s)
                for (int k = 1; k <= N_; ++k) {
                    pos_max = std::max({pos_max, std::abs(d2(px_index(k, N_))),
                                        std::abs(d2(px_index(k, N_) + 1))});
                    vel_max = std::max({vel_max, std::abs(d2(vx_index(k, N_))),
                                        std::abs(d2(vx_index(k, N_) + 1))});
                }
            }
        hist.r_prim.push_back(std::sqrt(rp2));
        hist.r_dual.push_back(rho_ * std::sqrt(rd2));
        hist.r_pos.push_back(pos_max);
        hist.r_vel.push_back(vel_max);
        hist.h2_viol.push_back(h2_viol_(xi, xnow));
        z_prev = z;
    }

    // 4. store for next-cycle warm start -- NEVER store a non-finite iterate. One
    // infeasible node solve returns NaN xi, the dual update writes NaN into lam, and
    // a stored NaN warm start reproduces itself every cycle after (PF2 2026-07-16:
    // a single NaN at t=330 froze the fleet for the remaining 1264 s, 7 legs lost).
    // Dropping the warm start costs one cold-start cycle -- the cycle_==0 path.
    bool finite = true;
    for (const int i : dogs_)
        if (!xi.at(i).allFinite()) { finite = false; break; }
    if (finite) {
        prev_xi_ = xi;
        prev_z_ = z;
        prev_lam_ = lam;
        has_prev_ = true;
        cycle_ += 1;
    } else {
        // Dropping the coordinator warm start is NOT enough on its own: a NaN q
        // (via the NaN dual) poisons each OSQP workspace's internal warm-start
        // iterate, and warm_start=1 keeps that NaN forever -- the cold-start
        // uncoupled_() next cycle solves the SAME poisoned workspaces and returns
        // NaN again (verified: a poisoned edge stays NaN on every later finite
        // solve). Tear every workspace down so the cycle_==0 rebuild is truly clean.
        for (const int i : dogs_) node_[i]->reset_solver();
        for (const EdgeKey& e : edges_) edge_[e]->reset_solver();
        has_prev_ = false;
        cycle_ = 0;
    }
    return {xi, hist};
}

}  // namespace admm
