// Node QP implementation — mirrors admm/node_subproblem.py method by method.
#include "legged_upper_control/admm_node_qp.hpp"

#include <limits>

namespace admm {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

double h_obstacle(const Eigen::Vector2d& p, const Eigen::Vector2d& p_obs, double r_eff) {
    const double e0 = p(0) - p_obs(0), e1 = p(1) - p_obs(1);
    return (e0 * e0 + e1 * e1) - r_eff * r_eff;
}

double h_wall(const Eigen::Vector2d& p, const Eigen::Vector2d& n_w,
              const Eigen::Vector2d& p_w, double d_safe) {
    return (n_w(0) * (p(0) - p_w(0)) + n_w(1) * (p(1) - p_w(1))) - d_safe;
}

NodeSubproblem::NodeSubproblem(std::vector<Obstacle> obstacles, std::vector<Wall> walls,
                               double robot_margin, double q_pos, double q_v,
                               double r_accel, double w_pred, int n,
                               double rho_consensus, int n_neighbors, double w_form)
    : N_(n),
      n_xi_(xi_dim(n)),
      obstacles_(std::move(obstacles)),
      walls_(std::move(walls)),
      robot_margin_(robot_margin),
      q_pos_(q_pos),
      q_v_(q_v),
      r_accel_(r_accel),
      w_pred_(w_pred),
      rho_consensus_(rho_consensus),
      w_form_(w_form),
      n_neighbors_(n_neighbors) {
    const int n_feat = static_cast<int>(obstacles_.size() + walls_.size());
    n_soft_per_ = N_ - 2;
    n_slack_ = n_feat * n_soft_per_;
    nvar_ = n_xi_ + n_slack_;

    r_dyn_ = 0;
    r_vb_ = r_dyn_ + 4 * N_;
    r_ab_ = r_vb_ + 2 * N_;
    r_hard_ = r_ab_ + 2 * N_;
    r_soft_ = r_hard_ + n_feat;
    r_snn_ = r_soft_ + n_slack_;
    nrow_ = r_snn_ + n_slack_;

    build_fixed_();
    build_cbf_layout_();
    build_P_();
}

std::vector<NodeSubproblem::Feat> NodeSubproblem::features_() const {
    std::vector<Feat> f;
    for (const Obstacle& o : obstacles_)
        f.push_back({true, o.pos, Eigen::Vector2d::Zero(), o.radius + robot_margin_});
    for (const Wall& w : walls_) f.push_back({false, w.normal, w.point, w.d_safe});
    return f;
}

void NodeSubproblem::build_fixed_() {
    const int N = N_;
    lo_base_.assign(nrow_, 0.0);
    up_base_.assign(nrow_, 0.0);
    auto put = [&](long long r, long long c, double v) { trips_.push_back({r, c, v}); };

    const Eigen::Matrix4d Ad = make_Ad();
    const Eigen::Matrix<double, 4, 2> Bd = make_Bd();
    // dynamics equality: x_{k+1} = Ad x_k + Bd a_k ; x_0 measured (b_eq set in lu_)
    for (int k = 0; k < N; ++k) {
        const int r0 = r_dyn_ + 4 * k;
        for (int d = 0; d < 4; ++d) put(r0 + d, x_index(k + 1, N) + d, 1.0);
        for (int d = 0; d < 4; ++d)
            for (int c = 0; c < 2; ++c) put(r0 + d, a_index(k, N) + c, -Bd(d, c));
        if (k >= 1)
            for (int d = 0; d < 4; ++d)
                for (int c = 0; c < 4; ++c) put(r0 + d, x_index(k, N) + c, -Ad(d, c));
    }
    // velocity bounds on states k=1..N
    for (int k = 1; k <= N; ++k) {
        const int rvx = r_vb_ + 2 * (k - 1);
        put(rvx, vx_index(k, N), 1.0);
        put(rvx + 1, vy_index(k, N), 1.0);
        lo_base_[rvx] = -MAX_VX;
        up_base_[rvx] = MAX_VX;
        lo_base_[rvx + 1] = -MAX_VY;
        up_base_[rvx + 1] = MAX_VY;
    }
    // accel bounds on a_k, k=0..N-1
    for (int k = 0; k < N; ++k) {
        const int rax = r_ab_ + 2 * k;
        put(rax, ax_index(k, N), 1.0);
        put(rax + 1, ay_index(k, N), 1.0);
        lo_base_[rax] = -MAX_AX;
        up_base_[rax] = MAX_AX;
        lo_base_[rax + 1] = -MAX_AY;
        up_base_[rax + 1] = MAX_AY;
    }
    // slack >= 0 identity rows + slack column -1 in soft CBF rows
    for (int j = 0; j < n_slack_; ++j) {
        put(r_snn_ + j, n_xi_ + j, 1.0);
        lo_base_[r_snn_ + j] = 0.0;
        up_base_[r_snn_ + j] = kInf;
        put(r_soft_ + j, n_xi_ + j, -1.0);
    }
    n_fixed_trips_ = trips_.size();
}

void NodeSubproblem::build_cbf_layout_() {
    const int N = N_;
    const int n_feat = static_cast<int>(obstacles_.size() + walls_.size());
    for (int f = 0; f < n_feat; ++f) {
        cbf_.push_back({f, 0, r_hard_ + f, a_index(0, N), true});
        int kk = 0;
        for (int k = 1; k <= N - 2; ++k, ++kk) {
            const int soft_j = f * n_soft_per_ + kk;
            cbf_.push_back({f, k, r_soft_ + soft_j, a_index(k, N), false});
        }
    }
    // add CBF value slots to the triplet list (2 per entry), then freeze the pattern
    for (const CbfEntry& e : cbf_) {
        trips_.push_back({e.row, e.col0, 0.0});
        trips_.push_back({e.row, e.col0 + 1, 0.0});
    }
    A_pat_ = build_csc_pattern(nrow_, nvar_, trips_);
    trip_vals_.resize(trips_.size());
    for (std::size_t t = 0; t < trips_.size(); ++t) trip_vals_[t] = trips_[t].v;
}

void NodeSubproblem::build_P_() {
    // Diagonal Hessian. scipy sp.diags(...).tocsc() DROPS exact-zero entries;
    // mirror that so the CSC pattern matches osqp-python's bit for bit.
    const int N = N_;
    std::vector<double> diag(nvar_, 0.0);
    for (int k = 1; k < N; ++k) {
        diag[px_index(k, N)] = 2.0 * q_pos_;
        diag[py_index(k, N)] = 2.0 * q_pos_;
    }
    diag[px_index(N, N)] = 2.0 * 10.0 * q_pos_;
    diag[py_index(N, N)] = 2.0 * 10.0 * q_pos_;
    diag[vx_index(N, N)] = 2.0 * q_v_;
    diag[vy_index(N, N)] = 2.0 * q_v_;
    for (int k = 0; k < N; ++k) {
        diag[ax_index(k, N)] = 2.0 * r_accel_;
        diag[ay_index(k, N)] = 2.0 * r_accel_;
    }
    for (int j = 0; j < n_slack_; ++j) diag[n_xi_ + j] = 2.0 * w_pred_;
    if (rho_consensus_ > 0.0 && n_neighbors_ > 0)
        for (int j = 0; j < n_xi_; ++j) diag[j] += rho_consensus_ * n_neighbors_;

    std::vector<Trip> pt;
    for (int j = 0; j < nvar_; ++j)
        if (diag[j] != 0.0) pt.push_back({j, j, diag[j]});
    P_pat_ = build_csc_pattern(nvar_, nvar_, pt);
    std::vector<double> pv(pt.size());
    for (std::size_t t = 0; t < pt.size(); ++t) pv[t] = pt[t].v;
    scatter_values(P_pat_, pv, P_data_);
}

std::vector<double> NodeSubproblem::q_(const Eigen::MatrixXd& x_des,
                                       const Eigen::VectorXd* consensus_target,
                                       const Eigen::MatrixX2d* formation_grad) const {
    const int N = N_;
    std::vector<double> q(nvar_, 0.0);
    for (int k = 1; k < N; ++k) {
        q[px_index(k, N)] = -2.0 * q_pos_ * x_des(k - 1, 0);
        q[py_index(k, N)] = -2.0 * q_pos_ * x_des(k - 1, 1);
    }
    q[px_index(N, N)] = -2.0 * 10.0 * q_pos_ * x_des(N - 1, 0);
    q[py_index(N, N)] = -2.0 * 10.0 * q_pos_ * x_des(N - 1, 1);
    if (consensus_target != nullptr && rho_consensus_ > 0.0)
        for (int j = 0; j < n_xi_; ++j) q[j] += -rho_consensus_ * (*consensus_target)(j);
    if (formation_grad != nullptr && w_form_ > 0.0)
        for (int k = 1; k <= N; ++k) {
            q[px_index(k, N)] += w_form_ * (*formation_grad)(k - 1, 0);
            q[py_index(k, N)] += w_form_ * (*formation_grad)(k - 1, 1);
        }
    return q;
}

void NodeSubproblem::operating_point_(const Eigen::VectorXd& x_now,
                                      const Eigen::VectorXd& xbar,
                                      Eigen::MatrixX2d& pbar,
                                      Eigen::MatrixX2d& abar) const {
    const int N = N_;
    pbar.setZero();
    abar.setZero();
    pbar(0, 0) = x_now(0);
    pbar(0, 1) = x_now(1);
    for (int k = 1; k <= N; ++k) {
        pbar(k, 0) = xbar(px_index(k, N));
        pbar(k, 1) = xbar(px_index(k, N) + 1);
    }
    for (int k = 0; k < N; ++k) {
        abar(k, 0) = xbar(ax_index(k, N));
        abar(k, 1) = xbar(ax_index(k, N) + 1);
    }
}

void NodeSubproblem::cbf_row_(const Feat& feat, int k, const Eigen::MatrixX2d& pbar,
                              const Eigen::MatrixX2d& abar, double& c0, double& c1,
                              double& b) const {
    const Eigen::Vector2d pk(pbar(k, 0), pbar(k, 1));
    const Eigen::Vector2d pk1(pbar(k + 1, 0), pbar(k + 1, 1));
    const Eigen::Vector2d pk2(pbar(k + 2, 0), pbar(k + 2, 1));
    double h0, h1, h2, g0, g1;
    if (feat.is_obs) {
        h0 = h_obstacle(pk, feat.a, feat.c);
        h1 = h_obstacle(pk1, feat.a, feat.c);
        h2 = h_obstacle(pk2, feat.a, feat.c);
        const double e1x = pk1(0) - feat.a(0), e1y = pk1(1) - feat.a(1);
        const double e2x = pk2(0) - feat.a(0), e2y = pk2(1) - feat.a(1);
        g0 = COEF_HK2 * CBF_CONSTR_COEF * e2x + COEF_HK1 * CBF_CONSTR_COEF_P1 * e1x;
        g1 = COEF_HK2 * CBF_CONSTR_COEF * e2y + COEF_HK1 * CBF_CONSTR_COEF_P1 * e1y;
    } else {  // wall (affine, exact)
        h0 = h_wall(pk, feat.a, feat.b, feat.c);
        h1 = h_wall(pk1, feat.a, feat.b, feat.c);
        h2 = h_wall(pk2, feat.a, feat.b, feat.c);
        const double s = COEF_HK2 * A_TO_P2_COEF + COEF_HK1 * A_TO_P1_COEF;
        g0 = s * feat.a(0);
        g1 = s * feat.a(1);
    }
    const double hbar3 = COEF_HK2 * h2 + COEF_HK1 * h1 + COEF_HK * h0;
    c0 = -g0;
    c1 = -g1;
    b = hbar3 - (g0 * abar(k, 0) + g1 * abar(k, 1));
}

void NodeSubproblem::assemble_A_(const Eigen::MatrixX2d& pbar,
                                 const Eigen::MatrixX2d& abar,
                                 std::vector<double>& Ax, std::vector<double>& up) {
    const std::vector<Feat> feats = features_();
    up = up_base_;
    std::size_t t = n_fixed_trips_;
    for (const CbfEntry& e : cbf_) {
        double c0, c1, b;
        cbf_row_(feats[e.feat], e.k, pbar, abar, c0, c1, b);
        trip_vals_[t++] = c0;
        trip_vals_[t++] = c1;
        up[e.row] = b;
    }
    scatter_values(A_pat_, trip_vals_, Ax);
}

void NodeSubproblem::lu_(const Eigen::VectorXd& x_now, const std::vector<double>& up_cbf,
                         std::vector<double>& lo, std::vector<double>& up) const {
    lo = lo_base_;
    up = up_cbf;
    const Eigen::Matrix4d Ad = make_Ad();
    for (int r = 0; r < 4; ++r) {  // beq[0:4] = Ad @ x_now (zeros absorb exactly)
        double acc = 0.0;
        for (int c = 0; c < 4; ++c) acc += Ad(r, c) * x_now(c);
        lo[r_dyn_ + r] = acc;
        up[r_dyn_ + r] = acc;
    }
    for (int r = 4; r < 4 * N_; ++r) {
        lo[r_dyn_ + r] = 0.0;
        up[r_dyn_ + r] = 0.0;
    }
    for (const CbfEntry& e : cbf_) lo[e.row] = -kInf;
}

NodeSubproblem::Out NodeSubproblem::solve(const Eigen::VectorXd& x_now,
                                          const Eigen::MatrixXd& x_des,
                                          const Eigen::VectorXd* xbar,
                                          const Eigen::VectorXd* consensus_target,
                                          const Eigen::MatrixX2d* formation_grad) {
    const int N = N_;
    const std::vector<double> q = q_(x_des, consensus_target, formation_grad);

    Eigen::MatrixX2d pbar(N + 1, 2), abar(N, 2);
    if (xbar == nullptr) {  // cold start: soft warm-up
        pbar.setZero();
        abar.setZero();
        pbar(0, 0) = x_now(0);
        pbar(0, 1) = x_now(1);
        for (int k = 1; k <= N; ++k) {  // straight reference seed
            pbar(k, 0) = x_des(k - 1, 0);
            pbar(k, 1) = x_des(k - 1, 1);
        }
        const Out soft = solve_pass_(x_now, q, pbar, abar, /*drop_hard=*/true);
        if (soft.finite) {  // linearize the hard pass at the curved soft solution
            operating_point_(x_now, soft.xi, pbar, abar);
        }
        // else: hard pass runs at the reference seed (reported honestly)
    } else {
        operating_point_(x_now, *xbar, pbar, abar);
    }
    return solve_pass_(x_now, q, pbar, abar, /*drop_hard=*/false);
}

NodeSubproblem::PassData NodeSubproblem::debug_pass(
    const Eigen::VectorXd& x_now, const Eigen::MatrixXd& x_des,
    const Eigen::VectorXd& xbar, const Eigen::VectorXd* consensus_target,
    const Eigen::MatrixX2d* formation_grad, bool drop_hard) {
    PassData d;
    d.q = q_(x_des, consensus_target, formation_grad);
    Eigen::MatrixX2d pbar(N_ + 1, 2), abar(N_, 2);
    operating_point_(x_now, xbar, pbar, abar);
    std::vector<double> up_cbf;
    assemble_A_(pbar, abar, d.Ax, up_cbf);
    lu_(x_now, up_cbf, d.lo, d.up);
    if (drop_hard)
        for (const CbfEntry& e : cbf_)
            if (e.hard) d.up[e.row] = kInf;
    return d;
}

NodeSubproblem::Out NodeSubproblem::solve_pass_(const Eigen::VectorXd& x_now,
                                                const std::vector<double>& q,
                                                const Eigen::MatrixX2d& pbar,
                                                const Eigen::MatrixX2d& abar,
                                                bool drop_hard) {
    std::vector<double> Ax, up_cbf, lo, up;
    assemble_A_(pbar, abar, Ax, up_cbf);
    lu_(x_now, up_cbf, lo, up);
    if (drop_hard)
        for (const CbfEntry& e : cbf_)
            if (e.hard) up[e.row] = kInf;
    if (!prob_.is_setup()) {
        prob_.setup(P_pat_, P_data_, q, A_pat_, Ax, lo, up);
    } else {
        prob_.update(&q, lo, up, &Ax);
    }
    return out_from_res_(prob_.solve());
}

NodeSubproblem::Out NodeSubproblem::out_from_res_(const OsqpProblem::Result& res) const {
    const int N = N_;
    Out out;
    out.status = res.status;
    out.xi = res.x;
    out.finite = out.xi.allFinite();
    if (!out.finite) return out;
    out.x_pred.resize(N, 4);
    out.a_pred.resize(N, 2);
    for (int k = 1; k <= N; ++k)
        for (int d = 0; d < 4; ++d) out.x_pred(k - 1, d) = out.xi(x_index(k, N) + d);
    for (int k = 0; k < N; ++k)
        for (int d = 0; d < 2; ++d) out.a_pred(k, d) = out.xi(a_index(k, N) + d);
    out.a0 = Eigen::Vector2d(out.a_pred(0, 0), out.a_pred(0, 1));
    return out;
}

}  // namespace admm
