// Edge QP implementation — mirrors admm/edge_subproblem.py method by method.
#include "legged_upper_control/admm_edge_qp.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace admm {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

EdgeSubproblem::EdgeSubproblem(int n, double rho, double slack_lambda, int hard_through)
    : N_(n),
      rho_(rho),
      slack_lambda_(slack_lambda),
      hard_through_(std::max(0, hard_through)) {
    n_z_ = xi_dim(N_);
    n_cbf_ = N_ - 1;
    nvar_ = 2 * n_z_ + n_cbf_;
    r_cbf_ = 0;
    r_snn_ = n_cbf_;
    nrow_ = 2 * n_cbf_;

    build_fixed_();
    build_P_();
    // pre-assemble the A sparsity pattern now (spec E) with zero CBF coefficients;
    // set_linearization() writes the real frozen values each control cycle.
    set_linearization(Eigen::MatrixX2d::Zero(n_cbf_, 2), Eigen::VectorXd::Zero(n_cbf_));
}

void EdgeSubproblem::build_P_() {
    // rho on both z blocks, 2*lambda_slk on the slack block (all nonzero -> kept).
    std::vector<Trip> pt;
    for (int j = 0; j < 2 * n_z_; ++j) pt.push_back({j, j, rho_});
    for (int j = 2 * n_z_; j < nvar_; ++j) pt.push_back({j, j, 2.0 * slack_lambda_});
    P_pat_ = build_csc_pattern(nvar_, nvar_, pt);
    std::vector<double> pv(pt.size());
    for (std::size_t t = 0; t < pt.size(); ++t) pv[t] = pt[t].v;
    scatter_values(P_pat_, pv, P_data_);
}

void EdgeSubproblem::build_fixed_() {
    for (int k = 0; k < n_cbf_; ++k) {
        const long long scol = 2 * n_z_ + k;
        trips_.push_back({r_cbf_ + k, scol, -1.0});  // -s_k in CBF row
        trips_.push_back({r_snn_ + k, scol, 1.0});   // +s_k in s>=0 row
    }
    n_fixed_trips_ = trips_.size();

    lo_.assign(nrow_, -kInf);      // CBF rows: -inf <= . <= u_k
    up_base_.assign(nrow_, kInf);  // slack rows: 0 <= s <= +inf
    for (int k = 0; k < n_cbf_; ++k) lo_[r_snn_ + k] = 0.0;
    for (int k = 0; k < std::min(hard_through_, n_cbf_); ++k)
        up_base_[r_snn_ + k] = 0.0;  // s_k = 0 -> hard rows k=0..K-1

    // CBF value slots (4 per row: a^i_k, a^j_k push-pull), then freeze the pattern
    for (int k = 0; k < n_cbf_; ++k) {
        const long long aik = a_index(k, N_);
        const long long ajk = n_z_ + a_index(k, N_);
        const long long r = r_cbf_ + k;
        trips_.push_back({r, aik, 0.0});
        trips_.push_back({r, aik + 1, 0.0});
        trips_.push_back({r, ajk, 0.0});
        trips_.push_back({r, ajk + 1, 0.0});
    }
    A_pat_ = build_csc_pattern(nrow_, nvar_, trips_);
    trip_vals_.resize(trips_.size());
    for (std::size_t t = 0; t < trips_.size(); ++t) trip_vals_[t] = trips_[t].v;
}

void EdgeSubproblem::set_linearization(const Eigen::MatrixX2d& g,
                                       const Eigen::VectorXd& u) {
    if (g.rows() != n_cbf_ || u.size() != n_cbf_)
        throw std::runtime_error("set_linearization: bad g/u shape");
    std::size_t t = n_fixed_trips_;
    for (int k = 0; k < n_cbf_; ++k) {
        trip_vals_[t++] = -g(k, 0);  // col_a^i = -g  (push)
        trip_vals_[t++] = -g(k, 1);
        trip_vals_[t++] = +g(k, 0);  // col_a^j = +g  (pull)
        trip_vals_[t++] = +g(k, 1);
    }
    scatter_values(A_pat_, trip_vals_, A_data_);
    u_ = up_base_;
    for (int k = 0; k < n_cbf_; ++k) u_[r_cbf_ + k] = u(k);
    lin_set_ = true;
    if (prob_.is_setup())  // same pattern -> values-only update
        prob_.update_Alu(A_data_, lo_, u_);
}

std::vector<double> EdgeSubproblem::debug_q(const Eigen::VectorXd& xi_i,
                                            const Eigen::VectorXd& xi_j,
                                            const Eigen::VectorXd& lam_i,
                                            const Eigen::VectorXd& lam_j) const {
    std::vector<double> q(nvar_, 0.0);
    for (int j = 0; j < n_z_; ++j) q[j] = -rho_ * (xi_i(j) + lam_i(j));
    for (int j = 0; j < n_z_; ++j) q[n_z_ + j] = -rho_ * (xi_j(j) + lam_j(j));
    return q;  // slack block q = 0 (phi is pure quadratic)
}

EdgeSubproblem::Out EdgeSubproblem::solve(const Eigen::VectorXd& xi_i,
                                          const Eigen::VectorXd& xi_j,
                                          const Eigen::VectorXd& lam_i,
                                          const Eigen::VectorXd& lam_j) {
    if (!lin_set_) throw std::runtime_error("call set_linearization() before solve()");
    const std::vector<double> q = debug_q(xi_i, xi_j, lam_i, lam_j);
    if (!prob_.is_setup()) {
        prob_.setup(P_pat_, P_data_, q, A_pat_, A_data_, lo_, u_);
    } else {
        prob_.update_q_only(q);
    }
    const OsqpProblem::Result res = prob_.solve();
    Out out;
    if (!res.x.allFinite()) return out;  // ok=false, Python's (None, None, None)
    out.ok = true;
    out.z_i = res.x.segment(0, n_z_);
    out.z_j = res.x.segment(n_z_, n_z_);
    out.s = res.x.segment(2 * n_z_, n_cbf_);
    return out;
}

}  // namespace admm
