#pragma once
// Stage 2 edge QP — C++ mirror of admm/edge_subproblem.py (golden reference).
// Decision w = [ z^{ij,i} (6N) ; z^{ij,j} (6N) ; s (N-1) ]. Fixed P and A pattern;
// set_linearization refreshes the frozen CBF values once per control cycle; each
// ADMM iteration updates only q and warm-starts (never rebuilt).
// Implementations in src/admm_core/edge_qp.cpp.
#include <vector>

#include <Eigen/Dense>

#include "legged_upper_control/admm_constants.hpp"
#include "legged_upper_control/admm_qp_common.hpp"
#include "legged_upper_control/admm_rti.hpp"

namespace admm {

class EdgeSubproblem {
public:
    explicit EdgeSubproblem(int n = N, double rho = RHO,
                            double slack_lambda = SLACK_LAMBDA, int hard_through = 1);

    // frozen = rti linearize_edge output (built from the TRUE-BODY xi).
    void set_linearization(const Eigen::MatrixX2d& g, const Eigen::VectorXd& u);

    struct Out {
        bool ok = false;  // Python returns (None, None, None) on a dead solve
        Eigen::VectorXd z_i, z_j, s;
    };
    // min phi(s) + (rho/2)||z^i-xi^i-lam^i||^2 + (rho/2)||z^j-xi^j-lam^j||^2
    Out solve(const Eigen::VectorXd& xi_i, const Eigen::VectorXd& xi_j,
              const Eigen::VectorXd& lam_i, const Eigen::VectorXd& lam_j);

    int N_;

    // Drop the poisoned OSQP workspace so the next solve() rebuilds it clean.
    void reset_solver() { prob_.reset(); }

    // --- debug accessors for the bit-parity gate ---
    const CscPattern& debug_P_pattern() const { return P_pat_; }
    const std::vector<double>& debug_P_data() const { return P_data_; }
    const CscPattern& debug_A_pattern() const { return A_pat_; }
    const std::vector<double>& debug_A_data() const { return A_data_; }
    const std::vector<double>& debug_lo() const { return lo_; }
    const std::vector<double>& debug_u() const { return u_; }
    std::vector<double> debug_q(const Eigen::VectorXd& xi_i, const Eigen::VectorXd& xi_j,
                                const Eigen::VectorXd& lam_i,
                                const Eigen::VectorXd& lam_j) const;
    int nvar() const { return nvar_; }
    int nrow() const { return nrow_; }

private:
    void build_P_();
    void build_fixed_();

    int n_z_, n_cbf_, nvar_, nrow_;
    int r_cbf_, r_snn_;
    double rho_, slack_lambda_;
    int hard_through_;
    std::vector<Trip> trips_;
    std::vector<double> trip_vals_;
    std::size_t n_fixed_trips_ = 0;
    CscPattern A_pat_, P_pat_;
    std::vector<double> P_data_, A_data_;
    std::vector<double> lo_, up_base_, u_;
    bool lin_set_ = false;
    OsqpProblem prob_;
};

}  // namespace admm
