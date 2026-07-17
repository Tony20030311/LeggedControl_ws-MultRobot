#pragma once
// Stage 1 node QP — C++ mirror of admm/node_subproblem.py (golden reference).
// Same decision vector, same fixed-sparsity OSQP discipline (setup once, values-
// only updates, warm-started), same cold-start soft warm-up, same hard/soft split.
// Implementations in src/admm_core/node_qp.cpp mirror the Python operation order
// for bit-identical assembly (test_cpp_parity.py gate).
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "legged_upper_control/admm_constants.hpp"
#include "legged_upper_control/admm_qp_common.hpp"

namespace admm {

struct Obstacle {
    Eigen::Vector2d pos;
    double radius;
};

struct Wall {
    Eigen::Vector2d normal;
    Eigen::Vector2d point;
    double d_safe;
};

// h = ||p - p_obs||^2 - r_eff^2
double h_obstacle(const Eigen::Vector2d& p, const Eigen::Vector2d& p_obs, double r_eff);
// h = n_w . (p - p_w) - d_safe
double h_wall(const Eigen::Vector2d& p, const Eigen::Vector2d& n_w,
              const Eigen::Vector2d& p_w, double d_safe);

class NodeSubproblem {
public:
    struct Out {
        std::string status;
        Eigen::VectorXd xi;   // full nvar solution (xi block + slacks), like res.x
        bool finite = false;  // np.all(np.isfinite(xi))
        Eigen::MatrixXd x_pred;   // (N,4) when finite
        Eigen::MatrixX2d a_pred;  // (N,2) when finite
        Eigen::Vector2d a0;
    };

    NodeSubproblem(std::vector<Obstacle> obstacles, std::vector<Wall> walls,
                   double robot_margin = 0.30, double q_pos = 10.0, double q_v = 1.0,
                   double r_accel = 0.5, double w_pred = 20.0, int n = N,
                   double rho_consensus = 0.0, int n_neighbors = 0,
                   double w_form = 0.0);

    // xbar / consensus_target / formation_grad: nullptr == Python None.
    Out solve(const Eigen::VectorXd& x_now, const Eigen::MatrixXd& x_des,
              const Eigen::VectorXd* xbar, const Eigen::VectorXd* consensus_target,
              const Eigen::MatrixX2d* formation_grad);

    int N_;

    // Drop the poisoned OSQP workspace so the next solve() rebuilds it clean.
    void reset_solver() { prob_.reset(); }

    // --- debug accessors for the bit-parity gate (test_cpp_parity.py) ---
    struct PassData {
        std::vector<double> q, Ax, lo, up;
    };
    PassData debug_pass(const Eigen::VectorXd& x_now, const Eigen::MatrixXd& x_des,
                        const Eigen::VectorXd& xbar,
                        const Eigen::VectorXd* consensus_target,
                        const Eigen::MatrixX2d* formation_grad, bool drop_hard);
    const CscPattern& debug_P_pattern() const { return P_pat_; }
    const std::vector<double>& debug_P_data() const { return P_data_; }
    const CscPattern& debug_A_pattern() const { return A_pat_; }
    int nvar() const { return nvar_; }
    int nrow() const { return nrow_; }
    int n_slack() const { return n_slack_; }

private:
    struct Feat {  // uniform feature: obstacles first, then walls
        bool is_obs;
        Eigen::Vector2d a;  // obs: pos      | wall: normal
        Eigen::Vector2d b;  // obs: (unused) | wall: point
        double c;           // obs: r_eff    | wall: d_safe
    };
    struct CbfEntry {
        int feat, k, row, col0;
        bool hard;
    };

    std::vector<Feat> features_() const;
    void build_fixed_();
    void build_cbf_layout_();
    void build_P_();
    std::vector<double> q_(const Eigen::MatrixXd& x_des,
                           const Eigen::VectorXd* consensus_target,
                           const Eigen::MatrixX2d* formation_grad) const;
    void operating_point_(const Eigen::VectorXd& x_now, const Eigen::VectorXd& xbar,
                          Eigen::MatrixX2d& pbar, Eigen::MatrixX2d& abar) const;
    void cbf_row_(const Feat& feat, int k, const Eigen::MatrixX2d& pbar,
                  const Eigen::MatrixX2d& abar, double& c0, double& c1,
                  double& b) const;
    void assemble_A_(const Eigen::MatrixX2d& pbar, const Eigen::MatrixX2d& abar,
                     std::vector<double>& Ax, std::vector<double>& up);
    void lu_(const Eigen::VectorXd& x_now, const std::vector<double>& up_cbf,
             std::vector<double>& lo, std::vector<double>& up) const;
    Out solve_pass_(const Eigen::VectorXd& x_now, const std::vector<double>& q,
                    const Eigen::MatrixX2d& pbar, const Eigen::MatrixX2d& abar,
                    bool drop_hard);
    Out out_from_res_(const OsqpProblem::Result& res) const;

    int n_xi_, n_soft_per_, n_slack_, nvar_, nrow_;
    int r_dyn_, r_vb_, r_ab_, r_hard_, r_soft_, r_snn_;
    std::vector<Obstacle> obstacles_;
    std::vector<Wall> walls_;
    double robot_margin_, q_pos_, q_v_, r_accel_, w_pred_, rho_consensus_, w_form_;
    int n_neighbors_;
    std::vector<Trip> trips_;
    std::vector<double> trip_vals_;
    std::size_t n_fixed_trips_ = 0;
    std::vector<CbfEntry> cbf_;
    CscPattern A_pat_, P_pat_;
    std::vector<double> P_data_;
    std::vector<double> lo_base_, up_base_;
    OsqpProblem prob_;
};

}  // namespace admm
