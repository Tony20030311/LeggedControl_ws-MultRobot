#pragma once
// Stage 2 RTI linearizer — C++ mirror of admm/rti_linearizer.py (golden reference).
//
// OPERATING-POINT CONTRACT (same as the Python module): the linearization operating
// point is the TRUE-BODY xi ONLY — never the copies z. Full linearization: a_k enters
// the barrier through BOTH h_{k+2} (3/2 Ts^2 path) and h_{k+1} (1/2 Ts^2 path).
//
// Implementations in src/admm_core/rti.cpp mirror the Python operation order
// expression-by-expression (test_cpp_parity.py randomized bit gate).
#include <Eigen/Dense>

#include "legged_upper_control/admm_constants.hpp"

namespace admm {

// h = ||e||^2 - D_MIN^2, e = p^i - p^j (2-vector)
double edge_h(const Eigen::Vector2d& e);

// Xiong discrete second-order HOCBF three-point value; >= 0 is the constraint.
double three_point(double h0, double h1, double h2);

// p_m for m=0..n: p_0 = xnow[:2] (measured), p_m = xibar state position (m=1..n).
Eigen::MatrixX2d positions(const Eigen::VectorXd& xibar, const Eigen::VectorXd& xnow,
                           int n);

// Operating point for the next control cycle: previous true-body xi shifted one step.
Eigen::VectorXd shift_xi(const Eigen::VectorXd& xi, int n = N);

// Realized inter-agent three-point HOCBF value per k=0..N-2 at TRUE-BODY positions.
Eigen::VectorXd realized_hbar(const Eigen::VectorXd& xi_i, const Eigen::VectorXd& xi_j,
                              const Eigen::VectorXd& xnow_i,
                              const Eigen::VectorXd& xnow_j, int n = N);

struct EdgeLin {
    Eigen::MatrixX2d g;       // (n-1, 2) d(three-point)/d a^i_k
    Eigen::VectorXd u;        // (n-1,)   edge-row bound
    Eigen::VectorXd Hbar;     // (n-1,)   three-point at the operating point
    Eigen::MatrixX2d abar_i;  // (n-1, 2) frozen accels a^i_k
    Eigen::MatrixX2d abar_j;  // (n-1, 2) frozen accels a^j_k
    Eigen::MatrixX2d e;       // (n+1, 2) relative positions (diagnostics/tests)
    int n = N;
};

// Freeze the edge inter-agent CBF for accel-steps k=0..n-2 (full linearization).
EdgeLin linearize_edge(const Eigen::VectorXd& xibar_i, const Eigen::VectorXd& xibar_j,
                       const Eigen::VectorXd& xnow_i, const Eigen::VectorXd& xnow_j,
                       int n = N);

}  // namespace admm
