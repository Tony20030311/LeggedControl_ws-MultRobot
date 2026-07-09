#pragma once
// Stage 0 constants — C++ mirror of legged_upper_control/admm/constants.py.
// The Python module is the golden reference; keep formulas in lockstep, expression
// by expression (operation order preserved for bit-identical doubles).
// test_constants.py (parametrized over both implementations) is the gate.
//
// Constexpr values + index one-liners live here; matrix builders in constants.cpp.
#include <Eigen/Dense>

// OSQP's headers #define RHO (their default rho); it would shadow admm::RHO.
#ifdef RHO
#undef RHO
#endif

namespace admm {

// --- locked parameters (spec_updates_v2 detail 4) ---
inline constexpr int N = 20;
inline constexpr double TS = 0.10;
inline constexpr double GAMMA1 = 0.30;
inline constexpr double GAMMA2 = 0.30;

inline constexpr int N_X = 4;
inline constexpr int N_U = 2;
inline constexpr int XI_DIM = 6 * N;

// --- box bounds (文件二 C4.4) ---
inline constexpr double MAX_VX = 0.55;
inline constexpr double MAX_VY = 0.35;
inline constexpr double MAX_AX = 1.0;
inline constexpr double MAX_AY = 1.0;

// --- discretized double integrator ---
inline constexpr double BD_POS_COEF = 0.5 * (TS * TS);  // mirrors 0.5 * TS**2
inline constexpr double BD_VEL_COEF = TS;

Eigen::Matrix4d make_Ad();
Eigen::Matrix<double, 4, 2> make_Bd();

// --- second-order (Xiong) discrete HOCBF coefficients (文件二 C3) ---
// g1 g2 is POSITIVE (from (1-g1)(1-g2)); a negative sign gives 0.31, off by 37%.
inline constexpr double COEF_HK2 = 1.0;
inline constexpr double COEF_HK1 = -(2.0 - GAMMA1 - GAMMA2);
inline constexpr double COEF_HK = 1.0 - GAMMA1 - GAMMA2 + GAMMA1 * GAMMA2;

Eigen::Vector3d hocbf_coefs();  // ordered [h_k, h_{k+1}, h_{k+2}]

// --- exact a_k -> p_{k+2} / p_{k+1} and CBF-constraint scalars ---
inline constexpr double A_TO_P2_COEF = 1.5 * (TS * TS);        // mirrors 1.5 * TS**2
inline constexpr double CBF_CONSTR_COEF = 3.0 * (TS * TS);     // mirrors 3.0 * TS**2
inline constexpr double A_TO_P1_COEF = 0.5 * (TS * TS);        // mirrors 0.5 * TS**2
inline constexpr double CBF_CONSTR_COEF_P1 = 2.0 * A_TO_P1_COEF;

// --- stage 2: ADMM / edge-QP constants (tuning parameters, spec F) ---
inline constexpr double RHO = 20.0;
inline constexpr double SLACK_LAMBDA = 5.0;
inline constexpr double D_MIN = 0.6;
inline constexpr int P_ITERS = 20;
inline constexpr int K_SEND = 10;

// --- decision-vector index formulas (0-based; ALWAYS 4*n-based, never hardcoded) ---
// xi = [ x_1 .. x_N (4N) , a_0 .. a_{N-1} (2N) ]; states k=1..n, accels k=0..n-1.
inline int xi_dim(int n = N) { return 6 * n; }
inline int x_index(int k, int /*n*/ = N) { return 4 * (k - 1); }
inline int px_index(int k, int /*n*/ = N) { return 4 * (k - 1); }
inline int py_index(int k, int /*n*/ = N) { return 4 * (k - 1) + 1; }
inline int vx_index(int k, int /*n*/ = N) { return 4 * (k - 1) + 2; }
inline int vy_index(int k, int /*n*/ = N) { return 4 * (k - 1) + 3; }
inline int a_index(int k, int n = N) { return 4 * n + 2 * k; }
inline int ax_index(int k, int n = N) { return 4 * n + 2 * k; }
inline int ay_index(int k, int n = N) { return 4 * n + 2 * k + 1; }

}  // namespace admm
