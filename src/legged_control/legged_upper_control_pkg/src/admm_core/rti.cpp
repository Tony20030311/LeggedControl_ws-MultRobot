// RTI linearizer implementation — mirrors admm/rti_linearizer.py operation order.
#include "legged_upper_control/admm_rti.hpp"

#include <algorithm>

namespace admm {

double edge_h(const Eigen::Vector2d& e) {
    return (e(0) * e(0) + e(1) * e(1)) - D_MIN * D_MIN;
}

double three_point(double h0, double h1, double h2) {
    return COEF_HK2 * h2 + COEF_HK1 * h1 + COEF_HK * h0;
}

Eigen::MatrixX2d positions(const Eigen::VectorXd& xibar, const Eigen::VectorXd& xnow,
                           int n) {
    Eigen::MatrixX2d p = Eigen::MatrixX2d::Zero(n + 1, 2);
    p(0, 0) = xnow(0);
    p(0, 1) = xnow(1);
    for (int m = 1; m <= n; ++m) {
        p(m, 0) = xibar(px_index(m, n));
        p(m, 1) = xibar(px_index(m, n) + 1);
    }
    return p;
}

Eigen::VectorXd shift_xi(const Eigen::VectorXd& xi, int n) {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(xi_dim(n));
    for (int k = 1; k <= n; ++k) {
        const int src = std::min(k + 1, n);
        out.segment<4>(x_index(k, n)) = xi.segment<4>(x_index(src, n));
    }
    for (int k = 0; k < n; ++k) {
        const int src = std::min(k + 1, n - 1);
        out.segment<2>(a_index(k, n)) = xi.segment<2>(a_index(src, n));
    }
    return out;
}

Eigen::VectorXd realized_hbar(const Eigen::VectorXd& xi_i, const Eigen::VectorXd& xi_j,
                              const Eigen::VectorXd& xnow_i,
                              const Eigen::VectorXd& xnow_j, int n) {
    const Eigen::MatrixX2d pi = positions(xi_i, xnow_i, n);
    const Eigen::MatrixX2d pj = positions(xi_j, xnow_j, n);
    Eigen::VectorXd h(n + 1);
    for (int m = 0; m <= n; ++m)
        h(m) = edge_h(Eigen::Vector2d(pi(m, 0) - pj(m, 0), pi(m, 1) - pj(m, 1)));
    Eigen::VectorXd out(n - 1);
    for (int k = 0; k < n - 1; ++k) out(k) = three_point(h(k), h(k + 1), h(k + 2));
    return out;
}

EdgeLin linearize_edge(const Eigen::VectorXd& xibar_i, const Eigen::VectorXd& xibar_j,
                       const Eigen::VectorXd& xnow_i, const Eigen::VectorXd& xnow_j,
                       int n) {
    const Eigen::MatrixX2d pi = positions(xibar_i, xnow_i, n);
    const Eigen::MatrixX2d pj = positions(xibar_j, xnow_j, n);

    EdgeLin r;
    r.n = n;
    r.e = pi - pj;  // (n+1, 2)
    Eigen::VectorXd h(n + 1);
    for (int m = 0; m <= n; ++m) h(m) = edge_h(Eigen::Vector2d(r.e(m, 0), r.e(m, 1)));

    const int nk = n - 1;  // k = 0..n-2
    r.g = Eigen::MatrixX2d::Zero(nk, 2);
    r.u = Eigen::VectorXd::Zero(nk);
    r.Hbar = Eigen::VectorXd::Zero(nk);
    r.abar_i = Eigen::MatrixX2d::Zero(nk, 2);
    r.abar_j = Eigen::MatrixX2d::Zero(nk, 2);
    for (int k = 0; k < nk; ++k) {
        r.Hbar(k) = three_point(h(k), h(k + 1), h(k + 2));
        // g_k = COEF_HK2*CBF_CONSTR_COEF*e_{k+2} + COEF_HK1*CBF_CONSTR_COEF_P1*e_{k+1}
        r.g(k, 0) = COEF_HK2 * CBF_CONSTR_COEF * r.e(k + 2, 0)
                    + COEF_HK1 * CBF_CONSTR_COEF_P1 * r.e(k + 1, 0);
        r.g(k, 1) = COEF_HK2 * CBF_CONSTR_COEF * r.e(k + 2, 1)
                    + COEF_HK1 * CBF_CONSTR_COEF_P1 * r.e(k + 1, 1);
        r.abar_i(k, 0) = xibar_i(ax_index(k, n));
        r.abar_i(k, 1) = xibar_i(ax_index(k, n) + 1);
        r.abar_j(k, 0) = xibar_j(ax_index(k, n));
        r.abar_j(k, 1) = xibar_j(ax_index(k, n) + 1);
        // u_k = Hbar_k - g_k . (abar_i_k - abar_j_k)
        r.u(k) = r.Hbar(k) - (r.g(k, 0) * (r.abar_i(k, 0) - r.abar_j(k, 0))
                              + r.g(k, 1) * (r.abar_i(k, 1) - r.abar_j(k, 1)));
    }
    return r;
}

}  // namespace admm
