// LaplacianFormation implementation — mirrors core/formation.py operation order.
#include "legged_upper_control/admm_formation.hpp"

#include <cmath>

#include "legged_upper_control/admm_qp_common.hpp"

namespace admm {

LaplacianFormation::LaplacianFormation(
    const std::map<std::string, std::vector<Eigen::Vector2d>>& formation_configs) {
    for (const auto& kv : formation_configs) {
        L_hat_des_cache_[kv.first] = compute_L_hat(kv.second);
        offset_cache_[kv.first] = kv.second;
    }
}

void LaplacianFormation::set_formation(const std::string& name) {
    const auto it = L_hat_des_cache_.find(name);
    if (it == L_hat_des_cache_.end()) return;  // Python: logwarn + return
    if (name != current_) {
        L_hat_des_ = it->second;
        has_des_ = true;
        current_ = name;
    }
}

const std::vector<Eigen::Vector2d>* LaplacianFormation::current_offsets() const {
    const auto it = offset_cache_.find(current_);
    return it == offset_cache_.end() ? nullptr : &it->second;
}

void LaplacianFormation::compute_L_hat_with_internals(
    const std::vector<Eigen::Vector2d>& pos, Eigen::MatrixXd& L_hat,
    Eigen::MatrixXd& w, Eigen::VectorXd& d) {
    const int n = static_cast<int>(pos.size());
    w = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos[i](0) - pos[j](0);
            const double dy = pos[i](1) - pos[j](1);
            const double d2 = dx * dx + dy * dy;  // np.sum((pi-pj)**2), 2 elems
            w(i, j) = d2;
            w(j, i) = d2;
        }
    d = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {  // np.sum(w, axis=1): sequential over j
        double acc = 0.0;
        for (int j = 0; j < n; ++j) acc += w(i, j);
        d(i) = acc;
    }
    L_hat = Eigen::MatrixXd::Identity(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (d(i) > 1e-12 && d(j) > 1e-12) {
                const double val = -w(i, j) / std::sqrt(d(i) * d(j));
                L_hat(i, j) = val;
                L_hat(j, i) = val;
            }
}

Eigen::MatrixXd LaplacianFormation::compute_L_hat(const std::vector<Eigen::Vector2d>& pos) {
    Eigen::MatrixXd L_hat, w;
    Eigen::VectorXd d;
    compute_L_hat_with_internals(pos, L_hat, w, d);
    return L_hat;
}

double LaplacianFormation::df_dw(int i, int j, const Eigen::MatrixXd& w,
                                 const Eigen::VectorXd& d, const Eigen::MatrixXd& df_dL,
                                 int n) {
    const double d_i = d(i), d_j = d(j);
    if (d_i < 1e-12 || d_j < 1e-12) return 0.0;

    double result = 0.0;

    // 1. direct effect of w_ij on L_hat_ij
    const double sqrt_didj = std::sqrt(d_i * d_j);
    const double direct = -1.0 / sqrt_didj + w(i, j) / (2.0 * d_i * sqrt_didj)
                          + w(i, j) / (2.0 * d_j * sqrt_didj);
    result += (df_dL(i, j) + df_dL(j, i)) * direct;

    // 2. effect on L_hat_ik (k != j) through d_i
    for (int k = 0; k < n; ++k) {
        if (k == i || k == j) continue;
        const double d_k = d(k);
        if (d_k < 1e-12) continue;
        const double sqrt_didk = std::sqrt(d_i * d_k);
        const double effect_ik = w(i, k) / (2.0 * d_i * sqrt_didk);
        result += (df_dL(i, k) + df_dL(k, i)) * effect_ik;
    }

    // 3. effect on L_hat_jk (k != i) through d_j
    for (int k = 0; k < n; ++k) {
        if (k == i || k == j) continue;
        const double d_k = d(k);
        if (d_k < 1e-12) continue;
        const double sqrt_djdk = std::sqrt(d_j * d_k);
        const double effect_jk = w(j, k) / (2.0 * d_j * sqrt_djdk);
        result += (df_dL(j, k) + df_dL(k, j)) * effect_jk;
    }

    return result;
}

std::pair<double, std::vector<Eigen::Vector2d>> LaplacianFormation::compute(
    const std::vector<Eigen::Vector2d>& positions) const {
    const int n = static_cast<int>(positions.size());
    if (!has_des_)
        return {0.0, std::vector<Eigen::Vector2d>(n, Eigen::Vector2d::Zero())};

    Eigen::MatrixXd L_hat, w;
    Eigen::VectorXd d;
    compute_L_hat_with_internals(positions, L_hat, w, d);

    const Eigen::MatrixXd diff = L_hat - L_hat_des_;
    // np.sum(diff**2): C-order flatten, numpy pairwise summation
    std::vector<double> sq(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            sq[static_cast<std::size_t>(i) * n + j] = diff(i, j) * diff(i, j);
    const double f = numpy_pairwise_sum(sq.data(), sq.size());

    const Eigen::MatrixXd df_dL = 2.0 * diff;

    std::vector<Eigen::Vector2d> grad_p;
    grad_p.reserve(n);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector2d grad_i = Eigen::Vector2d::Zero();
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const double d_i = d(i), d_j = d(j);
            if (d_i < 1e-12 || d_j < 1e-12) continue;  // degenerate overlap -> 0 grad
            const double df_dw_ij = df_dw(i, j, w, d, df_dL, n);
            // dw_ij/dp_i = 2 (p_i - p_j)
            const double dwx = 2.0 * (positions[i](0) - positions[j](0));
            const double dwy = 2.0 * (positions[i](1) - positions[j](1));
            grad_i(0) += df_dw_ij * dwx;
            grad_i(1) += df_dw_ij * dwy;
        }
        grad_p.push_back(grad_i);
    }
    return {f, grad_p};
}

}  // namespace admm
