// C++ mirror of admm/reference.py — see admm_reference.hpp for the contract.
#include "legged_upper_control/admm_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace admm {

Eigen::VectorXd cumulative_arclength(const Eigen::MatrixX2d& wp) {
    const int m = static_cast<int>(wp.rows());
    Eigen::VectorXd cum(m);
    cum(0) = 0.0;
    for (int i = 0; i + 1 < m; ++i) {
        const double dx = wp(i + 1, 0) - wp(i, 0);
        const double dy = wp(i + 1, 1) - wp(i, 1);
        cum(i + 1) = cum(i) + std::sqrt(dx * dx + dy * dy);  // np.cumsum is sequential
    }
    return cum;
}

Eigen::Vector2d point_at_arclength(const Eigen::MatrixX2d& wp,
                                   const Eigen::VectorXd& cum, double s) {
    // Single-point path: no segment to interpolate -- pin on the point. Without this
    // the index clamp below lands on row -1 (out-of-bounds read, UB) whenever a
    // degenerate 1-waypoint path reaches build_reference (seen via rescue, PF2).
    if (wp.rows() < 2) return Eigen::Vector2d(wp(0, 0), wp(0, 1));
    const double L = cum(cum.size() - 1);
    s = std::min(std::max(s, 0.0), L);
    // np.searchsorted(cum, s, side="right") - 1
    int i = static_cast<int>(
        std::upper_bound(cum.data(), cum.data() + cum.size(), s) - cum.data()) - 1;
    i = std::min(std::max(i, 0), static_cast<int>(wp.rows()) - 2);
    const double seg_len = cum(i + 1) - cum(i);
    const double t = seg_len < 1e-12 ? 0.0 : (s - cum(i)) / seg_len;
    return Eigen::Vector2d(wp(i, 0) + t * (wp(i + 1, 0) - wp(i, 0)),
                           wp(i, 1) + t * (wp(i + 1, 1) - wp(i, 1)));
}

double project_arclength(const Eigen::Vector2d& pos, const Eigen::MatrixX2d& wp,
                         const Eigen::VectorXd& cum) {
    double best_s = 0.0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i + 1 < static_cast<int>(wp.rows()); ++i) {
        const double ab0 = wp(i + 1, 0) - wp(i, 0);
        const double ab1 = wp(i + 1, 1) - wp(i, 1);
        const double L2 = ab0 * ab0 + ab1 * ab1;  // ab @ ab (2-vec, proven order)
        double t = L2 < 1e-12
                       ? 0.0
                       : ((pos(0) - wp(i, 0)) * ab0 + (pos(1) - wp(i, 1)) * ab1) / L2;
        t = std::min(std::max(t, 0.0), 1.0);
        const double proj0 = wp(i, 0) + t * ab0;
        const double proj1 = wp(i, 1) + t * ab1;
        const double d2 = (pos(0) - proj0) * (pos(0) - proj0) +
                          (pos(1) - proj1) * (pos(1) - proj1);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_s = cum(i) + t * std::sqrt(L2);
        }
    }
    return best_s;
}

Eigen::MatrixXd build_reference(const Eigen::Vector2d& cur_pos,
                                const Eigen::MatrixX2d& waypoints,
                                int n, double ts, double v_cruise, double d_brake) {
    const Eigen::VectorXd cum = cumulative_arclength(waypoints);
    const double L = cum(cum.size() - 1);
    double s = project_arclength(cur_pos, waypoints, cum);
    Eigen::MatrixXd x_des = Eigen::MatrixXd::Zero(n, 4);
    for (int k = 0; k < n; ++k) {
        const double d_remain = std::max(0.0, L - s);
        const double v_ref =
            d_brake > 1e-9 ? v_cruise * std::min(1.0, d_remain / d_brake) : v_cruise;
        s = std::min(L, s + v_ref * ts);
        const Eigen::Vector2d p = point_at_arclength(waypoints, cum, s);
        x_des(k, 0) = p(0);
        x_des(k, 1) = p(1);
    }
    return x_des;
}

}  // namespace admm
