#pragma once
// C6 reference sampler — C++ mirror of admm/reference.py (golden reference).
//
// Position-only anchors along a waypoint polyline (spec B6/B7: velocity dims 0);
// deceleration lives in the reference via v_ref = v_cruise * min(1, d_remain/d_brake)
// with d_remain the remaining arc length (spec B5).
//
// Implementations in src/admm_core/reference.cpp mirror the Python operation order
// expression-by-expression (test_cpp_parity.py test_c6_reference bit gate).
#include <Eigen/Dense>

#include "legged_upper_control/admm_constants.hpp"

namespace admm {

// seg lengths -> cumulative arc length, cum[0] = 0 (mirror of _cumulative; the
// [:, :2] slice is the caller's job — waypoints arrive as (M,2) already).
Eigen::VectorXd cumulative_arclength(const Eigen::MatrixX2d& wp);

// Point on the polyline at arc length s (clamped to [0, L]).
Eigen::Vector2d point_at_arclength(const Eigen::MatrixX2d& wp,
                                   const Eigen::VectorXd& cum, double s);

// Arc length of the closest point on the polyline to pos (mirror of _project).
double project_arclength(const Eigen::Vector2d& pos, const Eigen::MatrixX2d& wp,
                         const Eigen::VectorXd& cum);

// N position anchors ahead of cur_pos: (n,4) rows [px, py, 0, 0].
Eigen::MatrixXd build_reference(const Eigen::Vector2d& cur_pos,
                                const Eigen::MatrixX2d& waypoints,
                                int n = N, double ts = TS,
                                double v_cruise = 0.30, double d_brake = 0.80);

}  // namespace admm
