// MotionAdapter implementation — mirrors admm/motion_adapter.py operation order.
#include "legged_upper_control/admm_motion_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace admm {

double py_mod(double a, double b) {
    // CPython float_mod: fmod, then correct the sign to follow the divisor.
    double r = std::fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    return r;
}

double wrap_to_pi(double angle) { return py_mod(angle + M_PI, 2.0 * M_PI) - M_PI; }

Eigen::VectorXd pad_ocs2_state(double px, double py, double vx, double vy, double yaw,
                               double com_height,
                               const Eigen::VectorXd& default_joints) {
    Eigen::VectorXd s = Eigen::VectorXd::Zero(OCS2_STATE_DIM);
    s(MOM_LIN_X) = vx;  // normalized linear momentum = COM velocity (/mass)
    s(MOM_LIN_Y) = vy;
    // MOM_LIN_Z, MOM_ANG_X/Y/Z: stay 0 (no angular-momentum reference, ratified)
    s(BASE_PX) = px;
    s(BASE_PY) = py;
    s(BASE_Z) = com_height;  // forced constant (flat ground)
    s(BASE_YAW) = yaw;       // euler-Z; BASE_PITCH, BASE_ROLL stay 0
    for (int j = 0; j < N_JOINTS; ++j) s(JOINT_START + j) = default_joints(j);
    return s;
}

double yaw_step(double vx, double vy, double prev_yaw, double v_freeze,
                double ema_alpha) {
    double raw;
    if (std::hypot(vx, vy) < v_freeze) {
        raw = prev_yaw;  // freeze -> avoid atan2(0,0) jitter
    } else {
        raw = std::atan2(vy, vx);
    }
    const double delta = wrap_to_pi(raw - prev_yaw);  // shortest-arc increment
    return prev_yaw + ema_alpha * delta;              // accumulate continuously
}

Eigen::VectorXd yaw_trajectory(const Eigen::VectorXd& vx_arr,
                               const Eigen::VectorXd& vy_arr, double seed_yaw,
                               double v_freeze, double ema_alpha) {
    const int n = static_cast<int>(vx_arr.size());
    Eigen::VectorXd yaws = Eigen::VectorXd::Zero(n);
    double prev = seed_yaw;
    for (int k = 0; k < n; ++k) {
        prev = yaw_step(vx_arr(k), vy_arr(k), prev, v_freeze, ema_alpha);
        yaws(k) = prev;
    }
    return yaws;
}

Eigen::VectorXd yaw_trajectory_path(const Eigen::VectorXd& px_arr,
                                    const Eigen::VectorXd& py_arr, double seed_yaw,
                                    int lookahead_M, double pos_eps, double ema_alpha,
                                    bool has_max_dyaw, double max_dyaw) {
    const int n = static_cast<int>(px_arr.size());
    Eigen::VectorXd yaws = Eigen::VectorXd::Zero(n);
    double prev = seed_yaw;
    for (int k = 0; k < n; ++k) {
        const int j = std::min(k + lookahead_M, n - 1);  // clamp to horizon end
        const double dx = px_arr(j) - px_arr(k);
        const double dy = py_arr(j) - py_arr(k);
        double raw;
        if (std::hypot(dx, dy) < pos_eps) {
            raw = prev;  // trajectory ~stopped -> hold heading
        } else {
            raw = std::atan2(dy, dx);
        }
        double step = ema_alpha * wrap_to_pi(raw - prev);  // shortest-arc increment
        if (has_max_dyaw)                                  // yaw slew-rate cap per step
            step = std::max(-max_dyaw, std::min(max_dyaw, step));
        prev = prev + step;  // continuous accumulate
        yaws(k) = prev;
    }
    return yaws;
}

int safe_prefix_length(const Eigen::VectorXd& px_i, const Eigen::VectorXd& py_i,
                       const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>&
                           others_pos,
                       double d_safe, int k_max) {
    int K = 0;
    const int kmax = std::min<int>(k_max, static_cast<int>(px_i.size()));
    for (int m = 0; m < kmax; ++m) {
        bool clear = true;
        for (const auto& other : others_pos) {
            if (std::hypot(px_i(m) - other.first(m), py_i(m) - other.second(m)) <
                d_safe) {
                clear = false;
                break;
            }
        }
        if (!clear) break;
        K = m + 1;
    }
    return std::max(1, K);
}

MotionAdapter::MotionAdapter(double com_height, const Eigen::VectorXd& default_joints,
                             int n, double ts, double v_freeze, double ema_alpha,
                             int input_dim, std::string yaw_mode, int lookahead_M,
                             double pos_eps, int k_send, bool has_max_yaw_rate,
                             double max_yaw_rate)
    : N_(n),
      k_send_(k_send),
      com_height_(com_height),
      default_joints_(default_joints),
      ts_(ts),
      v_freeze_(v_freeze),
      ema_alpha_(ema_alpha),
      input_dim_(input_dim),
      yaw_mode_(std::move(yaw_mode)),
      lookahead_M_(lookahead_M),
      pos_eps_(pos_eps) {
    if (default_joints_.size() != N_JOINTS)
        throw std::runtime_error("default_joints must be a 12-vector");
    if (has_max_yaw_rate) {
        has_max_dyaw_ = true;
        max_dyaw_ = max_yaw_rate * ts_;  // rad per horizon step
    }
}

MotionAdapter::Target MotionAdapter::build_target(const Eigen::VectorXd& xi, double t0,
                                                  double seed_yaw,
                                                  const std::string& yaw_mode_override,
                                                  int k_send_override) const {
    const int N = N_;
    int ns = k_send_override < 0 ? k_send_ : k_send_override;
    ns = std::max(1, std::min(ns, N));  // clamp to [1, N]
    Eigen::VectorXd px(N), py(N), vx(N), vy(N);
    for (int k = 1; k <= N; ++k) {
        px(k - 1) = xi(px_index(k, N));
        py(k - 1) = xi(py_index(k, N));
        vx(k - 1) = xi(vx_index(k, N));
        vy(k - 1) = xi(vy_index(k, N));
    }
    const std::string& mode = yaw_mode_override.empty() ? yaw_mode_ : yaw_mode_override;
    Eigen::VectorXd yaws;
    if (mode == "tangent") {
        yaws = yaw_trajectory(vx, vy, seed_yaw, v_freeze_, ema_alpha_);
    } else {  // 'path' (Route B, default)
        yaws = yaw_trajectory_path(px, py, seed_yaw, lookahead_M_, pos_eps_,
                                   ema_alpha_, has_max_dyaw_, max_dyaw_);
    }

    Target t;
    t.times = Eigen::VectorXd::Zero(ns);
    t.states = Eigen::MatrixXd::Zero(ns, OCS2_STATE_DIM);
    for (int j = 0; j < ns; ++j) {
        const int k = j + 1;
        t.times(j) = t0 + k * ts_;
        t.states.row(j) =
            pad_ocs2_state(px(j), py(j), vx(j), vy(j), yaws(j), com_height_,
                           default_joints_);
    }
    t.inputs = Eigen::MatrixXd::Zero(ns, input_dim_);  // zero control
    t.next_seed = yaws(ns - 1);
    return t;
}

MotionAdapter::Target MotionAdapter::adapt(const Eigen::VectorXd& xi, int dog,
                                           double t0) {
    const auto it = seed_.find(dog);
    const double seed = it == seed_.end() ? 0.0 : it->second;
    Target out = build_target(xi, t0, seed);
    seed_[dog] = out.next_seed;
    return out;
}

}  // namespace admm
