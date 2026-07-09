#pragma once
// MotionAdapter — C++ mirror of admm/motion_adapter.py (golden reference).
// ADMM xi -> OCS2 24D centroidal target trajectory; Route-B path-lookahead yaw
// bypass with low-speed freeze, EMA'd WRAPPED increment on a CONTINUOUS
// accumulator, optional slew-rate cap; dynamic safe-prefix truncation helper.
// Implementations in src/admm_core/motion_adapter.cpp.
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "legged_upper_control/admm_constants.hpp"

namespace admm {

// --- OCS2 centroidal state index constants ---
inline constexpr int OCS2_STATE_DIM = 24;
inline constexpr int MOM_LIN_X = 0, MOM_LIN_Y = 1, MOM_LIN_Z = 2;
inline constexpr int MOM_ANG_X = 3, MOM_ANG_Y = 4, MOM_ANG_Z = 5;
inline constexpr int BASE_PX = 6, BASE_PY = 7, BASE_Z = 8;
inline constexpr int BASE_YAW = 9, BASE_PITCH = 10, BASE_ROLL = 11;
inline constexpr int JOINT_START = 12, N_JOINTS = 12;

// CPython float % (fmod + sign-of-divisor correction) — NOT std::fmod.
double py_mod(double a, double b);
// (angle + pi) % (2 pi) - pi, with Python modulo semantics (core/geometry.py)
double wrap_to_pi(double angle);

// 4D ADMM state (+ bypass yaw) -> 24D OCS2 centroidal state.
Eigen::VectorXd pad_ocs2_state(double px, double py, double vx, double vy, double yaw,
                               double com_height, const Eigen::VectorXd& default_joints);

// One yaw-bypass step (tangent mode): freeze at low speed, EMA wrapped increment.
double yaw_step(double vx, double vy, double prev_yaw, double v_freeze,
                double ema_alpha);

// Sweep k over the horizon carrying prev_yaw (tangent mode).
Eigen::VectorXd yaw_trajectory(const Eigen::VectorXd& vx_arr,
                               const Eigen::VectorXd& vy_arr, double seed_yaw,
                               double v_freeze, double ema_alpha);

// Route B yaw: position lookahead over the SAME xi positions; freeze when the
// trajectory has essentially stopped; optional per-step slew cap (has_max_dyaw).
Eigen::VectorXd yaw_trajectory_path(const Eigen::VectorXd& px_arr,
                                    const Eigen::VectorXd& py_arr, double seed_yaw,
                                    int lookahead_M, double pos_eps, double ema_alpha,
                                    bool has_max_dyaw, double max_dyaw);

// Leading horizon steps (1..k_max) of dog i's path >= d_safe from every other dog.
int safe_prefix_length(const Eigen::VectorXd& px_i, const Eigen::VectorXd& py_i,
                       const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>&
                           others_pos,
                       double d_safe, int k_max);

class MotionAdapter {
public:
    struct Target {
        Eigen::VectorXd times;   // (ns,)
        Eigen::MatrixXd states;  // (ns, 24)
        Eigen::MatrixXd inputs;  // (ns, input_dim) zeros
        double next_seed;
    };

    MotionAdapter(double com_height, const Eigen::VectorXd& default_joints, int n = N,
                  double ts = TS, double v_freeze = 0.05, double ema_alpha = 0.3,
                  int input_dim = 24, std::string yaw_mode = "path",
                  int lookahead_M = 5, double pos_eps = 0.02, int k_send = K_SEND,
                  bool has_max_yaw_rate = false, double max_yaw_rate = 0.0);

    // yaw_mode_override: "" = use self.yaw_mode; k_send_override: -1 = self.k_send
    Target build_target(const Eigen::VectorXd& xi, double t0 = 0.0,
                        double seed_yaw = 0.0,
                        const std::string& yaw_mode_override = "",
                        int k_send_override = -1) const;

    // Stateful per-dog wrapper: uses/updates this dog's yaw seed across cycles.
    Target adapt(const Eigen::VectorXd& xi, int dog, double t0 = 0.0);

    int N_;
    int k_send_;

    const std::string& yaw_mode() const { return yaw_mode_; }
    double ts() const { return ts_; }
    double v_freeze() const { return v_freeze_; }
    double ema_alpha() const { return ema_alpha_; }
    int lookahead_M() const { return lookahead_M_; }
    double pos_eps() const { return pos_eps_; }

private:
    double com_height_;
    Eigen::VectorXd default_joints_;
    double ts_, v_freeze_, ema_alpha_;
    int input_dim_;
    std::string yaw_mode_;
    int lookahead_M_;
    double pos_eps_;
    bool has_max_dyaw_ = false;
    double max_dyaw_ = 0.0;
    std::map<int, double> seed_;
};

}  // namespace admm
