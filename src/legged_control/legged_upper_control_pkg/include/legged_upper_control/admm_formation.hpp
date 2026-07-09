#pragma once
// LaplacianFormation — C++ mirror of core/formation.py (golden reference).
// Normalized-Laplacian formation similarity metric (ZJU FAST-Lab, ICRA 2022):
// f = ||L_hat - L_hat_des||_F^2 with the analytic dphi/dp_i gradient. rospy logs
// dropped (logging is not behavior). Implementations in src/admm_core/formation.cpp
// mirror the Python operation order for bit-identical doubles.
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace admm {

class LaplacianFormation {
public:
    // formation_configs: {name: [(x,y), ...]} centroid-relative offsets
    explicit LaplacianFormation(
        const std::map<std::string, std::vector<Eigen::Vector2d>>& formation_configs);

    void set_formation(const std::string& name);  // unknown name -> no-op (warn)
    const std::string& current_formation() const { return current_; }
    const std::vector<Eigen::Vector2d>* current_offsets() const;

    // f and per-dog world-frame gradient dphi/dp_i.
    std::pair<double, std::vector<Eigen::Vector2d>> compute(
        const std::vector<Eigen::Vector2d>& positions) const;

    static Eigen::MatrixXd compute_L_hat(const std::vector<Eigen::Vector2d>& pos);

private:
    static void compute_L_hat_with_internals(const std::vector<Eigen::Vector2d>& pos,
                                             Eigen::MatrixXd& L_hat, Eigen::MatrixXd& w,
                                             Eigen::VectorXd& d);
    static double df_dw(int i, int j, const Eigen::MatrixXd& w, const Eigen::VectorXd& d,
                        const Eigen::MatrixXd& df_dL, int n);

    std::map<std::string, Eigen::MatrixXd> L_hat_des_cache_;
    std::map<std::string, std::vector<Eigen::Vector2d>> offset_cache_;
    std::string current_;
    bool has_des_ = false;
    Eigen::MatrixXd L_hat_des_;
};

}  // namespace admm
