#pragma once
// ADMMCoordinator — C++ mirror of admm/admm_coordinator.py (golden reference).
// Node-edge splitting per control cycle: operating point (cold start = uncoupled
// MPC z0; else shift), edge CBF + formation gradient frozen ONCE outside the loop
// (B8), fixed P_ITERS of node -> edge -> dual, residual/h2 measurement, warm-start
// storage. z is stored PER EDGE (B1). Implementations in src/admm_core/coordinator.cpp.
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "legged_upper_control/admm_constants.hpp"
#include "legged_upper_control/admm_edge_qp.hpp"
#include "legged_upper_control/admm_formation.hpp"
#include "legged_upper_control/admm_node_qp.hpp"
#include "legged_upper_control/admm_rti.hpp"

namespace admm {

// Node update (19) neighbor sum: sum_{j in N(i)} (z^{ij,i} - lam^{ij,i}).
using EdgeKey = std::pair<int, int>;
using EdgeVecs = std::map<EdgeKey, std::map<int, Eigen::VectorXd>>;

Eigen::VectorXd consensus_target(const EdgeVecs& z, const EdgeVecs& lam, int i,
                                 const std::vector<int>& neighbors);

class ADMMCoordinator {
public:
    struct Hist {
        std::vector<double> r_prim, r_dual, h2_viol;
        std::vector<double> r_pos, r_vel;  // max |xi-z| in position (m) / velocity (m/s), per iter
        int edge_fail = 0;
    };

    // formation: non-owning (nullptr = off); binding keeps the Python-side object
    // alive. obstacles/walls feed every node's local CBF.
    // parallel: OpenMP over the (independent) node and edge solves inside each
    // ADMM iteration. Results are bit-identical to sequential (own OSQP workspace
    // per subproblem); only wall-clock changes. Default OFF.
    ADMMCoordinator(int p_iters, double rho, std::vector<int> dogs,
                    std::vector<EdgeKey> edges, const LaplacianFormation* formation,
                    double w_form, std::vector<Obstacle> obstacles,
                    std::vector<Wall> walls, int hard_through, bool parallel = false,
                    double robot_margin = 0.30);

    // One control cycle. xnow[i] = state4, xdes[i] = (N,4).
    std::pair<std::map<int, Eigen::VectorXd>, Hist> step(
        const std::map<int, Eigen::VectorXd>& xnow,
        const std::map<int, Eigen::MatrixXd>& xdes);

    int N_;
    int cycle_ = 0;

    // read-only warm-start state (verify gates inspect these; None before cycle 1)
    bool has_prev() const { return has_prev_; }
    const EdgeVecs& prev_z() const { return prev_z_; }
    const EdgeVecs& prev_lam() const { return prev_lam_; }
    const std::map<int, std::vector<int>>& neighbors() const { return neighbors_; }
    // Raw obstacle list (mirrors Python self.obstacles): the publisher reads this to
    // build its A* planner. Stored here so the C++ coordinator is a drop-in.
    const std::vector<Obstacle>& obstacles() const { return obstacles_; }

private:
    std::map<int, Eigen::VectorXd> uncoupled_(
        const std::map<int, Eigen::VectorXd>& xnow,
        const std::map<int, Eigen::MatrixXd>& xdes);
    double h2_viol_(const std::map<int, Eigen::VectorXd>& xi,
                    const std::map<int, Eigen::VectorXd>& xnow) const;
    std::map<int, Eigen::MatrixX2d> formation_grad_(
        const std::map<int, Eigen::VectorXd>& xibar) const;

    double rho_;
    int P_;
    int hard_through_;
    bool parallel_;
    std::vector<int> dogs_;
    std::vector<EdgeKey> edges_;
    const LaplacianFormation* formation_;
    double w_form_;
    std::vector<Obstacle> obstacles_;
    std::map<int, std::vector<int>> neighbors_;
    std::map<int, std::unique_ptr<NodeSubproblem>> node_;
    std::map<EdgeKey, std::unique_ptr<EdgeSubproblem>> edge_;
    int nz_;
    bool has_prev_ = false;
    std::map<int, Eigen::VectorXd> prev_xi_;
    EdgeVecs prev_z_, prev_lam_;
};

}  // namespace admm
