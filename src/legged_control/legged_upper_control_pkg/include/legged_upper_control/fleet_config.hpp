#pragma once
// C6 fleet configuration + slot math — C++ mirror of the module-level data and
// functions of scripts/ocs2_fleet_publisher.py (ARENAS / FORMATIONS / DEFAULT_GOALS,
// rot2d / centroid_slot_targets / min_cost_assignment). Single source of truth:
// the pybind module re-exports these so the Python side reads the same literals
// (test_cpp_parity.py test_c6_fleet_*).
#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

#include "legged_upper_control/admm_node_qp.hpp"  // Obstacle, Wall

namespace admm {

struct ArenaRect {  // A*-only wall box {center, size}
    Eigen::Vector2d center;
    Eigen::Vector2d size;
};

struct Arena {
    std::vector<Obstacle> obstacles;
    std::vector<Wall> walls;      // empty for arenas without a "walls" key
    std::vector<ArenaRect> rects; // empty for arenas without a "rects" key
    std::map<int, Eigen::Vector2d> goals;
};

const std::map<std::string, Arena>& arenas();
const std::map<std::string, std::vector<Eigen::Vector2d>>& formations();
const std::map<int, Eigen::Vector2d>& default_goals();

// 2D rotation matrix (world frame).
Eigen::Matrix2d rot2d(double yaw);

// Per-slot world targets so the fleet CENTROID lands on goal_c, V facing yaw.
std::vector<Eigen::Vector2d> centroid_slot_targets(
    const Eigen::Vector2d& goal_c, const std::vector<Eigen::Vector2d>& offsets,
    double yaw);

// Permutation of slot indices minimising sum ||pos[k]-slot[perm[k]]||^2
// (lexicographic-first on ties, mirroring itertools.permutations + strict <).
std::vector<int> min_cost_assignment(const std::vector<Eigen::Vector2d>& positions,
                                     const std::vector<Eigen::Vector2d>& slots);

}  // namespace admm
