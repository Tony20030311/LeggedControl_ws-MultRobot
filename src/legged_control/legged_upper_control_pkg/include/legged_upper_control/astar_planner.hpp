#pragma once
// C6 A* planner — C++ mirror of core/planning.py AStarPlanner, USED SUBSET ONLY.
//
// ponytail: only the surface the ADMM path calls is ported — ctor, plan(),
// find_reachable_goal(). The legacy-only surface (set_planning_margin, narrow
// wall-gap blocking, forbidden zones, nearest_free, segment_crosses_occupied,
// max_dist) stays Python (fleet/manager.py); port it if the legacy stack ever
// migrates. With planning_margin == robot_radius the plan() fallback branch of
// the Python original is dead and is not mirrored.
//
// Bit-parity notes (test_cpp_parity.py test_c6_astar_*):
// - py_hypot mirrors CPython 3.8 math.hypot (scaled compensated sum) — libm
//   hypot differs in the last bit on ~70% of random float pairs.
// - Python round() is round-half-to-even -> std::nearbyint, NOT std::round.
// - heap entries (f, g, x, y) are pairwise distinct with a strict total order
//   (a re-push of the same cell requires strictly smaller g), so ANY min-heap
//   pops in exactly sorted order — std::priority_queue needs no heapq mirror.
#include <Eigen/Dense>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace admm {

// CPython 3.8 Modules/mathmodule.c vector_norm() for the 2-arg case.
double py_hypot(double x, double y);

// Python int(round(v)): round-half-to-even to integer.
int py_round_int(double v);

struct AStarCircle {
    double px, py;
    double r;  // astar_radius if given else radius (resolved at parse time)
};

struct AStarRect {
    double cx, cy, sx, sy;
    double margin;  // astar_margin if given else robot_radius (resolved at parse)
};

class AStarPlanner {
public:
    AStarPlanner(double resolution, double robot_radius,
                 std::vector<AStarCircle> obstacles,
                 double x_min = 0.0, double x_max = 10.0,
                 double y_min = -5.0, double y_max = 5.0,
                 double boundary_margin = 0.45,
                 std::vector<AStarRect> rect_obstacles = {});

    // World-frame waypoints (possibly empty). Mirrors plan() with the dead
    // fallback branch elided (planning margin is always the base radius here).
    std::vector<Eigen::Vector2d> plan(const Eigen::Vector2d& start,
                                      const Eigen::Vector2d& goal) const;

    // (has_candidate, candidate, path). Mirrors find_reachable_goal(max_dist=None).
    struct ReachableGoal {
        bool has_candidate = false;
        Eigen::Vector2d candidate{0.0, 0.0};
        std::vector<Eigen::Vector2d> path;
    };
    ReachableGoal find_reachable_goal(const Eigen::Vector2d& start,
                                      const Eigen::Vector2d& goal) const;

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    const std::vector<uint8_t>& omap() const { return omap_; }

private:
    std::pair<int, int> w2g(double x, double y) const;
    Eigen::Vector2d g2w(int ix, int iy) const;
    bool is_free(int ix, int iy) const;
    std::vector<uint8_t> build_map(double inflate) const;
    std::vector<Eigen::Vector2d> plan_on_map(const Eigen::Vector2d& start,
                                             const Eigen::Vector2d& goal) const;
    // (dist, point) candidates sorted stably by dist, mirrored insertion order.
    std::vector<std::pair<double, Eigen::Vector2d>> nearest_free_candidates(
        const Eigen::Vector2d& goal) const;
    bool goal_candidate_has_clearance(const Eigen::Vector2d& point) const;

    double res_, robot_radius_, boundary_margin_;
    double x_min_, x_max_, y_min_, y_max_;
    int nx_, ny_;
    std::vector<AStarCircle> obstacles_;
    std::vector<AStarRect> rects_;
    std::vector<uint8_t> omap_;  // [ix * ny_ + iy]
    // Python instance defaults (setters are legacy-only and not ported)
    double goal_min_obstacle_clearance_ = 0.20;
    double goal_min_wall_clearance_ = 0.65;
};

}  // namespace admm
