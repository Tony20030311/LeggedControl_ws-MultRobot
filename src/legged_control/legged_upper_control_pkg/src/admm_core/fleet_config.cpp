// C++ mirror of scripts/ocs2_fleet_publisher.py module-level data + slot math.
// Literals MUST stay byte-identical to the Python source (parity gate compares
// the re-exported py dicts against the publisher's until the .py is deleted).
#include "legged_upper_control/fleet_config.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace admm {

const std::map<std::string, Arena>& arenas() {
    static const std::map<std::string, Arena> kArenas = [] {
        std::map<std::string, Arena> a;
        a["obstacles"] = Arena{
            {{{4.0, 0.40}, 0.30}, {{5.5, 0.40}, 0.30}},
            {},
            {},
            {{1, {7.0, 0.0}}, {2, {6.3, 0.5}}, {3, {6.3, -0.5}}},
        };
        Arena plum;
        for (const auto& p : std::vector<Eigen::Vector2d>{
                 {3.4, 1.1}, {3.4, -1.1}, {4.7, 0.0}, {4.7, 2.1}, {4.7, -2.1},
                 {6.0, 1.1}, {6.0, -1.1}, {7.3, 0.0}, {7.3, 2.1}, {7.3, -2.1}})
            plum.obstacles.push_back({p, 0.30});
        plum.goals = {{1, {9.0, 0.0}}, {2, {8.3, 0.5}}, {3, {8.3, -0.5}}};
        a["plum"] = plum;
        Arena door;
        for (const auto& p : std::vector<Eigen::Vector2d>{
                 {7.5, 2.5}, {8.5, -1.5}, {9.0, 3.5}, {7.0, -3.0}, {8.0, 0.5}})
            door.obstacles.push_back({p, 0.30});  // 5 field cylinders (r_eff 0.60)
        for (const auto& p : std::vector<Eigen::Vector2d>{
                 {6.0, 1.25}, {6.0, 1.75}, {6.0, -1.25}, {6.0, -1.75}})
            door.obstacles.push_back({p, 0.15});  // door posts (r_eff 0.45)
        door.walls = {
            {{-1.0, 0.0}, {9.925, 0.0}, 0.30},  // right x=10
            {{0.0, -1.0}, {0.0, 4.925}, 0.30},  // top y=5
            {{0.0, 1.0}, {0.0, -4.925}, 0.30},  // bottom y=-5
        };
        door.rects = {
            {{6.0, 3.125}, {0.15, 3.75}},   // wall_left_upper
            {{6.0, -3.125}, {0.15, 3.75}},  // wall_left_lower
            {{10.0, 0.0}, {0.15, 10.0}},    // wall_right
            {{8.0, 5.0}, {4.0, 0.15}},      // wall_top
            {{8.0, -5.0}, {4.0, 0.15}},     // wall_bottom
        };
        door.goals = {{1, {9.3, 0.5}}, {2, {8.6, 1.0}}, {3, {8.6, 0.0}}};
        a["door"] = door;
        return a;
    }();
    return kArenas;
}

const std::map<std::string, std::vector<Eigen::Vector2d>>& formations() {
    static const std::map<std::string, std::vector<Eigen::Vector2d>> kFormations = {
        {"V", {{0.0, 0.0}, {-0.7, 0.5}, {-0.7, -0.5}}},
        {"column", {{0.0, 0.0}, {-0.7, 0.0}, {-1.4, 0.0}}},
        {"V_wide", {{0.0, 0.0}, {-0.7, 0.8}, {-0.7, -0.8}}},
    };
    return kFormations;
}

const std::map<int, Eigen::Vector2d>& default_goals() {
    static const std::map<int, Eigen::Vector2d> kGoals = {
        {1, {3.0, 0.0}}, {2, {2.3, 0.5}}, {3, {2.3, -0.5}}};
    return kGoals;
}

Eigen::Matrix2d rot2d(double yaw) {
    // GCC fuses adjacent cos()+sin() into glibc sincos(), whose cos differs from
    // a separate cos() in the last bit; Python calls them separately. Volatile
    // fn pointers force two real libm calls (bit parity, compiler-proof).
    static double (*volatile p_cos)(double) = static_cast<double (*)(double)>(&std::cos);
    static double (*volatile p_sin)(double) = static_cast<double (*)(double)>(&std::sin);
    const double c = p_cos(yaw), s = p_sin(yaw);
    Eigen::Matrix2d r;
    r << c, -s, s, c;
    return r;
}

std::vector<Eigen::Vector2d> centroid_slot_targets(
    const Eigen::Vector2d& goal_c, const std::vector<Eigen::Vector2d>& offsets,
    double yaw) {
    // np.mean(offsets, axis=0): sequential per-component sum, then / n
    const auto n = offsets.size();
    Eigen::Vector2d mean(0.0, 0.0);
    for (const auto& o : offsets) {
        mean(0) += o(0);
        mean(1) += o(1);
    }
    mean(0) /= static_cast<double>(n);
    mean(1) /= static_cast<double>(n);
    const Eigen::Matrix2d R = rot2d(yaw);
    std::vector<Eigen::Vector2d> slots;
    slots.reserve(n);
    for (const auto& o : offsets) {
        const double ox = o(0) - mean(0), oy = o(1) - mean(1);
        // R @ o mirrored per component, then goal_c + ...
        slots.emplace_back(goal_c(0) + (R(0, 0) * ox + R(0, 1) * oy),
                           goal_c(1) + (R(1, 0) * ox + R(1, 1) * oy));
    }
    return slots;
}

std::vector<int> min_cost_assignment(const std::vector<Eigen::Vector2d>& positions,
                                     const std::vector<Eigen::Vector2d>& slots) {
    const int n = static_cast<int>(positions.size());
    std::vector<int> best(n), perm(n);
    std::iota(best.begin(), best.end(), 0);
    std::iota(perm.begin(), perm.end(), 0);
    double bestcost = 1e18;
    do {  // std::next_permutation from identity == itertools lexicographic order
        double c = 0.0;  // Python sum(): sequential
        for (int k = 0; k < n; ++k) {
            const double dx = positions[k](0) - slots[perm[k]](0);
            const double dy = positions[k](1) - slots[perm[k]](1);
            c += dx * dx + dy * dy;  // np.dot 2-vec
        }
        if (c < bestcost) {
            bestcost = c;
            best = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

}  // namespace admm
