// C++ mirror of core/planning.py AStarPlanner (used subset) — see astar_planner.hpp.
#include "legged_upper_control/astar_planner.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <queue>

namespace admm {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

inline int64_t cell_key(int ix, int iy) {
    return (static_cast<int64_t>(ix) << 32) ^
           static_cast<int64_t>(static_cast<uint32_t>(iy));
}

// heap entry; std::greater-style comparator over the (f, g, x, y) tuple.
struct Entry {
    double f, g;
    int x, y;
};
struct EntryGreater {
    bool operator()(const Entry& a, const Entry& b) const {
        if (a.f != b.f) return a.f > b.f;
        if (a.g != b.g) return a.g > b.g;
        if (a.x != b.x) return a.x > b.x;
        return a.y > b.y;
    }
};

// _MOVES: 8-connected, same order as the Python class constant.
struct Move {
    int dx, dy;
    double cost;
};
const Move kMoves[8] = {
    {1, 0, 1.0},  {-1, 0, 1.0}, {0, 1, 1.0},  {0, -1, 1.0},
    {1, 1, std::sqrt(2.0)},  {1, -1, std::sqrt(2.0)},
    {-1, 1, std::sqrt(2.0)}, {-1, -1, std::sqrt(2.0)},
};

}  // namespace

double py_hypot(double x, double y) {
    // CPython 3.8 math_hypot + vector_norm, argument order preserved.
    double vec[2] = {std::fabs(x), std::fabs(y)};
    double max = 0.0;
    bool found_nan = false;
    for (double v : vec) {
        if (std::isnan(v)) found_nan = true;
        if (v > max) max = v;
    }
    if (std::isinf(max)) return max;
    if (found_nan) return std::numeric_limits<double>::quiet_NaN();
    if (max == 0.0) return max;
    double csum = 1.0, frac = 0.0;
    for (double v : vec) {
        v /= max;
        v = v * v;
        const double oldcsum = csum;
        csum += v;
        frac += (oldcsum - csum) + v;
    }
    return max * std::sqrt(csum - 1.0 + frac);
}

int py_round_int(double v) {
    return static_cast<int>(std::nearbyint(v));  // FE_TONEAREST = half-to-even
}

AStarPlanner::AStarPlanner(double resolution, double robot_radius,
                           std::vector<AStarCircle> obstacles, double x_min,
                           double x_max, double y_min, double y_max,
                           double boundary_margin,
                           std::vector<AStarRect> rect_obstacles)
    : res_(resolution),
      robot_radius_(robot_radius),
      boundary_margin_(boundary_margin),
      x_min_(x_min),
      x_max_(x_max),
      y_min_(y_min),
      y_max_(y_max),
      obstacles_(std::move(obstacles)),
      rects_(std::move(rect_obstacles)) {
    nx_ = py_round_int((x_max_ - x_min_) / res_) + 1;
    ny_ = py_round_int((y_max_ - y_min_) / res_) + 1;
    omap_ = build_map(robot_radius_);
}

std::vector<uint8_t> AStarPlanner::build_map(double inflate) const {
    // xs[i] = i*res + x_min (np.arange(nx)*res + x_min); conditions OR'd per cell
    // in the Python mask order (walls, circles, rects) — OR is order-free.
    std::vector<uint8_t> omap(static_cast<size_t>(nx_) * ny_, 0);
    const double wall_inflate = std::max(boundary_margin_, inflate);
    for (int i = 0; i < nx_; ++i) {
        const double x = static_cast<double>(i) * res_ + x_min_;
        for (int j = 0; j < ny_; ++j) {
            const double y = static_cast<double>(j) * res_ + y_min_;
            bool occ = x <= x_min_ + wall_inflate || x >= x_max_ - wall_inflate ||
                       y <= y_min_ + wall_inflate || y >= y_max_ - wall_inflate;
            for (const auto& o : obstacles_) {
                if (occ) break;
                const double r = o.r + inflate;
                occ = (x - o.px) * (x - o.px) + (y - o.py) * (y - o.py) <= r * r;
            }
            for (const auto& rc : rects_) {
                if (occ) break;
                const double margin = std::max(rc.margin, inflate);
                const double hx = 0.5 * rc.sx + margin;
                const double hy = 0.5 * rc.sy + margin;
                occ = std::fabs(x - rc.cx) <= hx && std::fabs(y - rc.cy) <= hy;
            }
            if (occ) omap[static_cast<size_t>(i) * ny_ + j] = 1;
        }
    }
    return omap;
}

std::pair<int, int> AStarPlanner::w2g(double x, double y) const {
    return {py_round_int((x - x_min_) / res_), py_round_int((y - y_min_) / res_)};
}

Eigen::Vector2d AStarPlanner::g2w(int ix, int iy) const {
    return {x_min_ + ix * res_, y_min_ + iy * res_};
}

bool AStarPlanner::is_free(int ix, int iy) const {
    return 0 <= ix && ix < nx_ && 0 <= iy && iy < ny_ &&
           !omap_[static_cast<size_t>(ix) * ny_ + iy];
}

std::vector<Eigen::Vector2d> AStarPlanner::plan(const Eigen::Vector2d& start,
                                                const Eigen::Vector2d& goal) const {
    return plan_on_map(start, goal);
}

std::vector<Eigen::Vector2d> AStarPlanner::plan_on_map(
    const Eigen::Vector2d& start, const Eigen::Vector2d& goal) const {
    const auto [sx, sy] = w2g(start(0), start(1));
    const auto [gx, gy] = w2g(goal(0), goal(1));
    if (!is_free(gx, gy)) return {};
    std::unordered_map<int64_t, double> g_cost;
    std::unordered_map<int64_t, int64_t> came_from;
    g_cost[cell_key(sx, sy)] = 0.0;
    std::priority_queue<Entry, std::vector<Entry>, EntryGreater> heap;
    heap.push({py_hypot(static_cast<double>(sx - gx), static_cast<double>(sy - gy)),
               0.0, sx, sy});
    while (!heap.empty()) {
        const Entry cur = heap.top();
        heap.pop();
        const int cx = cur.x, cy = cur.y;
        const auto it = g_cost.find(cell_key(cx, cy));
        const double best = it == g_cost.end() ? kInf : it->second;
        if (cur.g > best + 1e-9) continue;
        if (cx == gx && cy == gy) {
            std::vector<Eigen::Vector2d> path;
            int64_t node = cell_key(gx, gy);
            for (auto f = came_from.find(node); f != came_from.end();
                 f = came_from.find(node)) {
                path.push_back(g2w(static_cast<int>(node >> 32),
                                   static_cast<int>(static_cast<uint32_t>(node))));
                node = f->second;
            }
            path.push_back(g2w(sx, sy));
            std::reverse(path.begin(), path.end());
            return path;
        }
        for (const auto& mv : kMoves) {
            const int nbx = cx + mv.dx, nby = cy + mv.dy;
            if (!is_free(nbx, nby)) continue;
            if (mv.dx != 0 && mv.dy != 0) {  // no diagonal corner cutting
                if (!is_free(cx + mv.dx, cy) || !is_free(cx, cy + mv.dy)) continue;
            }
            const double new_g = cur.g + mv.cost;
            const auto nb = g_cost.find(cell_key(nbx, nby));
            if (new_g < (nb == g_cost.end() ? kInf : nb->second)) {
                g_cost[cell_key(nbx, nby)] = new_g;
                came_from[cell_key(nbx, nby)] = cell_key(cx, cy);
                const double h = py_hypot(static_cast<double>(nbx - gx),
                                          static_cast<double>(nby - gy));
                heap.push({new_g + h, new_g, nbx, nby});
            }
        }
    }
    return {};
}

std::vector<std::pair<double, Eigen::Vector2d>>
AStarPlanner::nearest_free_candidates(const Eigen::Vector2d& goal) const {
    const auto [gx, gy] = w2g(goal(0), goal(1));
    const int max_r = 20;  // max_dist=None path only (legacy max_dist not ported)
    std::vector<std::pair<double, Eigen::Vector2d>> candidates;
    std::unordered_set<int64_t> seen;
    auto add_grid = [&](int ix, int iy) {
        if (seen.count(cell_key(ix, iy)) || !is_free(ix, iy)) return;
        const Eigen::Vector2d w = g2w(ix, iy);
        const double dist = py_hypot(w(0) - goal(0), w(1) - goal(1));
        seen.insert(cell_key(ix, iy));
        candidates.emplace_back(dist, w);
    };
    add_grid(gx, gy);
    // radial escape when the goal lies inside an inflated obstacle
    const double inflate = std::max(0.0, robot_radius_);  // planning margin == base
    for (const auto& o : obstacles_) {
        const double r = o.r + inflate;
        double dx = goal(0) - o.px, dy = goal(1) - o.py;
        double dist = std::sqrt(dx * dx + dy * dy);  // np.linalg.norm 2-vec
        if (dist > r + res_) continue;
        if (dist < 1e-6) {
            dx = 1.0;
            dy = 0.0;
            dist = 1.0;
        }
        const double proj_x = o.px + dx / dist * (r + res_);
        const double proj_y = o.py + dy / dist * (r + res_);
        const auto [pix, piy] = w2g(proj_x, proj_y);
        for (int ddx = -2; ddx <= 2; ++ddx)
            for (int ddy = -2; ddy <= 2; ++ddy) add_grid(pix + ddx, piy + ddy);
    }
    for (int r = 1; r <= max_r; ++r) {
        for (int dx = -r; dx <= r; ++dx) {
            add_grid(gx + dx, gy - r);
            add_grid(gx + dx, gy + r);
        }
        for (int dy = -r + 1; dy <= r - 1; ++dy) {
            add_grid(gx - r, gy + dy);
            add_grid(gx + r, gy + dy);
        }
    }
    // Python list.sort(key=dist) is stable; insertion order above is mirrored
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    return candidates;
}

bool AStarPlanner::goal_candidate_has_clearance(const Eigen::Vector2d& point) const {
    const double inflate = std::max(0.0, robot_radius_);
    for (const auto& o : obstacles_) {
        const double r = o.r + inflate;
        const double dx = point(0) - o.px, dy = point(1) - o.py;
        const double clearance = std::sqrt(dx * dx + dy * dy) - r;
        if (clearance < goal_min_obstacle_clearance_) return false;
    }
    const double wall_inflate = std::max(boundary_margin_, inflate);
    const double clearances[4] = {
        point(0) - (x_min_ + wall_inflate),
        (x_max_ - wall_inflate) - point(0),
        point(1) - (y_min_ + wall_inflate),
        (y_max_ - wall_inflate) - point(1),
    };
    double min_c = clearances[0];  // Python min(): first minimum wins
    for (int i = 1; i < 4; ++i)
        if (clearances[i] < min_c) min_c = clearances[i];
    return min_c >= goal_min_wall_clearance_;
}

AStarPlanner::ReachableGoal AStarPlanner::find_reachable_goal(
    const Eigen::Vector2d& start, const Eigen::Vector2d& goal) const {
    const auto [gx, gy] = w2g(goal(0), goal(1));
    const bool goal_is_free = is_free(gx, gy);
    auto candidates = nearest_free_candidates(goal);
    if (candidates.empty()) return {};
    if (!goal_is_free) {
        std::vector<std::pair<double, Eigen::Vector2d>> safe;
        for (const auto& c : candidates)
            if (goal_candidate_has_clearance(c.second)) safe.push_back(c);
        if (!safe.empty())
            candidates = std::move(safe);
        else
            return {};
    }
    for (const auto& c : candidates) {
        auto path = plan_on_map(start, c.second);
        if (!path.empty()) return {true, c.second, std::move(path)};
    }
    return {};
}

}  // namespace admm
