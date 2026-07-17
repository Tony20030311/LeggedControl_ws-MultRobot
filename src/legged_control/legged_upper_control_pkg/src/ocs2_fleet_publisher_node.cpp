// ocs2_fleet_publisher_node.cpp — roscpp mirror of scripts/ocs2_fleet_publisher.py
// (FleetPublisher). Links admm_core directly (no pybind). Behaviour-equivalent, not
// bit-identical: verified by scratchpad/arena/smoke_fleet_node.py (same gate the rospy
// shell passes) and the C6g Gazebo campaign. See legged_upper_control_pkg/CLAUDE.md.
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_msgs/mpc_target_trajectories.h>
#include <ocs2_msgs/mpc_state.h>
#include <ocs2_msgs/mpc_input.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "legged_upper_control/admm_constants.hpp"
#include "legged_upper_control/admm_reference.hpp"
#include "legged_upper_control/admm_motion_adapter.hpp"
#include "legged_upper_control/admm_formation.hpp"
#include "legged_upper_control/astar_planner.hpp"
#include "legged_upper_control/fleet_config.hpp"
#include "legged_upper_control/admm_coordinator.hpp"
#include "legged_upper_control/admm_node_qp.hpp"  // Obstacle, Wall

// OSQP's osqp.h (pulled in via admm_qp_common.hpp) re-#defines RHO as a macro after
// admm_constants.hpp #undef'd it; undo it here so admm::RHO resolves in this TU.
#ifdef RHO
#undef RHO
#endif

using admm::BASE_PX; using admm::BASE_PY; using admm::BASE_YAW;
using admm::MOM_LIN_X; using admm::MOM_LIN_Y;

namespace {
Eigen::VectorXd a1_default_joints() {
  Eigen::VectorXd j(12);
  j << 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5;
  return j;
}
Eigen::MatrixX2d to_mat(const std::vector<Eigen::Vector2d>& v) {
  Eigen::MatrixX2d m(static_cast<int>(v.size()), 2);
  for (int k = 0; k < static_cast<int>(v.size()); ++k) { m(k, 0) = v[k][0]; m(k, 1) = v[k][1]; }
  return m;
}
double clip(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
}  // namespace

class FleetPublisherNode {
 public:
  FleetPublisherNode() : pnh_("~") {
    dogs_ = {1, 2, 3};
    edges_ = {{1, 2}, {1, 3}, {2, 3}};
    N_ = admm::N; ts_ = admm::TS;
    pnh_.param("v", v_, 0.15);
    pnh_.param("com_height", com_height_, 0.30);
    pnh_.param<std::string>("formation", formation_name_, "V");
    pnh_.param("w_form", w_form_, 0.3);   // loose default: single-file through tight gaps
    pnh_.param<std::string>("arena", arena_name_, "");
    // no arena given -> use the one the scene launch set
    if (arena_name_.empty()) ros::param::get("/admm_arena", arena_name_);

    // arena selection: "" (or unknown) -> empty world (no obstacles, DEFAULT_GOALS)
    const auto& arenas = admm::arenas();
    auto ait = arenas.find(arena_name_);
    std::vector<admm::Obstacle> arena_obs;
    std::vector<admm::Wall> arena_walls;
    std::map<int, Eigen::Vector2d> arena_goals = admm::default_goals();
    if (ait != arenas.end()) {
      arena_obs = ait->second.obstacles;
      arena_walls = ait->second.walls;
      arena_rects_ = ait->second.rects;
      arena_goals = ait->second.goals;
    }
    for (int i : dogs_) {
      double gx, gy;
      pnh_.param("goal" + std::to_string(i) + "_x", gx, arena_goals[i][0]);
      pnh_.param("goal" + std::to_string(i) + "_y", gy, arena_goals[i][1]);
      goal_[i] = Eigen::Vector2d(gx, gy);
    }

    formation_.reset(new admm::LaplacianFormation(admm::formations()));
    formation_->set_formation(formation_name_);
    int hard_through; pnh_.param("hard_through", hard_through, 1);  // diag: # of hard edge-CBF steps (k=0..hard_through-1)
    double robot_margin; pnh_.param("robot_margin", robot_margin, 0.30);  // obstacle CBF inflation: r_eff = radius + this
    coord_.reset(new admm::ADMMCoordinator(admm::P_ITERS, admm::RHO, dogs_, edges_,
                                           formation_.get(), w_form_, arena_obs, arena_walls,
                                           hard_through, /*parallel=*/false, robot_margin));
    int lookahead; double pos_eps, yaw_rate_max;
    pnh_.param("lookahead_m", lookahead, 5);
    pnh_.param("pos_eps", pos_eps, 0.02);
    pnh_.param("yaw_rate_max", yaw_rate_max, 1.2);
    adapter_.reset(new admm::MotionAdapter(com_height_, a1_default_joints(), N_, ts_,
                                           /*v_freeze=*/0.05, /*ema_alpha=*/0.2, /*input_dim=*/24,
                                           /*yaw_mode=*/"path", lookahead, pos_eps,
                                           /*k_send=*/admm::K_SEND, /*has_max_yaw_rate=*/true,
                                           yaw_rate_max));

    pnh_.param("yaw_latch_r", r_latch_, 0.25);
    pnh_.param("yaw_latch_margin", latch_margin_, 0.10);
    int k_send_p; double send_margin;
    pnh_.param("k_send", k_send_p, admm::K_SEND);
    pnh_.param("send_margin", send_margin, 0.0);
    send_cap_ = k_send_p; d_safe_ = admm::D_MIN + send_margin;

    // #1 leader-aware follower speed (ACC/car-following): a follower must not out-run a
    // slower dog ahead of it. follow_gain<=0 disables. desired 0.72 < nominal V gap 0.86
    // so it never brakes in same-speed formation; only bites when the dog ahead slows.
    pnh_.param("follow_gain", follow_gain_, 0.5);
    pnh_.param("follow_desired", follow_desired_, 0.72);
    pnh_.param("follow_range", follow_range_, 1.5);
    double cone_deg; pnh_.param("follow_cone_deg", cone_deg, 60.0);
    follow_cone_cos_ = std::cos(cone_deg * M_PI / 180.0);
    // ACC cap floor: never brake below this. A PARKED dog ahead would otherwise pin the
    // cap at ~0 forever (passby deadlock, GB7 leg2: dog3 held 105s at cap 0.01) -- the
    // floor keeps the reference advancing so the CBF slides the dog around laterally,
    // while still shedding 75% of the approach speed (brake semantics preserved).
    pnh_.param("follow_floor", follow_floor_, 0.05);

    pnh_.param("use_astar", use_astar_, false);
    pnh_.param("astar_robot_radius", r_astar_, 0.35);
    pnh_.param("astar_res", astar_res_, 0.15);
    pnh_.param("astar_x_min", ax_min_, 0.0);
    pnh_.param("astar_x_max", ax_max_, 10.0);
    pnh_.param("astar_y_min", ay_min_, -5.0);
    pnh_.param("astar_y_max", ay_max_, 5.0);
    if (use_astar_ && !coord_->obstacles().empty()) {
      std::vector<admm::AStarCircle> circles;
      for (const auto& o : coord_->obstacles())
        circles.push_back({o.pos[0], o.pos[1], o.radius});
      std::vector<admm::AStarRect> rects;
      for (const auto& r : arena_rects_)
        rects.push_back({r.center[0], r.center[1], r.size[0], r.size[1], r_astar_});
      planner_.reset(new admm::AStarPlanner(astar_res_, r_astar_, circles, ax_min_, ax_max_,
                                            ay_min_, ay_max_, /*boundary_margin=*/0.45, rects));
    }

    // settle-short rescue (detect + reassign + detour). Detector always logs; actions
    // gated by ~rescue. Fingerprint Gazebo-validated 2026-07-15: 3 true / 0 false fires.
    pnh_.param("rescue", rescue_, true);
    pnh_.param("rescue_stall_s", rescue_stall_s_, 5.0);
    pnh_.param("rescue_v_eps", rescue_v_eps_, 0.02);
    pnh_.param("rescue_slot_err", rescue_slot_err_, 0.5);
    pnh_.param("rescue_hyst", rescue_hyst_, 0.05);
    pnh_.param("rescue_cooldown_s", rescue_cooldown_s_, 10.0);

    for (int i : dogs_) {
      std::string ns = "dog" + std::to_string(i);
      pub_[i] = nh_.advertise<ocs2_msgs::mpc_target_trajectories>(
          "/" + ns + "/" + ns + "_mpc_target", 1);
      subs_.push_back(nh_.subscribe<ocs2_msgs::mpc_observation>(
          "/" + ns + "/" + ns + "_mpc_observation", 1,
          boost::bind(&FleetPublisherNode::obsCb, this, _1, i)));
      subs_.push_back(nh_.subscribe<nav_msgs::Odometry>(
          "/" + ns + "/ground_truth/state", 1,
          boost::bind(&FleetPublisherNode::gtCb, this, _1, i)));
      subs_.push_back(nh_.subscribe<geometry_msgs::PoseStamped>(
          "/" + ns + "/goal", 1, boost::bind(&FleetPublisherNode::goalCb, this, _1, i)));
    }
    subs_.push_back(nh_.subscribe<geometry_msgs::PoseStamped>(
        "/formation/goal", 1, boost::bind(&FleetPublisherNode::formationGoalCb, this, _1)));

    pnh_.param("viz_markers", viz_, true);
    pnh_.param<std::string>("viz_frame", viz_frame_, "world");
    if (viz_) marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/formation/admm_markers", 1);

    bool kill_cpp; pnh_.param("kill_cpp_target", kill_cpp, false);
    if (kill_cpp) killCppTargets();

    std::string default_csv = ros::package::getPath("legged_upper_control") +
                              "/docs/progress/fleet_track.csv";
    pnh_.param<std::string>("log_csv", csv_path_, default_csv);
    csv_.open(csv_path_.c_str());
    if (csv_.is_open()) {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "# fleet rung2 formation=%s w_form=%.1f v=%.3f\n",
                    formation_name_.c_str(), w_form_, v_);
      csv_ << buf << "t,";
      for (size_t d = 0; d < dogs_.size(); ++d)
        for (size_t c = 0; c < kDbgCols.size(); ++c)
          csv_ << kDbgCols[c] << dogs_[d] << ((d + 1 == dogs_.size() && c + 1 == kDbgCols.size()) ? "" : ",");
      csv_ << "\n"; csv_.flush();
    }
    std::string hpath; pnh_.param<std::string>("log_hist_csv", hpath, "");
    if (!hpath.empty()) {
      hist_csv_.open(hpath.c_str());
      if (hist_csv_.is_open()) hist_csv_ << "# ADMM residuals; row = t, r_prim[0..P-1], r_dual[0..P-1], planchg_pos[0..P-1](m), planchg_vel[0..P-1](m/s)\n";
    }

    timer_ = nh_.createTimer(ros::Duration(ts_), &FleetPublisherNode::tick, this);
    ROS_INFO("[fleet_pub_cpp] dogs=3 formation=%s w_form=%.1f v=%.2f arena=%s obstacles=%zu",
             formation_name_.c_str(), w_form_, v_,
             arena_name_.empty() ? "(empty)" : arena_name_.c_str(), coord_->obstacles().size());
  }

 private:
  static const std::vector<std::string> kDbgCols;

  void obsCb(const ocs2_msgs::mpc_observation::ConstPtr& m, int i) { obs_[i] = *m; has_obs_[i] = true; }
  void gtCb(const nav_msgs::Odometry::ConstPtr& m, int i) { gt_[i] = *m; has_gt_[i] = true; }

  void goalCb(const geometry_msgs::PoseStamped::ConstPtr& m, int i) {
    goal_[i] = Eigen::Vector2d(m->pose.position.x, m->pose.position.y);
    path_.erase(i); yaw_latched_[i] = false;
    ROS_INFO("[fleet_pub_cpp] dog%d new goal (%.2f,%.2f)", i, goal_[i][0], goal_[i][1]);
    have_slots_ = false;  // per-dog mode: no slots, rescue disarmed
  }

  void formationGoalCb(const geometry_msgs::PoseStamped::ConstPtr& m) {
    for (int i : dogs_) if (!has_gt_[i]) { ROS_WARN("[fleet_pub_cpp] /formation/goal ignored: no state"); return; }
    Eigen::Vector2d goal_c(m->pose.position.x, m->pose.position.y);
    std::vector<Eigen::Vector2d> pos;
    Eigen::Vector2d centroid(0, 0);
    for (int i : dogs_) {
      Eigen::Vector2d p(gt_[i].pose.pose.position.x, gt_[i].pose.pose.position.y);
      pos.push_back(p); centroid += p;
    }
    centroid /= static_cast<double>(dogs_.size());
    Eigen::Vector2d d = goal_c - centroid;
    double yaw = (d.norm() > 0.25) ? std::atan2(d[1], d[0]) : last_formation_yaw_;
    last_formation_yaw_ = yaw;
    const auto* offsets = formation_->current_offsets();
    if (offsets == nullptr || offsets->size() != dogs_.size()) {
      ROS_WARN("[fleet_pub_cpp] /formation/goal ignored: no formation offsets"); return;
    }
    slots_ = admm::centroid_slot_targets(goal_c, *offsets, yaw);
    assign_ = admm::min_cost_assignment(pos, slots_);
    // a new goal is a clean slate for the rescue state: rescued_until_ too, else a per-dog
    // lockout earned under the PREVIOUS goal silently skips this dog for up to 2*cooldown.
    have_slots_ = true; stall_since_d_.clear(); cooldown_until_ = ros::Time(0);
    rescued_until_.clear();
    for (size_t k = 0; k < dogs_.size(); ++k) {
      goal_[dogs_[k]] = slots_[assign_[k]]; path_.erase(dogs_[k]); yaw_latched_[dogs_[k]] = false;
    }
    ROS_INFO("[fleet_pub_cpp] /formation/goal centroid (%.2f,%.2f) yaw=%.2f", goal_c[0], goal_c[1], yaw);
  }

  bool ready() const {
    for (int i : dogs_) {
      auto o = has_obs_.find(i); auto g = has_gt_.find(i);
      if (o == has_obs_.end() || !o->second) return false;
      if (g == has_gt_.end() || !g->second) return false;
      if (obs_.at(i).time == 0.0) return false;
    }
    return true;
  }

  // world X0 (px,py,vx,vy) + P0 (px,py): gt position + EMA/clamped obs velocity.
  void x0(int i, Eigen::Vector4d& X0, Eigen::Vector2d& P0) {
    const auto& s = obs_[i].state.value;
    const auto& gp = gt_[i].pose.pose.position;
    P0 = Eigen::Vector2d(gp.x, gp.y);
    Eigen::Vector2d vraw(s[MOM_LIN_X], s[MOM_LIN_Y]);
    if (!has_vema_[i]) { v_ema_[i] = vraw; has_vema_[i] = true; }
    else v_ema_[i] = 0.25 * vraw + 0.75 * v_ema_[i];
    double vx0 = clip(v_ema_[i][0], -admm::MAX_VX, admm::MAX_VX);
    double vy0 = clip(v_ema_[i][1], -admm::MAX_VY, admm::MAX_VY);
    X0 = Eigen::Vector4d(P0[0], P0[1], vx0, vy0);
  }

  Eigen::MatrixX2d waypoints(int i, const Eigen::Vector2d& P0i) {
    if (path_.find(i) != path_.end()) return to_mat(path_[i]);
    if (!planner_) return to_mat({P0i, goal_[i]});
    auto rg = planner_->find_reachable_goal(P0i, goal_[i]);
    path_[i] = rg.path.empty() ? std::vector<Eigen::Vector2d>{P0i, goal_[i]} : rg.path;
    return to_mat(path_[i]);
  }

  // #1 leader-aware follower cruise speed. Returns v_ unless a dog ahead of i (within the
  // travel-direction cone and follow_range) is slower: then ACC law caps i's speed at
  // v_ahead + gain*(gap - desired) so i decelerates to match rather than driving into it.
  // In normal formation all dogs cruise at ~v_ and gap>=desired -> returns v_ (no slowdown).
  double followSpeed(int i, const std::map<int, Eigen::VectorXd>& xnow) {
    if (follow_gain_ <= 0.0) return v_;
    const Eigen::Vector2d pi = xnow.at(i).head<2>();
    // travel direction = next reference waypoint when a path exists (an A* route or a
    // rescue detour bends away from the straight goal bearing; aiming the ACC cone at
    // the goal would brake for a teammate the path actually curves AROUND).
    Eigen::Vector2d tgt = goal_[i];
    const auto it = path_.find(i);
    if (it != path_.end() && it->second.size() >= 2) {
      // project onto the path first (nearest vertex), THEN look ahead from there --
      // scanning from the path start would aim the cone BACKWARD once the dog has
      // passed the first 0.3m, mutually braking any close pair (GB5 crawl bug).
      size_t k0 = 0; double dbest = 1e18;
      for (size_t k = 0; k < it->second.size(); ++k) {
        const double d = (it->second[k] - pi).norm();
        if (d < dbest) { dbest = d; k0 = k; }
      }
      for (size_t k = k0; k < it->second.size(); ++k)
        if ((it->second[k] - pi).norm() > 0.3) { tgt = it->second[k]; break; }
    }
    Eigen::Vector2d dir = tgt - pi;
    const double dn = dir.norm();
    if (dn < 1e-3) return v_;  // at goal: no travel direction to define "ahead"
    dir /= dn;
    double v_eff = v_;
    for (int j : dogs_) {
      if (j == i) continue;
      const Eigen::Vector2d d = xnow.at(j).head<2>() - pi;
      const double dist = d.norm();
      if (dist < 1e-3 || dist > follow_range_) continue;
      const double proj = d.dot(dir);
      if (proj <= 0.0 || proj / dist < follow_cone_cos_) continue;  // not ahead / outside cone
      const double v_ahead = std::max(0.0, xnow.at(j).segment<2>(2).dot(dir));
      const double v_allow = v_ahead + follow_gain_ * (dist - follow_desired_);
      v_eff = std::min(v_eff, std::max(v_allow, follow_floor_));
    }
    return v_eff;
  }

  // Settle-short deadlock detector. PER-DOG stall: dog i is stuck when it sits still
  // while > rescue_slot_err off its slot -- other dogs may well still be moving (a wedge
  // mid-transit stalls one dog at a time; the original fleet-wide stillness condition
  // detected the parked-fleet case but fired ~40s late on wedges, GB3 evidence).
  // Detector always logs; actions run only when ~rescue is true.
  void rescueTick(const std::map<int, Eigen::VectorXd>& xnow, const ros::Time& now) {
    if (!have_slots_) return;
    int cand = -1; double cerr = -1.0;
    for (int i : dogs_) {
      const double e = (xnow.at(i).head<2>() - goal_[i]).norm();
      const bool stuck = xnow.at(i).segment<2>(2).norm() < rescue_v_eps_ && e > rescue_slot_err_;
      if (!stuck) { stall_since_d_[i] = ros::Time(0); continue; }
      if (stall_since_d_[i].isZero()) { stall_since_d_[i] = now; continue; }
      // per-dog rescue cooldown: a freshly-detoured dog steps aside so the OTHER half of
      // a wedged pair gets routed too (GD leg7: worst-dog monopoly starved its partner)
      if (now < rescued_until_[i]) continue;
      if ((now - stall_since_d_[i]).toSec() >= rescue_stall_s_ && e > cerr) { cerr = e; cand = i; }
    }
    if (cand < 0 || now < cooldown_until_) return;
    std::string errs;
    for (int i : dogs_) {
      char b[16]; std::snprintf(b, sizeof(b), "%.2f", (xnow.at(i).head<2>() - goal_[i]).norm());
      errs += std::string(b) + (i == dogs_.back() ? "" : ",");
    }
    ROS_WARN("[rescue] FIRE errs=%s", errs.c_str());
    cooldown_until_ = now + ros::Duration(rescue_cooldown_s_);
    stall_since_d_.clear();
    if (!rescue_) return;
    // stage-1: re-match dogs->slots from CURRENT positions. Assignment was computed at
    // goal issuance from a transient formation; re-matching from the parked state gives a
    // cheap discrete escape when the deadlock is an assignment artifact. Hysteresis: only
    // accept a strictly better matching so symmetric ties never thrash.
    std::vector<Eigen::Vector2d> pos;
    for (int i : dogs_) pos.push_back(xnow.at(i).head<2>());
    const std::vector<int> a_new = admm::min_cost_assignment(pos, slots_);
    // score with the SAME metric min_cost_assignment minimises (sum of SQUARED distances,
    // fleet_config.cpp:117). Scoring its squared-optimal result by unsquared .norm() rejects
    // valid escapes: {2.00,0.10} sq 4.01 / sum 2.10 vs {1.40,1.40} sq 3.92 / sum 2.80 --
    // the solver picks the latter, an unsquared gate then vetoes it. rescue_hyst_ is now a
    // squared-distance margin (kept at its proven 0.05; it only has to break exact ties).
    double c_old = 0.0, c_new = 0.0;
    for (size_t k = 0; k < dogs_.size(); ++k) {
      c_old += (pos[k] - slots_[assign_[k]]).squaredNorm();
      c_new += (pos[k] - slots_[a_new[k]]).squaredNorm();
    }
    if (a_new != assign_ && c_new < c_old - rescue_hyst_) {
      for (size_t k = 0; k < dogs_.size(); ++k) {
        // ONLY touch dogs whose slot actually changed. Unlatching a dog parked ON its
        // goal re-arms the yaw-runaway the goal-proximity latch exists to prevent
        // (DF1 forensics 2026-07-16: dog1<->dog2 swap unlatched the UNINVOLVED dog3,
        // parked 0.06m from goal -> 0.57 m/s runaway -> QP infeasible -> HOLD storm).
        if (a_new[k] == assign_[k]) continue;
        goal_[dogs_[k]] = slots_[a_new[k]]; path_.erase(dogs_[k]); yaw_latched_[dogs_[k]] = false;
      }
      assign_ = a_new;
      ROS_WARN("[rescue] stage-1 reassign (cost %.2f -> %.2f)", c_old, c_new);
      return;
    }
    // stage-2: symmetric deadlock (re-matching is a no-op) -> break symmetry by giving the
    // WORST dog a reference that routes AROUND its parked teammates (BVC-style detour).
    // Teammate disc D_MIN - r_astar keeps the A* centreline >= D_MIN from a parked dog.
    // No path even after a start nudge = genuinely fenced out -> hold and log (retrying
    // is free after the cooldown; the fleet may have shifted by then).
    const int worst = cand;  // the stalled dog with the largest slot error
    const double mate_r = std::max(0.30, admm::D_MIN - r_astar_);
    std::vector<admm::AStarCircle> circles;
    for (const auto& o : coord_->obstacles())
      circles.push_back({o.pos[0], o.pos[1], o.radius});
    Eigen::Vector2d pw = xnow.at(worst).head<2>(), nearest_mate = pw;
    double dmin_mate = 1e9;
    for (int j : dogs_) {
      if (j == worst) continue;
      const Eigen::Vector2d pj = xnow.at(j).head<2>();
      circles.push_back({pj[0], pj[1], mate_r});
      const double d = (pj - pw).norm();
      if (d < dmin_mate) { dmin_mate = d; nearest_mate = pj; }
    }
    std::vector<admm::AStarRect> rects;
    for (const auto& r : arena_rects_)
      rects.push_back({r.center[0], r.center[1], r.size[0], r.size[1], r_astar_});
    const admm::AStarPlanner local(astar_res_, r_astar_, circles, ax_min_, ax_max_,
                                   ay_min_, ay_max_, /*boundary_margin=*/0.45, rects);
    // plan(), NOT find_reachable_goal: hard-fail -> "fenced out" wait is the PROVEN
    // behavior. Relocation sounds nicer but degenerates when the slot cell rounds into
    // the boundary band (PF2 2026-07-16: every candidate collapsed to the dog's own
    // cell, wp 5->2->1, dragging the dog AWAY from a slot it was 0.7 m from). A fenced
    // dog just holds; retry is free after the cooldown once the fleet has shifted.
    std::vector<Eigen::Vector2d> p = local.plan(pw, goal_[worst]);
    if (p.empty() && dmin_mate < 1e9) {
      // start cell may sit inside a teammate's inflated disc -> nudge start outward
      const Eigen::Vector2d away = (pw - nearest_mate).normalized();
      const Eigen::Vector2d s2 = nearest_mate + away * (mate_r + r_astar_ + astar_res_);
      p = local.plan(s2, goal_[worst]);
      if (!p.empty()) p.insert(p.begin(), pw);
    }
    if (p.empty()) { ROS_WARN("[rescue] fenced out dog%d (no path)", worst); return; }
    path_[worst] = p; yaw_latched_[worst] = false;
    rescued_until_[worst] = now + ros::Duration(2.0 * rescue_cooldown_s_);
    ROS_WARN("[rescue] stage-2 detour dog%d wp=%zu", worst, p.size());
  }

  void tick(const ros::TimerEvent&) {
    if (!ready()) { ROS_WARN_THROTTLE(2.0, "[fleet_pub_cpp] waiting for all 3 obs+gt ..."); return; }
    std::map<int, Eigen::VectorXd> xnow;
    std::map<int, Eigen::Vector2d> P0;
    std::map<int, Eigen::MatrixXd> xdes;
    for (int i : dogs_) {
      Eigen::Vector4d X0; Eigen::Vector2d p0; x0(i, X0, p0);
      xnow[i] = X0; P0[i] = p0;
    }
    rescueTick(xnow, ros::Time::now());
    for (int i : dogs_) {
      // waypoints() WRITES path_, followSpeed() READS it to pick the travel direction.
      // As two arguments of one call their order is unspecified -- on the tick after any
      // path_.erase(i) followSpeed could run first, miss the cache and aim the ACC cone
      // down the straight goal bearing. Sequence them explicitly.
      const Eigen::MatrixX2d wp = waypoints(i, P0[i]);
      xdes[i] = admm::build_reference(P0[i], wp, admm::N, admm::TS,
                                      followSpeed(i, xnow), 0.80);
    }

    std::map<int, Eigen::VectorXd> xi;
    admm::ADMMCoordinator::Hist hist;
    try {
      auto res = coord_->step(xnow, xdes);
      xi = res.first; hist = res.second;
    } catch (const std::exception& e) {
      ROS_WARN_THROTTLE(2.0, "[fleet_pub_cpp] coord.step failed: %s", e.what()); return;
    }

    if (hist_csv_.is_open()) {
      char b[32]; std::snprintf(b, sizeof(b), "%.4f,", obs_[dogs_[0]].time); hist_csv_ << b;
      for (size_t k = 0; k < hist.r_prim.size(); ++k) { std::snprintf(b, sizeof(b), "%.6g", hist.r_prim[k]); hist_csv_ << b << ","; }
      for (size_t k = 0; k < hist.r_dual.size(); ++k) { std::snprintf(b, sizeof(b), "%.6g", hist.r_dual[k]); hist_csv_ << b << ","; }
      for (size_t k = 0; k < hist.r_pos.size(); ++k) { std::snprintf(b, sizeof(b), "%.6g", hist.r_pos[k]); hist_csv_ << b << ","; }
      for (size_t k = 0; k < hist.r_vel.size(); ++k) { std::snprintf(b, sizeof(b), "%.6g", hist.r_vel[k]); hist_csv_ << b << (k + 1 < hist.r_vel.size() ? "," : ""); }
      hist_csv_ << "\n"; hist_csv_.flush();
    }

    for (int i : dogs_)
      if (!xi[i].allFinite()) { ROS_WARN_THROTTLE(1.0, "[fleet_pub_cpp] ADMM xi non-finite -> HOLD"); return; }

    std::map<int, Eigen::VectorXd> wpx, wpy;
    for (int i : dogs_) {
      Eigen::VectorXd px(admm::N), py(admm::N);
      for (int k = 1; k <= admm::N; ++k) { px[k - 1] = xi[i][admm::px_index(k)]; py[k - 1] = xi[i][admm::py_index(k)]; }
      wpx[i] = px; wpy[i] = py;
    }
    if (viz_) publishMarkers(wpx, wpy);

    std::map<int, std::vector<double>> dbg;  // per-dog instrumentation for CSV
    for (int i : dogs_) {
      const auto& s = obs_[i].state.value;
      Eigen::VectorXd xi_mpc = xi[i];
      double dx = s[BASE_PX] - P0[i][0], dy = s[BASE_PY] - P0[i][1];
      for (int k = 1; k <= admm::N; ++k) { xi_mpc[admm::px_index(k)] += dx; xi_mpc[admm::py_index(k)] += dy; }
      std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> others;
      for (int j : dogs_) if (j != i) others.push_back({wpx[j], wpy[j]});
      int k_send = admm::safe_prefix_length(wpx[i], wpy[i], others, d_safe_, send_cap_);
      auto out = adapter_->build_target(xi_mpc, obs_[i].time, s[BASE_YAW], "path", k_send);
      Eigen::MatrixXd states = out.states;  // (ns, 24)

      double d_goal = std::hypot(P0[i][0] - goal_[i][0], P0[i][1] - goal_[i][1]);
      if (!yaw_latched_[i]) {
        if (d_goal < r_latch_) { yaw_latched_[i] = true; yaw_latch_val_[i] = s[BASE_YAW]; }
      } else if (d_goal > r_latch_ + latch_margin_) {
        yaw_latched_[i] = false;
      }
      if (yaw_latched_[i])
        for (int r = 0; r < states.rows(); ++r) states(r, BASE_YAW) = yaw_latch_val_[i];

      pub_[i].publish(toMsg(out.times, states));
      dbg[i] = {gt_[i].pose.pose.position.x, gt_[i].pose.pose.position.y,
                (double)s[BASE_PX], (double)s[BASE_PY], dx, dy,
                (double)s[MOM_LIN_X], (double)s[MOM_LIN_Y], xnow[i][2], xnow[i][3],
                states(0, BASE_PX), states(0, BASE_PY), (double)s[BASE_YAW],
                states(0, BASE_YAW), (double)k_send,
                // world-frame assigned slot: makes per-dog arrival auditable offline
                goal_[i][0], goal_[i][1]};
    }
    logCsv(dbg);
  }

  ocs2_msgs::mpc_target_trajectories toMsg(const Eigen::VectorXd& times, const Eigen::MatrixXd& states) {
    ocs2_msgs::mpc_target_trajectories m;
    int ns = static_cast<int>(times.size());
    for (int j = 0; j < ns; ++j) {
      m.timeTrajectory.push_back(times[j]);
      ocs2_msgs::mpc_state st; st.value.resize(24);
      for (int c = 0; c < 24; ++c) st.value[c] = static_cast<float>(states(j, c));
      m.stateTrajectory.push_back(st);
      ocs2_msgs::mpc_input in; in.value.assign(24, 0.0f);
      m.inputTrajectory.push_back(in);
    }
    return m;
  }

  void logCsv(const std::map<int, std::vector<double>>& dbg) {
    if (!csv_.is_open()) return;
    char b[32]; std::snprintf(b, sizeof(b), "%.4f", obs_[dogs_[0]].time); csv_ << b;
    for (int i : dogs_) {
      for (double v : dbg.at(i)) { std::snprintf(b, sizeof(b), "%.5f", v); csv_ << "," << b; }
    }
    csv_ << "\n"; csv_.flush();
  }

  void publishMarkers(const std::map<int, Eigen::VectorXd>& wpx, const std::map<int, Eigen::VectorXd>& wpy) {
    visualization_msgs::MarkerArray arr;
    int j = 0;
    for (const auto& o : coord_->obstacles()) {
      std::vector<Eigen::Vector3d> pts;
      for (int k = 0; k <= 24; ++k) {
        double a = 2 * M_PI * k / 24.0;
        pts.emplace_back(o.pos[0] + o.radius * std::cos(a), o.pos[1] + o.radius * std::sin(a), 0.05);
      }
      arr.markers.push_back(lineMarker("obstacles", j++, pts, 0.55, 0.55, 0.55, 0.03));
    }
    int w = 0;  // walls: rect obstacle outlines (door arena walls/posts frames)
    for (const auto& r : arena_rects_) {
      const double hx = r.size[0] / 2.0, hy = r.size[1] / 2.0;
      std::vector<Eigen::Vector3d> pts;
      pts.emplace_back(r.center[0] - hx, r.center[1] - hy, 0.05);
      pts.emplace_back(r.center[0] + hx, r.center[1] - hy, 0.05);
      pts.emplace_back(r.center[0] + hx, r.center[1] + hy, 0.05);
      pts.emplace_back(r.center[0] - hx, r.center[1] + hy, 0.05);
      pts.emplace_back(r.center[0] - hx, r.center[1] - hy, 0.05);
      arr.markers.push_back(lineMarker("walls", w++, pts, 0.85, 0.85, 0.85, 0.05));
    }
    static const std::map<int, Eigen::Vector3d> rgb = {
        {1, {0.90, 0.20, 0.20}}, {2, {0.20, 0.80, 0.30}}, {3, {0.20, 0.45, 0.95}}};
    std::map<int, Eigen::Vector2d> pos;
    for (int i : dogs_) pos[i] = Eigen::Vector2d(gt_[i].pose.pose.position.x, gt_[i].pose.pose.position.y);
    for (int i : dogs_) {
      Eigen::Vector3d c = rgb.at(i);
      std::vector<Eigen::Vector3d> roll;
      for (int k = 0; k < wpx.at(i).size(); ++k) roll.emplace_back(wpx.at(i)[k], wpy.at(i)[k], 0.1);
      arr.markers.push_back(lineMarker("rollout", i, roll, c[0], c[1], c[2], 0.05));
      arr.markers.push_back(sphereMarker("dog", i, pos[i], c, 0.22));
      arr.markers.push_back(sphereMarker("slot", i, goal_[i], c, 0.12));
    }
    std::vector<Eigen::Vector3d> ep;
    for (size_t a = 0; a < dogs_.size(); ++a)
      for (size_t b = a + 1; b < dogs_.size(); ++b) {
        ep.emplace_back(pos[dogs_[a]][0], pos[dogs_[a]][1], 0.1);
        ep.emplace_back(pos[dogs_[b]][0], pos[dogs_[b]][1], 0.1);
      }
    auto fm = lineMarker("formation", 0, ep, 1.0, 0.85, 0.1, 0.03);
    fm.type = visualization_msgs::Marker::LINE_LIST;
    arr.markers.push_back(fm);
    marker_pub_.publish(arr);
  }

  visualization_msgs::Marker lineMarker(const std::string& ns, int id,
                                        const std::vector<Eigen::Vector3d>& pts,
                                        double r, double g, double b, double w) {
    visualization_msgs::Marker m;
    m.header.frame_id = viz_frame_; m.lifetime = ros::Duration(3 * ts_);
    m.ns = ns; m.id = id; m.type = visualization_msgs::Marker::LINE_STRIP;
    m.action = visualization_msgs::Marker::ADD; m.scale.x = w; m.pose.orientation.w = 1.0;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
    for (const auto& p : pts) { geometry_msgs::Point q; q.x = p[0]; q.y = p[1]; q.z = p[2]; m.points.push_back(q); }
    return m;
  }

  visualization_msgs::Marker sphereMarker(const std::string& ns, int id, const Eigen::Vector2d& xy,
                                          const Eigen::Vector3d& c, double d) {
    visualization_msgs::Marker m;
    m.header.frame_id = viz_frame_; m.lifetime = ros::Duration(3 * ts_);
    m.ns = ns; m.id = id; m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = m.scale.y = m.scale.z = d; m.pose.orientation.w = 1.0;
    m.pose.position.x = xy[0]; m.pose.position.y = xy[1]; m.pose.position.z = 0.1;
    m.color.r = c[0]; m.color.g = c[1]; m.color.b = c[2]; m.color.a = 1.0;
    return m;
  }

  void killCppTargets() {
    std::vector<std::string> targets;
    for (int i : dogs_) targets.push_back("/dog" + std::to_string(i) + "/legged_robot_target");
    std::vector<bool> killed(targets.size(), false);
    for (int attempt = 0; attempt < 20; ++attempt) {
      std::string live;
      if (FILE* p = popen("rosnode list 2>/dev/null", "r")) {
        char ln[256]; while (fgets(ln, sizeof(ln), p)) live += ln; pclose(p);
      }
      bool all = true;
      for (size_t t = 0; t < targets.size(); ++t) {
        if (killed[t]) continue;
        if (live.find(targets[t]) != std::string::npos) {
          std::string cmd = "rosnode kill " + targets[t] + " >/dev/null 2>&1";
          if (std::system(cmd.c_str()) == 0) killed[t] = true;
        }
        if (!killed[t]) all = false;
      }
      if (all) break;
      ros::Duration(0.5).sleep();
    }
  }

  ros::NodeHandle nh_, pnh_;
  std::vector<int> dogs_;
  std::vector<admm::EdgeKey> edges_;
  int N_; double ts_, v_, com_height_, w_form_;
  std::string formation_name_, arena_name_, viz_frame_, csv_path_;
  std::map<int, Eigen::Vector2d> goal_;
  std::unique_ptr<admm::LaplacianFormation> formation_;
  std::unique_ptr<admm::ADMMCoordinator> coord_;
  std::unique_ptr<admm::MotionAdapter> adapter_;
  std::unique_ptr<admm::AStarPlanner> planner_;
  std::vector<admm::ArenaRect> arena_rects_;
  double last_formation_yaw_ = 0.0, r_latch_, latch_margin_, d_safe_;
  double follow_gain_, follow_desired_, follow_range_, follow_cone_cos_, follow_floor_;  // #1 leader-aware brake
  bool rescue_{true}, have_slots_{false};
  double rescue_stall_s_, rescue_v_eps_, rescue_slot_err_, rescue_hyst_, rescue_cooldown_s_;
  double r_astar_, astar_res_, ax_min_, ax_max_, ay_min_, ay_max_;
  std::vector<Eigen::Vector2d> slots_;
  std::vector<int> assign_;
  std::map<int, ros::Time> stall_since_d_, rescued_until_;
  ros::Time cooldown_until_{0};
  int send_cap_; bool use_astar_, viz_;
  std::map<int, bool> yaw_latched_; std::map<int, double> yaw_latch_val_;
  std::map<int, std::vector<Eigen::Vector2d>> path_;
  std::map<int, ocs2_msgs::mpc_observation> obs_; std::map<int, bool> has_obs_;
  std::map<int, nav_msgs::Odometry> gt_; std::map<int, bool> has_gt_;
  std::map<int, Eigen::Vector2d> v_ema_; std::map<int, bool> has_vema_;
  std::map<int, ros::Publisher> pub_;
  ros::Publisher marker_pub_;
  std::vector<ros::Subscriber> subs_;
  ros::Timer timer_;
  std::ofstream csv_, hist_csv_;
};

const std::vector<std::string> FleetPublisherNode::kDbgCols = {
    "gt_x", "gt_y", "estpx", "estpy", "dx", "dy", "obsvx", "obsvy",
    "x0vx", "x0vy", "tgt1x", "tgt1y", "seed", "cmd", "ksend", "slotx", "sloty"};

int main(int argc, char** argv) {
  ros::init(argc, argv, "ocs2_fleet_publisher");
  FleetPublisherNode node;
  ros::spin();
  return 0;
}
