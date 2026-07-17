// ADMM-CBF-DMPC C++ core — pybind11 module.
// Staged port of legged_upper_control/admm (Python stays as the golden reference).
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <osqp.h>

#include "legged_upper_control/admm_constants.hpp"  // #undef's OSQP's RHO macro
#include "legged_upper_control/admm_rti.hpp"
#include "legged_upper_control/admm_node_qp.hpp"
#include "legged_upper_control/admm_edge_qp.hpp"
#include "legged_upper_control/admm_formation.hpp"
#include "legged_upper_control/admm_motion_adapter.hpp"
#include "legged_upper_control/admm_coordinator.hpp"
#include "legged_upper_control/admm_reference.hpp"
#include "legged_upper_control/astar_planner.hpp"
#include "legged_upper_control/fleet_config.hpp"

namespace py = pybind11;

static Eigen::VectorXd roundtrip(const Eigen::VectorXd& x) { return x; }

// n=None (Python default) -> admm::N; otherwise the explicit horizon.
static int resolve_n(const py::object& n) {
    return n.is_none() ? admm::N : n.cast<int>();
}

static std::vector<admm::Obstacle> parse_obstacles(const py::object& obstacles) {
    std::vector<admm::Obstacle> out;
    if (obstacles.is_none()) return out;
    for (const auto& item : obstacles.cast<py::list>()) {
        const py::dict d = item.cast<py::dict>();
        const py::sequence pos = d["pos"].cast<py::sequence>();
        out.push_back({Eigen::Vector2d(pos[0].cast<double>(), pos[1].cast<double>()),
                       d["radius"].cast<double>()});
    }
    return out;
}

static std::vector<admm::Wall> parse_walls(const py::object& walls) {
    std::vector<admm::Wall> out;
    if (walls.is_none()) return out;
    for (const auto& item : walls.cast<py::list>()) {
        const py::dict d = item.cast<py::dict>();
        const py::sequence nrm = d["normal"].cast<py::sequence>();
        const py::sequence pnt = d["point"].cast<py::sequence>();
        const double d_safe =
            d.contains("d_safe") ? d["d_safe"].cast<double>() : 0.4;  // w.get default
        out.push_back({Eigen::Vector2d(nrm[0].cast<double>(), nrm[1].cast<double>()),
                       Eigen::Vector2d(pnt[0].cast<double>(), pnt[1].cast<double>()),
                       d_safe});
    }
    return out;
}

static std::vector<admm::AStarCircle> parse_astar_obstacles(const py::object& obstacles) {
    std::vector<admm::AStarCircle> out;
    if (obstacles.is_none()) return out;
    for (const auto& item : obstacles.cast<py::list>()) {
        const py::dict d = item.cast<py::dict>();
        const py::sequence pos = d["pos"].cast<py::sequence>();
        const double r = d.contains("astar_radius") ? d["astar_radius"].cast<double>()
                                                    : d["radius"].cast<double>();
        out.push_back({pos[0].cast<double>(), pos[1].cast<double>(), r});
    }
    return out;
}

static std::vector<admm::AStarRect> parse_astar_rects(const py::object& rects,
                                                      double robot_radius) {
    std::vector<admm::AStarRect> out;
    if (rects.is_none()) return out;
    for (const auto& item : rects.cast<py::list>()) {
        const py::dict d = item.cast<py::dict>();
        const py::sequence c = d["center"].cast<py::sequence>();
        const py::sequence s = d["size"].cast<py::sequence>();
        const double margin = d.contains("astar_margin")
                                  ? d["astar_margin"].cast<double>()
                                  : robot_radius;
        out.push_back({c[0].cast<double>(), c[1].cast<double>(), s[0].cast<double>(),
                       s[1].cast<double>(), margin});
    }
    return out;
}

// arena -> py dict with EXACTLY the publisher's literal shapes (tuples for points,
// lists for collections, walls/rects keys only where the Python dict has them).
static py::dict arena_to_py(const admm::Arena& a) {
    py::dict d;
    py::list obs;
    for (const auto& o : a.obstacles) {
        py::dict od;
        od["pos"] = py::make_tuple(o.pos(0), o.pos(1));
        od["radius"] = o.radius;
        obs.append(od);
    }
    d["obstacles"] = obs;
    if (!a.walls.empty()) {
        py::list walls;
        for (const auto& w : a.walls) {
            py::dict wd;
            wd["normal"] = py::make_tuple(w.normal(0), w.normal(1));
            wd["point"] = py::make_tuple(w.point(0), w.point(1));
            wd["d_safe"] = w.d_safe;
            walls.append(wd);
        }
        d["walls"] = walls;
    }
    if (!a.rects.empty()) {
        py::list rects;
        for (const auto& r : a.rects) {
            py::dict rd;
            rd["center"] = py::make_tuple(r.center(0), r.center(1));
            rd["size"] = py::make_tuple(r.size(0), r.size(1));
            rects.append(rd);
        }
        d["rects"] = rects;
    }
    py::dict goals;
    for (const auto& kv : a.goals)
        goals[py::cast(kv.first)] = py::make_tuple(kv.second(0), kv.second(1));
    d["goals"] = goals;
    return d;
}

static py::object node_out_to_py(const admm::NodeSubproblem::Out& o) {
    py::dict d;
    d["status"] = o.status;
    d["xi"] = o.xi;
    if (o.finite) {
        d["x_pred"] = o.x_pred;
        d["a_pred"] = o.a_pred;
        d["a0"] = o.a0;
    } else {
        d["x_pred"] = py::none();
        d["a_pred"] = py::none();
        d["a0"] = py::none();
    }
    return d;
}

PYBIND11_MODULE(admm_core_cpp, m) {
    m.doc() = "ADMM-CBF-DMPC C++ core (staged port of legged_upper_control/admm)";
    m.def("roundtrip", &roundtrip, "Eigen<->numpy roundtrip check");
    m.def("osqp_version", []() { return std::string(osqp_version()); },
          "Version of the linked OSQP C library (must equal osqp-python's 0.6.3)");
    m.def("h_obstacle",
          [](const Eigen::VectorXd& p, const Eigen::VectorXd& p_obs, double r_eff) {
              return admm::h_obstacle(Eigen::Vector2d(p(0), p(1)),
                                      Eigen::Vector2d(p_obs(0), p_obs(1)), r_eff);
          },
          py::arg("p"), py::arg("p_obs"), py::arg("r_eff"));
    m.def("h_wall",
          [](const Eigen::VectorXd& p, const Eigen::VectorXd& n_w,
             const Eigen::VectorXd& p_w, double d_safe) {
              return admm::h_wall(Eigen::Vector2d(p(0), p(1)),
                                  Eigen::Vector2d(n_w(0), n_w(1)),
                                  Eigen::Vector2d(p_w(0), p_w(1)), d_safe);
          },
          py::arg("p"), py::arg("n_w"), py::arg("p_w"), py::arg("d_safe"));

    // --- C1: constants (drop-in for `import constants as C` in the gate tests) ---
    auto c = m.def_submodule("constants", "C++ mirror of admm/constants.py");
    c.attr("N") = admm::N;
    c.attr("TS") = admm::TS;
    c.attr("GAMMA1") = admm::GAMMA1;
    c.attr("GAMMA2") = admm::GAMMA2;
    c.attr("N_X") = admm::N_X;
    c.attr("N_U") = admm::N_U;
    c.attr("XI_DIM") = admm::XI_DIM;
    c.attr("MAX_VX") = admm::MAX_VX;
    c.attr("MAX_VY") = admm::MAX_VY;
    c.attr("MAX_AX") = admm::MAX_AX;
    c.attr("MAX_AY") = admm::MAX_AY;
    c.attr("BD_POS_COEF") = admm::BD_POS_COEF;
    c.attr("BD_VEL_COEF") = admm::BD_VEL_COEF;
    c.attr("Ad") = admm::make_Ad();
    c.attr("Bd") = admm::make_Bd();
    c.attr("COEF_HK2") = admm::COEF_HK2;
    c.attr("COEF_HK1") = admm::COEF_HK1;
    c.attr("COEF_HK") = admm::COEF_HK;
    c.attr("HOCBF_COEFS") = admm::hocbf_coefs();
    c.attr("A_TO_P2_COEF") = admm::A_TO_P2_COEF;
    c.attr("CBF_CONSTR_COEF") = admm::CBF_CONSTR_COEF;
    c.attr("A_TO_P1_COEF") = admm::A_TO_P1_COEF;
    c.attr("CBF_CONSTR_COEF_P1") = admm::CBF_CONSTR_COEF_P1;
    c.attr("RHO") = admm::RHO;
    c.attr("SLACK_LAMBDA") = admm::SLACK_LAMBDA;
    c.attr("D_MIN") = admm::D_MIN;
    c.attr("P_ITERS") = admm::P_ITERS;
    c.attr("K_SEND") = admm::K_SEND;
    c.def("xi_dim", &admm::xi_dim, py::arg("n") = admm::N);
    c.def("x_index", &admm::x_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("px_index", &admm::px_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("py_index", &admm::py_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("vx_index", &admm::vx_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("vy_index", &admm::vy_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("a_index", &admm::a_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("ax_index", &admm::ax_index, py::arg("k"), py::arg("n") = admm::N);
    c.def("ay_index", &admm::ay_index, py::arg("k"), py::arg("n") = admm::N);

    // --- C2: RTI linearizer (drop-in for `import rti_linearizer as rti`) ---
    auto r = m.def_submodule("rti", "C++ mirror of admm/rti_linearizer.py");
    r.def("edge_h",
          [](const Eigen::VectorXd& e) {
              return admm::edge_h(Eigen::Vector2d(e(0), e(1)));
          },
          py::arg("e"));
    r.def("three_point", &admm::three_point, py::arg("h0"), py::arg("h1"),
          py::arg("h2"));
    r.def("shift_xi",
          [](const Eigen::VectorXd& xi, const py::object& n) {
              return admm::shift_xi(xi, resolve_n(n));
          },
          py::arg("xi"), py::arg("n") = py::none());
    r.def("realized_hbar",
          [](const Eigen::VectorXd& xi_i, const Eigen::VectorXd& xi_j,
             const Eigen::VectorXd& xnow_i, const Eigen::VectorXd& xnow_j,
             const py::object& n) {
              return admm::realized_hbar(xi_i, xi_j, xnow_i, xnow_j, resolve_n(n));
          },
          py::arg("xi_i"), py::arg("xi_j"), py::arg("xnow_i"), py::arg("xnow_j"),
          py::arg("n") = py::none());
    r.def("linearize_edge",
          [](const Eigen::VectorXd& xibar_i, const Eigen::VectorXd& xibar_j,
             const Eigen::VectorXd& xnow_i, const Eigen::VectorXd& xnow_j,
             const py::object& n) {
              const admm::EdgeLin lin = admm::linearize_edge(
                  xibar_i, xibar_j, xnow_i, xnow_j, resolve_n(n));
              py::dict d;
              d["g"] = lin.g;
              d["u"] = lin.u;
              d["Hbar"] = lin.Hbar;
              d["abar_i"] = lin.abar_i;
              d["abar_j"] = lin.abar_j;
              d["e"] = lin.e;
              d["N"] = lin.n;
              return d;
          },
          py::arg("xibar_i"), py::arg("xibar_j"), py::arg("xnow_i"),
          py::arg("xnow_j"), py::arg("n") = py::none());

    // --- C3: node QP (drop-in for node_subproblem.NodeSubproblem) ---
    py::class_<admm::NodeSubproblem>(m, "NodeSubproblem")
        .def(py::init([](const py::object& obstacles, const py::object& walls,
                         double robot_margin, double q_pos, double q_v, double r_accel,
                         double w_pred, const py::object& n, double rho_consensus,
                         int n_neighbors, double w_form) {
                 return new admm::NodeSubproblem(
                     parse_obstacles(obstacles), parse_walls(walls), robot_margin,
                     q_pos, q_v, r_accel, w_pred, resolve_n(n), rho_consensus,
                     n_neighbors, w_form);
             }),
             py::arg("obstacles") = py::none(), py::arg("walls") = py::none(),
             py::arg("robot_margin") = 0.30, py::arg("q_pos") = 10.0,
             py::arg("q_v") = 1.0, py::arg("r_accel") = 0.5, py::arg("w_pred") = 20.0,
             py::arg("n") = py::none(), py::arg("rho_consensus") = 0.0,
             py::arg("n_neighbors") = 0, py::arg("w_form") = 0.0)
        .def_readonly("N", &admm::NodeSubproblem::N_)
        .def("solve",
             [](admm::NodeSubproblem& self, const Eigen::VectorXd& x_now,
                const Eigen::MatrixXd& x_des, const py::object& xbar,
                const py::object& consensus_target, const py::object& formation_grad) {
                 Eigen::VectorXd xb, ct;
                 Eigen::MatrixX2d fg;
                 const Eigen::VectorXd* pxb = nullptr;
                 const Eigen::VectorXd* pct = nullptr;
                 const Eigen::MatrixX2d* pfg = nullptr;
                 if (!xbar.is_none()) { xb = xbar.cast<Eigen::VectorXd>(); pxb = &xb; }
                 if (!consensus_target.is_none()) {
                     ct = consensus_target.cast<Eigen::VectorXd>();
                     pct = &ct;
                 }
                 if (!formation_grad.is_none()) {
                     fg = formation_grad.cast<Eigen::MatrixX2d>();
                     pfg = &fg;
                 }
                 return node_out_to_py(self.solve(x_now, x_des, pxb, pct, pfg));
             },
             py::arg("x_now"), py::arg("x_des"), py::arg("xbar") = py::none(),
             py::arg("consensus_target") = py::none(),
             py::arg("formation_grad") = py::none())
        .def("_debug_P",
             [](const admm::NodeSubproblem& self) {
                 return py::make_tuple(self.debug_P_pattern().p, self.debug_P_pattern().i,
                                       self.debug_P_data());
             })
        .def("_debug_A_pattern",
             [](const admm::NodeSubproblem& self) {
                 return py::make_tuple(self.debug_A_pattern().p, self.debug_A_pattern().i);
             })
        .def("_debug_pass",
             [](admm::NodeSubproblem& self, const Eigen::VectorXd& x_now,
                const Eigen::MatrixXd& x_des, const Eigen::VectorXd& xbar,
                const py::object& consensus_target, const py::object& formation_grad,
                bool drop_hard) {
                 Eigen::VectorXd ct;
                 Eigen::MatrixX2d fg;
                 const Eigen::VectorXd* pct = nullptr;
                 const Eigen::MatrixX2d* pfg = nullptr;
                 if (!consensus_target.is_none()) {
                     ct = consensus_target.cast<Eigen::VectorXd>();
                     pct = &ct;
                 }
                 if (!formation_grad.is_none()) {
                     fg = formation_grad.cast<Eigen::MatrixX2d>();
                     pfg = &fg;
                 }
                 const auto d = self.debug_pass(x_now, x_des, xbar, pct, pfg, drop_hard);
                 py::dict out;
                 out["q"] = d.q;
                 out["Ax"] = d.Ax;
                 out["lo"] = d.lo;
                 out["up"] = d.up;
                 return out;
             },
             py::arg("x_now"), py::arg("x_des"), py::arg("xbar"),
             py::arg("consensus_target") = py::none(),
             py::arg("formation_grad") = py::none(), py::arg("drop_hard") = false);

    // --- C3: edge QP (drop-in for edge_subproblem.EdgeSubproblem) ---
    py::class_<admm::EdgeSubproblem>(m, "EdgeSubproblem")
        .def(py::init([](const py::object& n, const py::object& rho,
                         const py::object& slack_lambda, int hard_through) {
                 return new admm::EdgeSubproblem(
                     resolve_n(n), rho.is_none() ? admm::RHO : rho.cast<double>(),
                     slack_lambda.is_none() ? admm::SLACK_LAMBDA
                                            : slack_lambda.cast<double>(),
                     hard_through);
             }),
             py::arg("n") = py::none(), py::arg("rho") = py::none(),
             py::arg("slack_lambda") = py::none(), py::arg("hard_through") = 1)
        .def_readonly("N", &admm::EdgeSubproblem::N_)
        .def("set_linearization",
             [](admm::EdgeSubproblem& self, const py::dict& frozen) {
                 self.set_linearization(frozen["g"].cast<Eigen::MatrixX2d>(),
                                        frozen["u"].cast<Eigen::VectorXd>());
             },
             py::arg("frozen"))
        .def("solve",
             [](admm::EdgeSubproblem& self, const Eigen::VectorXd& xi_i,
                const Eigen::VectorXd& xi_j, const Eigen::VectorXd& lam_i,
                const Eigen::VectorXd& lam_j) -> py::object {
                 const admm::EdgeSubproblem::Out o = self.solve(xi_i, xi_j, lam_i, lam_j);
                 if (!o.ok) return py::make_tuple(py::none(), py::none(), py::none());
                 return py::make_tuple(o.z_i, o.z_j, o.s);
             },
             py::arg("xi_i"), py::arg("xi_j"), py::arg("lam_i"), py::arg("lam_j"))
        .def("reset_solver", &admm::EdgeSubproblem::reset_solver)
        .def("_debug_P",
             [](const admm::EdgeSubproblem& self) {
                 return py::make_tuple(self.debug_P_pattern().p, self.debug_P_pattern().i,
                                       self.debug_P_data());
             })
        .def("_debug_A",
             [](const admm::EdgeSubproblem& self) {
                 return py::make_tuple(self.debug_A_pattern().p, self.debug_A_pattern().i,
                                       self.debug_A_data());
             })
        .def("_debug_lu",
             [](const admm::EdgeSubproblem& self) {
                 return py::make_tuple(self.debug_lo(), self.debug_u());
             })
        .def("_debug_q", &admm::EdgeSubproblem::debug_q, py::arg("xi_i"),
             py::arg("xi_j"), py::arg("lam_i"), py::arg("lam_j"));

    // --- C4: LaplacianFormation (drop-in for core.formation.LaplacianFormation) ---
    py::class_<admm::LaplacianFormation>(m, "LaplacianFormation")
        .def(py::init([](const py::dict& formation_configs) {
                 std::map<std::string, std::vector<Eigen::Vector2d>> cfg;
                 for (const auto& kv : formation_configs) {
                     std::vector<Eigen::Vector2d> pts;
                     for (const auto& p : kv.second.cast<py::sequence>()) {
                         const py::sequence s = p.cast<py::sequence>();
                         pts.emplace_back(s[0].cast<double>(), s[1].cast<double>());
                     }
                     cfg[kv.first.cast<std::string>()] = pts;
                 }
                 return new admm::LaplacianFormation(cfg);
             }),
             py::arg("formation_configs"))
        .def("set_formation", &admm::LaplacianFormation::set_formation, py::arg("name"))
        .def_property_readonly("current_formation",
                               [](const admm::LaplacianFormation& self) -> py::object {
                                   if (self.current_formation().empty())
                                       return py::none();
                                   return py::cast(self.current_formation());
                               })
        .def_property_readonly("current_offsets",
                               [](const admm::LaplacianFormation& self) -> py::object {
                                   const auto* off = self.current_offsets();
                                   if (off == nullptr) return py::none();
                                   py::list out;
                                   for (const auto& p : *off) out.append(py::cast(p));
                                   return out;
                               })
        .def("compute",
             [](const admm::LaplacianFormation& self, const py::sequence& positions) {
                 std::vector<Eigen::Vector2d> pos;
                 for (const auto& p : positions) {
                     const Eigen::VectorXd v = p.cast<Eigen::VectorXd>();
                     pos.emplace_back(v(0), v(1));
                 }
                 const auto r = self.compute(pos);
                 py::list grads;
                 for (const auto& g : r.second) grads.append(py::cast(g));
                 return py::make_tuple(r.first, grads);
             },
             py::arg("positions"));

    // --- C4: motion adapter (drop-in for `import motion_adapter as ma`) ---
    auto ma = m.def_submodule("motion_adapter", "C++ mirror of admm/motion_adapter.py");
    ma.attr("OCS2_STATE_DIM") = admm::OCS2_STATE_DIM;
    ma.attr("MOM_LIN_X") = admm::MOM_LIN_X;
    ma.attr("MOM_LIN_Y") = admm::MOM_LIN_Y;
    ma.attr("MOM_LIN_Z") = admm::MOM_LIN_Z;
    ma.attr("MOM_ANG_X") = admm::MOM_ANG_X;
    ma.attr("MOM_ANG_Y") = admm::MOM_ANG_Y;
    ma.attr("MOM_ANG_Z") = admm::MOM_ANG_Z;
    ma.attr("BASE_PX") = admm::BASE_PX;
    ma.attr("BASE_PY") = admm::BASE_PY;
    ma.attr("BASE_Z") = admm::BASE_Z;
    ma.attr("BASE_YAW") = admm::BASE_YAW;
    ma.attr("BASE_PITCH") = admm::BASE_PITCH;
    ma.attr("BASE_ROLL") = admm::BASE_ROLL;
    ma.attr("JOINT_START") = admm::JOINT_START;
    ma.attr("N_JOINTS") = admm::N_JOINTS;
    ma.def("wrap_to_pi", &admm::wrap_to_pi, py::arg("angle"));
    ma.def("pad_ocs2_state", &admm::pad_ocs2_state, py::arg("px"), py::arg("py"),
           py::arg("vx"), py::arg("vy"), py::arg("yaw"), py::arg("com_height"),
           py::arg("default_joints"));
    ma.def("yaw_step", &admm::yaw_step, py::arg("vx"), py::arg("vy"),
           py::arg("prev_yaw"), py::arg("v_freeze"), py::arg("ema_alpha"));
    ma.def("yaw_trajectory", &admm::yaw_trajectory, py::arg("vx_arr"),
           py::arg("vy_arr"), py::arg("seed_yaw"), py::arg("v_freeze"),
           py::arg("ema_alpha"));
    ma.def("yaw_trajectory_path",
           [](const Eigen::VectorXd& px_arr, const Eigen::VectorXd& py_arr,
              double seed_yaw, int lookahead_M, double pos_eps, double ema_alpha,
              const py::object& max_dyaw) {
               return admm::yaw_trajectory_path(
                   px_arr, py_arr, seed_yaw, lookahead_M, pos_eps, ema_alpha,
                   !max_dyaw.is_none(),
                   max_dyaw.is_none() ? 0.0 : max_dyaw.cast<double>());
           },
           py::arg("px_arr"), py::arg("py_arr"), py::arg("seed_yaw"),
           py::arg("lookahead_M"), py::arg("pos_eps"), py::arg("ema_alpha"),
           py::arg("max_dyaw") = py::none());
    ma.def("safe_prefix_length",
           [](const Eigen::VectorXd& px_i, const Eigen::VectorXd& py_i,
              const py::sequence& others_pos, double d_safe, int k_max) {
               std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> others;
               for (const auto& o : others_pos) {
                   const py::sequence pair = o.cast<py::sequence>();
                   others.emplace_back(pair[0].cast<Eigen::VectorXd>(),
                                       pair[1].cast<Eigen::VectorXd>());
               }
               return admm::safe_prefix_length(px_i, py_i, others, d_safe, k_max);
           },
           py::arg("px_i"), py::arg("py_i"), py::arg("others_pos"), py::arg("d_safe"),
           py::arg("k_max"));

    auto target_to_py = [](const admm::MotionAdapter::Target& t) {
        py::dict d;
        d["times"] = t.times;
        d["states"] = t.states;
        d["inputs"] = t.inputs;
        d["next_seed"] = t.next_seed;
        return d;
    };
    py::class_<admm::MotionAdapter>(ma, "MotionAdapter")
        .def(py::init([](double com_height, const Eigen::VectorXd& default_joints,
                         const py::object& n, const py::object& ts, double v_freeze,
                         double ema_alpha, int input_dim, const std::string& yaw_mode,
                         int lookahead_M, double pos_eps, const py::object& k_send,
                         const py::object& max_yaw_rate) {
                 return new admm::MotionAdapter(
                     com_height, default_joints, resolve_n(n),
                     ts.is_none() ? admm::TS : ts.cast<double>(), v_freeze, ema_alpha,
                     input_dim, yaw_mode, lookahead_M, pos_eps,
                     k_send.is_none() ? admm::K_SEND : k_send.cast<int>(),
                     !max_yaw_rate.is_none(),
                     max_yaw_rate.is_none() ? 0.0 : max_yaw_rate.cast<double>());
             }),
             py::arg("com_height"), py::arg("default_joints"), py::arg("n") = py::none(),
             py::arg("ts") = py::none(), py::arg("v_freeze") = 0.05,
             py::arg("ema_alpha") = 0.3, py::arg("input_dim") = 24,
             py::arg("yaw_mode") = "path", py::arg("lookahead_M") = 5,
             py::arg("pos_eps") = 0.02, py::arg("k_send") = py::none(),
             py::arg("max_yaw_rate") = py::none())
        .def_readonly("N", &admm::MotionAdapter::N_)
        .def_readonly("k_send", &admm::MotionAdapter::k_send_)
        .def_property_readonly("yaw_mode", &admm::MotionAdapter::yaw_mode)
        .def_property_readonly("ts", &admm::MotionAdapter::ts)
        .def_property_readonly("v_freeze", &admm::MotionAdapter::v_freeze)
        .def_property_readonly("ema_alpha", &admm::MotionAdapter::ema_alpha)
        .def_property_readonly("lookahead_M", &admm::MotionAdapter::lookahead_M)
        .def_property_readonly("pos_eps", &admm::MotionAdapter::pos_eps)
        .def("build_target",
             [target_to_py](const admm::MotionAdapter& self, const Eigen::VectorXd& xi,
                            double t0, double seed_yaw, const py::object& yaw_mode,
                            const py::object& k_send) {
                 return target_to_py(self.build_target(
                     xi, t0, seed_yaw,
                     yaw_mode.is_none() ? std::string() : yaw_mode.cast<std::string>(),
                     k_send.is_none() ? -1 : k_send.cast<int>()));
             },
             py::arg("xi"), py::arg("t0") = 0.0, py::arg("seed_yaw") = 0.0,
             py::arg("yaw_mode") = py::none(), py::arg("k_send") = py::none())
        .def("adapt",
             [target_to_py](admm::MotionAdapter& self, const Eigen::VectorXd& xi,
                            int dog, double t0) {
                 return target_to_py(self.adapt(xi, dog, t0));
             },
             py::arg("xi"), py::arg("dog"), py::arg("t0") = 0.0);

    // --- C4: coordinator (drop-in for `import admm_coordinator as ac`) ---
    auto ac = m.def_submodule("coordinator", "C++ mirror of admm/admm_coordinator.py");
    ac.def("consensus_target",
           [](const py::dict& z, const py::dict& lam, int i, const py::sequence& nbrs) {
               admm::EdgeVecs zc, lc;
               auto load = [](const py::dict& src, admm::EdgeVecs& dst) {
                   for (const auto& kv : src) {
                       const py::tuple e = kv.first.cast<py::tuple>();
                       const admm::EdgeKey key(e[0].cast<int>(), e[1].cast<int>());
                       for (const auto& kv2 : kv.second.cast<py::dict>())
                           dst[key][kv2.first.cast<int>()] =
                               kv2.second.cast<Eigen::VectorXd>();
                   }
               };
               load(z, zc);
               load(lam, lc);
               std::vector<int> nb;
               for (const auto& j : nbrs) nb.push_back(j.cast<int>());
               return admm::consensus_target(zc, lc, i, nb);
           },
           py::arg("z"), py::arg("lam"), py::arg("i"), py::arg("neighbors"));
    py::class_<admm::ADMMCoordinator>(ac, "ADMMCoordinator")
        .def(py::init([](const py::object& p_iters, const py::object& rho,
                         const py::sequence& dogs, const py::sequence& edges,
                         const py::object& formation, double w_form,
                         const py::object& obstacles, const py::object& walls,
                         int hard_through, bool parallel, double robot_margin) {
                 std::vector<int> dg;
                 for (const auto& d : dogs) dg.push_back(d.cast<int>());
                 std::vector<admm::EdgeKey> eg;
                 for (const auto& e : edges) {
                     const py::sequence s = e.cast<py::sequence>();
                     eg.emplace_back(s[0].cast<int>(), s[1].cast<int>());
                 }
                 const admm::LaplacianFormation* form =
                     formation.is_none()
                         ? nullptr
                         : formation.cast<const admm::LaplacianFormation*>();
                 return new admm::ADMMCoordinator(
                     p_iters.is_none() ? admm::P_ITERS : p_iters.cast<int>(),
                     rho.is_none() ? admm::RHO : rho.cast<double>(), dg, eg, form,
                     w_form, parse_obstacles(obstacles), parse_walls(walls),
                     hard_through, parallel, robot_margin);
             }),
             py::arg("p_iters") = py::none(), py::arg("rho") = py::none(),
             py::arg("dogs") = py::make_tuple(1, 2),
             py::arg("edges") = py::make_tuple(py::make_tuple(1, 2)),
             py::arg("formation") = py::none(), py::arg("w_form") = 0.0,
             py::arg("obstacles") = py::none(), py::arg("walls") = py::none(),
             py::arg("hard_through") = 1, py::arg("parallel") = false,
             py::arg("robot_margin") = 0.30,
             py::keep_alive<1, 6>())  // coordinator keeps the formation alive
        .def_readonly("N", &admm::ADMMCoordinator::N_)
        .def_readonly("cycle", &admm::ADMMCoordinator::cycle_)
        .def_property_readonly("neighbors",
                               [](const admm::ADMMCoordinator& self) {
                                   py::dict d;
                                   for (const auto& kv : self.neighbors())
                                       d[py::cast(kv.first)] = kv.second;
                                   return d;
                               })
        .def_property_readonly("obstacles",
                               [](const admm::ADMMCoordinator& self) {
                                   py::list obs;  // mirrors Python self.obstacles
                                   for (const auto& o : self.obstacles()) {
                                       py::dict od;
                                       od["pos"] = py::make_tuple(o.pos(0), o.pos(1));
                                       od["radius"] = o.radius;
                                       obs.append(od);
                                   }
                                   return obs;
                               })
        .def_property_readonly("prev_z",
                               [](const admm::ADMMCoordinator& self) -> py::object {
                                   if (!self.has_prev()) return py::none();
                                   py::dict d;
                                   for (const auto& kv : self.prev_z()) {
                                       py::dict inner;
                                       for (const auto& kv2 : kv.second)
                                           inner[py::cast(kv2.first)] = kv2.second;
                                       d[py::make_tuple(kv.first.first,
                                                        kv.first.second)] = inner;
                                   }
                                   return d;
                               })
        .def_property_readonly("prev_lam",
                               [](const admm::ADMMCoordinator& self) -> py::object {
                                   if (!self.has_prev()) return py::none();
                                   py::dict d;
                                   for (const auto& kv : self.prev_lam()) {
                                       py::dict inner;
                                       for (const auto& kv2 : kv.second)
                                           inner[py::cast(kv2.first)] = kv2.second;
                                       d[py::make_tuple(kv.first.first,
                                                        kv.first.second)] = inner;
                                   }
                                   return d;
                               })
        .def("step",
             [](admm::ADMMCoordinator& self, const py::dict& xnow,
                const py::dict& xdes) {
                 std::map<int, Eigen::VectorXd> xn;
                 std::map<int, Eigen::MatrixXd> xd;
                 for (const auto& kv : xnow)
                     xn[kv.first.cast<int>()] = kv.second.cast<Eigen::VectorXd>();
                 for (const auto& kv : xdes)
                     xd[kv.first.cast<int>()] = kv.second.cast<Eigen::MatrixXd>();
                 const auto r = self.step(xn, xd);
                 py::dict xi;
                 for (const auto& kv : r.first) xi[py::cast(kv.first)] = kv.second;
                 py::dict hist;
                 hist["r_prim"] = r.second.r_prim;
                 hist["r_dual"] = r.second.r_dual;
                 hist["h2_viol"] = r.second.h2_viol;
                 hist["edge_fail"] = r.second.edge_fail;
                 return py::make_tuple(xi, hist);
             },
             py::arg("xnow"), py::arg("xdes"));

    // ---------------- C6: reference (admm/reference.py) ----------------
    auto ref = m.def_submodule("reference", "C++ mirror of admm/reference.py");
    ref.def("point_at_arclength", &admm::point_at_arclength,
            "Point on the polyline at arc length s (clamped to [0, L])",
            py::arg("wp"), py::arg("cum"), py::arg("s"));
    ref.def("build_reference",
            [](const Eigen::Vector2d& cur_pos, const Eigen::MatrixX2d& waypoints,
               const py::object& n, const py::object& ts, double v_cruise,
               double d_brake) {
                return admm::build_reference(
                    cur_pos, waypoints, resolve_n(n),
                    ts.is_none() ? admm::TS : ts.cast<double>(), v_cruise, d_brake);
            },
            "N position anchors ahead of cur_pos: (n,4) rows [px, py, 0, 0]",
            py::arg("cur_pos"), py::arg("waypoints"), py::arg("n") = py::none(),
            py::arg("ts") = py::none(), py::arg("v_cruise") = 0.30,
            py::arg("d_brake") = 0.80);

    // ---------------- C6: A* planner (core/planning.py, used subset) ----------------
    m.def("py_hypot", &admm::py_hypot,
          "CPython 3.8 math.hypot mirror (scaled compensated sum, NOT libm hypot)",
          py::arg("x"), py::arg("y"));
    py::class_<admm::AStarPlanner>(m, "AStarPlanner")
        .def(py::init([](double resolution, double robot_radius,
                         const py::object& obstacles, double x_min, double x_max,
                         double y_min, double y_max, double boundary_margin,
                         const py::object& rect_obstacles) {
                 return admm::AStarPlanner(
                     resolution, robot_radius, parse_astar_obstacles(obstacles),
                     x_min, x_max, y_min, y_max, boundary_margin,
                     parse_astar_rects(rect_obstacles, robot_radius));
             }),
             py::arg("resolution"), py::arg("robot_radius"), py::arg("obstacles"),
             py::arg("x_min") = 0.0, py::arg("x_max") = 10.0,
             py::arg("y_min") = -5.0, py::arg("y_max") = 5.0,
             py::arg("boundary_margin") = 0.45,
             py::arg("rect_obstacles") = py::none())
        .def("plan",
             [](const admm::AStarPlanner& self, const Eigen::Vector2d& start,
                const Eigen::Vector2d& goal) {
                 py::list out;
                 for (const auto& p : self.plan(start, goal))
                     out.append(py::make_tuple(p(0), p(1)));
                 return out;
             },
             py::arg("start"), py::arg("goal"))
        .def("find_reachable_goal",
             [](const admm::AStarPlanner& self, const Eigen::Vector2d& start,
                const Eigen::Vector2d& goal) {
                 const auto r = self.find_reachable_goal(start, goal);
                 py::list path;
                 for (const auto& p : r.path)
                     path.append(py::make_tuple(p(0), p(1)));
                 py::object cand =
                     r.has_candidate
                         ? py::object(py::make_tuple(r.candidate(0), r.candidate(1)))
                         : py::object(py::none());
                 return py::make_tuple(cand, path);
             },
             py::arg("start"), py::arg("goal"))
        .def("_debug_omap", [](const admm::AStarPlanner& self) {
            Eigen::MatrixXi omap(self.nx(), self.ny());
            for (int i = 0; i < self.nx(); ++i)
                for (int j = 0; j < self.ny(); ++j)
                    omap(i, j) = self.omap()[static_cast<size_t>(i) * self.ny() + j];
            return omap;
        });

    // ---------------- C6: fleet config (ocs2_fleet_publisher.py module level) ----
    {
        py::dict arenas;
        for (const auto& kv : admm::arenas()) arenas[py::cast(kv.first)] = arena_to_py(kv.second);
        m.attr("ARENAS") = arenas;
        py::dict formations;
        for (const auto& kv : admm::formations()) {
            py::list offs;
            for (const auto& o : kv.second) offs.append(py::make_tuple(o(0), o(1)));
            formations[py::cast(kv.first)] = offs;
        }
        m.attr("FORMATIONS") = formations;
        py::dict goals;
        for (const auto& kv : admm::default_goals())
            goals[py::cast(kv.first)] = py::make_tuple(kv.second(0), kv.second(1));
        m.attr("DEFAULT_GOALS") = goals;
    }
    m.def("rot2d", &admm::rot2d, "2D rotation matrix (world frame)", py::arg("yaw"));
    m.def("centroid_slot_targets", &admm::centroid_slot_targets,
          "Per-slot world targets: centroid on goal_c, V facing yaw",
          py::arg("goal_c"), py::arg("offsets"), py::arg("yaw"));
    m.def("min_cost_assignment",
          [](const std::vector<Eigen::Vector2d>& positions,
             const std::vector<Eigen::Vector2d>& slots) {
              return py::tuple(py::cast(admm::min_cost_assignment(positions, slots)));
          },
          "Slot permutation minimising sum of squared distances",
          py::arg("positions"), py::arg("slots"));
}
