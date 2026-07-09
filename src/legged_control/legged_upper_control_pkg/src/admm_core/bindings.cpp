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
}
