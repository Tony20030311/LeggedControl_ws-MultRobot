// ADMM-CBF-DMPC C++ core — pybind11 module.
// Staged port of legged_upper_control/admm (Python stays as the golden reference).
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <osqp.h>

#include "legged_upper_control/admm_constants.hpp"  // #undef's OSQP's RHO macro
#include "legged_upper_control/admm_rti.hpp"

namespace py = pybind11;

static Eigen::VectorXd roundtrip(const Eigen::VectorXd& x) { return x; }

// n=None (Python default) -> admm::N; otherwise the explicit horizon.
static int resolve_n(const py::object& n) {
    return n.is_none() ? admm::N : n.cast<int>();
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
    r.def("three_point", &admm::three_point,
          py::arg("h0"), py::arg("h1"), py::arg("h2"));
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
}
