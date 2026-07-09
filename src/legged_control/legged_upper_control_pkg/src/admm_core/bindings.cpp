// ADMM-CBF-DMPC C++ core — pybind11 module.
// Staged port of legged_upper_control/admm (Python stays as the golden reference).
// C0 skeleton: prove the toolchain — Eigen<->numpy roundtrip + linked OSQP version.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <osqp.h>

namespace py = pybind11;

static Eigen::VectorXd roundtrip(const Eigen::VectorXd& x) { return x; }

PYBIND11_MODULE(admm_core_cpp, m) {
    m.doc() = "ADMM-CBF-DMPC C++ core (staged port of legged_upper_control/admm)";
    m.def("roundtrip", &roundtrip, "Eigen<->numpy roundtrip check");
    m.def("osqp_version", []() { return std::string(osqp_version()); },
          "Version of the linked OSQP C library (must equal osqp-python's 0.6.3)");
}
