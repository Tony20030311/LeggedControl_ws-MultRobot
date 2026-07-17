#pragma once
// Shared CSC + OSQP plumbing for the ADMM C++ port (implementations in
// src/admm_core/qp_common.cpp).
//
// build_csc_pattern replicates scipy's canonical CSC exactly (column-major,
// row-sorted within a column, duplicates forbidden, explicit zeros KEPT) so the
// data array ordering — and therefore OSQP's internal FP summation order — is
// bit-identical to what osqp-python receives from `sp.csc_matrix((v,(r,c)))`.
// OsqpProblem mirrors osqp-python 0.6.3: same settings surface, same ±OSQP_INFTY
// clipping, same update order (lin_cost -> bounds -> A).
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <osqp.h>

#include "legged_upper_control/admm_constants.hpp"  // re-#undef's OSQP's RHO macro

namespace admm {

struct Trip {
    long long r, c;
    double v;
};

// Canonical CSC pattern; perm[t] = position of triplet t in the data array.
struct CscPattern {
    long long m = 0, n = 0;
    std::vector<c_int> p, i;
    std::vector<std::size_t> perm;
    std::size_t nnz() const { return i.size(); }
};

CscPattern build_csc_pattern(long long m, long long n, const std::vector<Trip>& t);

// Scatter triplet-order values into canonical-CSC data order.
void scatter_values(const CscPattern& pat, const std::vector<double>& trip_vals,
                    std::vector<double>& csc_data);

void clip_bounds(std::vector<double>& l, std::vector<double>& u);

// numpy's pairwise summation (add.reduce), mirrored exactly: sequential below 8
// elements, 8 partial accumulators + tree combine up to 128, recursive halving
// above. np.sum(arr) on >8 elements is NOT sequential — a sequential C++ sum
// diverges from the golden reference in the last bits.
double numpy_pairwise_sum(const double* a, std::size_t n);

// One OSQP problem, osqp-python-equivalent lifecycle: setup once with a fixed
// sparsity pattern, then values-only updates + warm-started solves.
class OsqpProblem {
public:
    OsqpProblem() = default;
    OsqpProblem(const OsqpProblem&) = delete;
    OsqpProblem& operator=(const OsqpProblem&) = delete;
    ~OsqpProblem();

    bool is_setup() const { return work_ != nullptr; }

    // Mirrors osqp.OSQP().setup(P=..., q=..., A=..., l=..., u=..., verbose=False,
    // warm_start=True, eps_abs=1e-6, eps_rel=1e-6, polish=True, max_iter=8000).
    void setup(const CscPattern& P, const std::vector<double>& Px,
               const std::vector<double>& q, const CscPattern& A,
               const std::vector<double>& Ax, std::vector<double> l,
               std::vector<double> u);

    // Mirrors osqp-python update(q=, l=, u=, Ax=) internal order.
    void update(const std::vector<double>* q, std::vector<double> l,
                std::vector<double> u, const std::vector<double>* Ax);
    void update_q_only(const std::vector<double>& q);
    // mirrors edge set_linearization: prob.update(Ax=..., l=..., u=...)
    void update_Alu(const std::vector<double>& Ax, std::vector<double> l,
                    std::vector<double> u);

    struct Result {
        std::string status;
        Eigen::VectorXd x;
    };
    Result solve();

    // Tear the OSQP workspace down so the next solve() re-runs setup() from
    // scratch. Needed because warm_start=1 keeps the internal x/y iterate across
    // solves: once a NaN q poisons that iterate it NEVER clears on later finite
    // solves (verified: edge stays NaN for every subsequent feasible solve), so a
    // full rebuild is the only way back to a clean cold start.
    void reset();

private:
    OSQPWorkspace* work_ = nullptr;
};

}  // namespace admm
