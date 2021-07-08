#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"

#include "plssvm/backends/OpenMP/svm-kernel.hpp"  // plssvm::kernel_type
#include "plssvm/detail/operators.hpp"

#include "fmt/core.h"  // fmt::print

#include <algorithm>  // std::copy
#include <cassert>    // assert
#include <vector>     // std::vector

namespace plssvm {

template <typename T>
OpenMP_CSVM<T>::OpenMP_CSVM(real_type cost, real_type epsilon, kernel_type kernel, real_type degree, real_type gamma, real_type coef0, bool info) :
    CSVM<T>(cost, epsilon, kernel, degree, gamma, coef0, info) {}

template <typename T>
auto OpenMP_CSVM<T>::generate_q() -> std::vector<real_type> {
    std::vector<real_type> q;
    q.reserve(data_.size());
    for (size_type i = 0; i < data_.size() - 1; ++i) {
        q.emplace_back(base_type::kernel_function(data_.back(), data_[i]));
    }
    return q;
}

template <typename T>
auto OpenMP_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    std::vector<real_type> x(b.size(), 1.0);
    const size_type dept = b.size();

    //    real_type *datlast = &data_.back()[0];
    //    const size_type dim = data_.back().size();

    // sanity checks
    assert((dim == num_features_) && "Size mismatch: dim != num_features_");
    assert((dept == num_data_points_ - 1) && "Size mismatch: dept != num_data_points_ - 1");

    // solve: r = b - (A * x)
    std::vector<real_type> r(b);
    std::vector<real_type> ones(b.size(), 1.0);

    // TODO: other kernels
    kernel_linear(data_, r, ones, QA_cost_, 1 / cost_, -1);
    // kernel_linear(b, data_, datlast, q.data(), r, ones.data(), dim, QA_cost_, 1 / cost_, -1);

    std::vector<real_type> d(r);

    // delta = r.T * r
    real_type delta = mult(r.data(), r.data(), r.size());
    const real_type delta0 = delta;

    size_type run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
        }
        // Ad = A * d
        std::vector<real_type> Ad(dept, 0.0);

        // TODO: other kernels
        kernel_linear(data_, Ad, d, QA_cost_, 1 / cost_, 1);
        // kernel_linear(b, data_, datlast, q.data(), Ad, d.data(), dim, QA_cost_, 1 / cost_, 1);

        const real_type alpha = delta / mult(d.data(), Ad.data(), d.size());
        x += mult(alpha, d.data(), d.size());

        // r = b - (A * x)
        std::copy(b.begin(), b.end(), r.begin());

        // TODO: other kernels
        kernel_linear(data_, r, x, QA_cost_, 1 / cost_, -1);
        // kernel_linear(b, data_, datlast, q.data(), r, x.data(), dim, QA_cost_, 1 / cost_, -1);

        delta = mult(r.data(), r.data(), r.size());

        // if we are exact enough stop CG iterations
        if (delta < eps * eps * delta0) {
            break;
        }

        const real_type beta = -mult(r.data(), Ad.data(), b.size()) / mult(d.data(), Ad.data(), d.size());
        add(mult(beta, d.data(), d.size()), r.data(), d.data(), d.size());
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
    }

    return x;
}

// explicitly instantiate template class
template class OpenMP_CSVM<float>;
template class OpenMP_CSVM<double>;

}  // namespace plssvm