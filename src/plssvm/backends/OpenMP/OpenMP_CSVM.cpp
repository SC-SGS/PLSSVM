#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"

#include "plssvm/backends/OpenMP/svm_kernel.hpp"  // plssvm::kernel_type
#include "plssvm/detail/assert.hpp"               // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"            // various operator overloads for std::vector and scalars
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::unsupported_kernel_type_exception
#include "plssvm/parameter.hpp"                   // plssvm::parameter

#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::copy
#include <vector>     // std::vector

namespace plssvm {

template <typename T>
OpenMP_CSVM<T>::OpenMP_CSVM(const parameter<T> &params) :
    OpenMP_CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
OpenMP_CSVM<T>::OpenMP_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info) :
    CSVM<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {
    if (print_info_) {
        fmt::print("Using OpenMP as backend.\n\n");
    }
}

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
void OpenMP_CSVM<T>::run_device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const int add) {
    switch (kernel_) {
        case kernel_type::linear:
            openmp::device_kernel_linear(q, ret, d, data, QA_cost_, 1 / cost_, add);
            break;
        case kernel_type::polynomial:
            openmp::device_kernel_poly(q, ret, d, data, QA_cost_, 1 / cost_, add, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            openmp::device_kernel_radial(q, ret, d, data, QA_cost_, 1 / cost_, add, gamma_);
            break;
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", static_cast<int>(kernel_)) };
    }
}

template <typename T>
auto OpenMP_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    alpha_.resize(b.size(), 1.0);
    const size_type dept = b.size();

    // sanity checks
    PLSSVM_ASSERT(dept == num_data_points_ - 1, "Sizes mismatch!: {} != {}", dept, num_data_points_ - 1);

    std::vector<real_type> r(b);

    // solve: r = b - (A * alpha_)
    run_device_kernel(q, r, alpha_, data_, -1);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<real_type> d(r);

    size_type run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
        }
        // Ad = A * d
        std::fill(Ad.begin(), Ad.end(), 0.0);
        run_device_kernel(q, Ad, d, data_, 1);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        alpha_ += alpha_cd * d;

        // (r = b - A * x)
        // r = b
        r = b;
        // r -= A * x
        run_device_kernel(q, r, alpha_, data_, -1);

        // (delta = r^T * r)
        const real_type delta_old = delta;
        delta = transposed{ r } * r;
        // if we are exact enough stop CG iterations
        if (delta <= eps * eps * delta0) {
            break;
        }

        // (beta = delta_new / delta_old)
        real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
    }

    return alpha_;
}

// explicitly instantiate template class
template class OpenMP_CSVM<float>;
template class OpenMP_CSVM<double>;

}  // namespace plssvm
