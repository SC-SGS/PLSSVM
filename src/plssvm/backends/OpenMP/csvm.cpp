#include "plssvm/backends/OpenMP/csvm.hpp"

#include "plssvm/backends/OpenMP/exceptions.hpp"  // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/q_kernel.hpp"    // plssvm::openmp::device_kernel_q_linear, plssvm::openmp::device_kernel_q_poly, plssvm::openmp::device_kernel_q_radial
#include "plssvm/backends/OpenMP/svm_kernel.hpp"  // plssvm::openmp::device_kernel_linear, plssvm::openmp::device_kernel_poly, plssvm::openmp::device_kernel_radial
#include "plssvm/csvm.hpp"                        // plssvm::csvm
#include "plssvm/detail/assert.hpp"               // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"            // various operator overloads for std::vector and scalars
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::unsupported_kernel_type_exception
#include "plssvm/kernel_types.hpp"                // plssvm::kernel_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/target_platform.hpp"             // plssvm::target_platform

#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::copy
#include <vector>     // std::vector

namespace plssvm::openmp {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    csvm{ params.target, params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
csvm<T>::csvm(const target_platform target, const kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
    ::plssvm::csvm<T>{ target, kernel, degree, gamma, coef0, cost, epsilon, print_info } {
    // check if supported target platform has been selected
    if (target_ != target_platform::automatic && target_ != target_platform::cpu) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the OpenMP backend!", target_) };
    }

    if (print_info_) {
        fmt::print("Using OpenMP as backend.\n\n");
    }
}

template <typename T>
auto csvm<T>::generate_q() -> std::vector<real_type> {
    std::vector<real_type> q(data_.size() - 1);
    switch (kernel_) {
        case kernel_type::linear:
            device_kernel_q_linear(q, data_);
            break;
        case kernel_type::polynomial:
            device_kernel_q_poly(q, data_, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            device_kernel_q_radial(q, data_, gamma_);
            break;
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", static_cast<int>(kernel_)) };
    }
    return q;
}

template <typename T>
void csvm<T>::run_device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const int add) {
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
auto csvm<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    using namespace plssvm::operators;

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
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::openmp
