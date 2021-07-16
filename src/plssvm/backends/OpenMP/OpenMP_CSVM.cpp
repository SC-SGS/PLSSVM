#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"

#include "plssvm/backends/OpenMP/svm-kernel.hpp"  // plssvm::kernel_type
#include "plssvm/detail/operators.hpp"
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::copy
#include <cassert>    // assert
#include <vector>     // std::vector

namespace plssvm {

template <typename T>
OpenMP_CSVM<T>::OpenMP_CSVM(parameter<T> &params) :
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
void OpenMP_CSVM<T>::run_device_kernel(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &ret, const std::vector<real_type> &d, const int sign) {
    // TODO: implement other kernels
    switch (kernel_) {
        case kernel_type::linear:
            device_kernel_linear(data, ret, d, QA_cost_, 1 / cost_, sign);
            break;
        case kernel_type::polynomial:
            device_kernel_poly(data, ret, d, QA_cost_, 1 / cost_, sign, gamma_, coef0_, degree_);
            break;
        case kernel_type::rbf:
            device_kernel_radial(data, ret, d, QA_cost_, 1 / cost_, sign, gamma_);
            break;
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", static_cast<int>(kernel_)) };
    }
}

template <typename T>
auto OpenMP_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    alpha_.resize(b.size(), 1.0);
    const size_type dept = b.size();
    // TODO: compare with CUDA

    //    real_type *datlast = &data_.back()[0];
    //    const size_type dim = data_.back().size();

    // sanity checks
    assert((dim == num_features_) && "Size mismatch: dim != num_features_");
    assert((dept == num_data_points_ - 1) && "Size mismatch: dept != num_data_points_ - 1");

    // solve: r = b - (A * x)
    std::vector<real_type> r(b);
    std::vector<real_type> ones(b.size(), 1.0);

    run_device_kernel(data_, r, ones, -1);
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

        run_device_kernel(data_, Ad, d, 1);
        // kernel_linear(b, data_, datlast, q.data(), Ad, d.data(), dim, QA_cost_, 1 / cost_, 1);

        const real_type alpha = delta / mult(d.data(), Ad.data(), d.size());
        alpha_ += mult(alpha, d.data(), d.size());

        // r = b - (A * x)
        std::copy(b.begin(), b.end(), r.begin());

        run_device_kernel(data_, r, alpha_, -1);
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

    return alpha_;
}

// explicitly instantiate template class
template class OpenMP_CSVM<float>;
template class OpenMP_CSVM<double>;

}  // namespace plssvm