/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

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
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "fmt/chrono.h"   // directly print std::chrono literals with fmt
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>  // std::fill, std::all_of
#include <chrono>     // std::chrono
#include <vector>     // std::vector

namespace plssvm::openmp {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    ::plssvm::csvm<T>{ params } {
    // check if supported target platform has been selected
    if (target_ != target_platform::automatic && target_ != target_platform::cpu) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the OpenMP backend!", target_) };
    } else {
#if !defined(PLSSVM_HAS_CPU_TARGET)
        throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
    }

    if (print_info_) {
        fmt::print("Using OpenMP as backend.\n\n");
    }
}

template <typename T>
auto csvm<T>::generate_q() -> std::vector<real_type> {
    std::vector<real_type> q(data_ptr_->size() - 1);
    switch (kernel_) {
        case kernel_type::linear:
            device_kernel_q_linear(q, *data_ptr_);
            break;
        case kernel_type::polynomial:
            device_kernel_q_poly(q, *data_ptr_, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            device_kernel_q_radial(q, *data_ptr_, gamma_);
            break;
    }
    return q;
}

template <typename T>
void csvm<T>::run_device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type add) {
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
    }
}

template <typename T>
auto csvm<T>::solver_CG(const std::vector<real_type> &b, const std::size_t imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    using namespace plssvm::operators;

    std::vector<real_type> alpha(b.size(), 1.0);
    const typename std::vector<real_type>::size_type dept = b.size();

    // sanity checks
    PLSSVM_ASSERT(dept == num_data_points_ - 1, "Sizes mismatch!: {} != {}", dept, num_data_points_ - 1);

    std::vector<real_type> r(b);

    // r = A + alpha_ (r = b - Ax)
    run_device_kernel(q, r, alpha, *data_ptr_, -1);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<real_type> d(r);

    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        auto iteration_end_time = std::chrono::steady_clock::now();
        auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        fmt::print("Done in {}.\n", iteration_duration);
        average_iteration_time += iteration_duration;
    };

    std::size_t run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}). ", run + 1, imax, delta, eps * eps * delta0);
        }
        iteration_start_time = std::chrono::steady_clock::now();

        // Ad = A * d (q = A * d)
        std::fill(Ad.begin(), Ad.end(), real_type{ 0.0 });
        run_device_kernel(q, Ad, d, *data_ptr_, 1);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        alpha += alpha_cd * d;

        if (run % 50 == 49) {
            // (r = b - A * x)
            // r = b
            r = b;
            // r -= A * x
            run_device_kernel(q, r, alpha, *data_ptr_, -1);
        } else {
            // r -= alpha_cd * Ad (r = r - alpha * q)
            r -= alpha_cd * Ad;
        }

        // (delta = r^T * r)
        const real_type delta_old = delta;
        delta = transposed{ r } * r;
        // if we are exact enough stop CG iterations
        if (delta <= eps * eps * delta0) {
            if (print_info_) {
                output_iteration_duration();
            }
            break;
        }

        // (beta = delta_new / delta_old)
        real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;

        if (print_info_) {
            output_iteration_duration();
        }
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}) and an average iteration time of {}.\n",
                   std::min(run + 1, imax),
                   delta,
                   eps * eps * delta0,
                   average_iteration_time / std::min(run + 1, imax));
    }

    return alpha;
}

template <typename T>
void csvm<T>::update_w() {
    // resize and reset all values to zero
    w_.resize(num_features_);
    std::fill(w_.begin(), w_.end(), real_type{ 0.0 });

    // calculate the w vector
    #pragma omp parallel for
    for (std::size_t feature_index = 0; feature_index < num_features_; ++feature_index) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+: temp)
        for (std::size_t data_index = 0; data_index < num_data_points_; ++data_index) {
            temp += (*alpha_ptr_)[data_index] * (*data_ptr_)[data_index][feature_index];
        }
        w_[feature_index] = temp;
    }
}

template <typename T>
auto csvm<T>::predict(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor

    // return empty vector if there are no points to predict
    if (points.empty()) {
        return std::vector<real_type>{};
    }

    // sanity checks
    if (!std::all_of(points.begin(), points.end(), [&](const std::vector<real_type> &point) { return point.size() == points.front().size(); })) {
        throw exception{ "All points in the prediction point vector must have the same number of features!" };
    } else if (points.front().size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per predict point ({})!", data_ptr_->front().size(), points.front().size()) };
    } else if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas provided for prediction!" };
    }

    PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());  // exception in constructor

    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> out(points.size(), bias_);
    if (kernel_ == kernel_type::linear) {
        // use faster methode in case of the linear kernel function
        if (w_.empty()) {
            update_w();
        }
    }

    #pragma omp parallel for
    for (typename std::vector<std::vector<real_type>>::size_type point_index = 0; point_index < points.size(); ++point_index) {
        if (kernel_ == kernel_type::linear) {
            // use faster methode in case of the linear kernel function
            out[point_index] += transposed{ w_ } * points[point_index];
        } else {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+: temp)
            for (std::size_t data_index = 0; data_index < num_data_points_; ++data_index) {
                temp += (*alpha_ptr_)[data_index] * base_type::kernel_function((*data_ptr_)[data_index], points[point_index]);
            }
            out[point_index] += temp;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Predicted {} data points in {}.\n", points.size(), std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    return out;
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::openmp
