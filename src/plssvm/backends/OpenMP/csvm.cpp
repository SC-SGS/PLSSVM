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
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "fmt/chrono.h"   // directly print std::chrono literals with fmt
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>  // std::fill, std::all_of
#include <iostream>   // std::cout, std::endl
#include <chrono>     // std::chrono
#include <vector>     // std::vector

namespace plssvm::openmp {

template <typename T>
csvm<T>::csvm(const target_platform target, parameter<real_type> params) : ::plssvm::csvm<T>{ std::move(params) } {
    this->init(target);
}

template <typename T>
void csvm<T>::init(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::cpu) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the OpenMP backend!", target) };
    } else {
#if !defined(PLSSVM_HAS_CPU_TARGET)
        throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
    }

    // get the number of used OpenMP threads
    int num_omp_threads = 0;
    #pragma omp parallel default(none) shared(num_omp_threads)
    {
        #pragma omp master
        num_omp_threads = omp_get_num_threads();
    }

    if (verbose) {
        std::cout << fmt::format("Using OpenMP as backend with {} threads.\n\n", num_omp_threads) << std::endl;
    }
}

template <typename T>
auto csvm<T>::generate_q(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &data) const -> std::vector<real_type> {
    std::vector<real_type> q(data.size() - 1);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            device_kernel_q_linear(q, data);
            break;
        case kernel_function_type::polynomial:
            device_kernel_q_poly(q, data, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            device_kernel_q_radial(q, data, params.gamma.value());
            break;
    }
    return q;
}

template <typename T>
void csvm<T>::run_device_kernel(const parameter<real_type> &params, const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type add) const {
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            openmp::device_kernel_linear(q, ret, d, data, QA_cost, 1 / params.cost, add);
            break;
        case kernel_function_type::polynomial:
            openmp::device_kernel_poly(q, ret, d, data, QA_cost, 1 / params.cost, add, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            openmp::device_kernel_radial(q, ret, d, data, QA_cost, 1 / params.cost, add, params.gamma.value());
            break;
    }
}

template <typename T>
auto csvm<T>::solve_system_of_linear_equations(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, const real_type eps, const size_type max_iter) const -> std::pair<std::vector<real_type>, real_type> {
    using namespace plssvm::operators;

    // create q vector
    const std::vector<real_type> q = this->generate_q(params, A);

    // calculate QA_costs
    const real_type QA_cost = kernel_function(A.back(), A.back(), params) + real_type{ 1.0 } / params.cost;

    // update b
    const real_type b_back_value = b.back();
    b.pop_back();
    b -= b_back_value;

    // CG

    std::vector<real_type> alpha(b.size(), 1.0);
    const typename std::vector<real_type>::size_type dept = b.size();

    // sanity checks
    PLSSVM_ASSERT(dept == A.size() - 1, "Sizes mismatch!: {} != {}", dept, A.size() - 1);

    std::vector<real_type> r(b);

    // r = A + alpha_ (r = b - Ax)
    run_device_kernel(params, q, r, alpha, A, QA_cost, -1);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<real_type> d(r);

    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        std::chrono::time_point iteration_end_time = std::chrono::steady_clock::now();
        auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        fmt::print("Done in {}.\n", iteration_duration);
        average_iteration_time += iteration_duration;
    };

    size_type iter = 0;
    for (; iter < max_iter; ++iter) {
        if (verbose) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}). ", iter + 1, max_iter, delta, eps * eps * delta0);
        }
        iteration_start_time = std::chrono::steady_clock::now();

        // Ad = A * d (q = A * d)
        std::fill(Ad.begin(), Ad.end(), real_type{ 0.0 });
        run_device_kernel(params, q, Ad, d, A, QA_cost, 1);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        alpha += alpha_cd * d;

        if (iter % 50 == 49) {
            // (r = b - A * x)
            // r = b
            r = b;
            // r -= A * x
            run_device_kernel(params, q, r, alpha, A, QA_cost, -1);
        } else {
            // r -= alpha_cd * Ad (r = r - alpha * q)
            r -= alpha_cd * Ad;
        }

        // (delta = r^T * r)
        const real_type delta_old = delta;
        delta = transposed{ r } * r;
        // if we are exact enough stop CG iterations
        if (delta <= eps * eps * delta0) {
            if (verbose) {
                output_iteration_duration();
            }
            break;
        }

        // (beta = delta_new / delta_old)
        const real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;

        if (verbose) {
            output_iteration_duration();
        }
    }
    if (verbose) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}) and an average iteration time of {}.\n",
                   std::min(iter + 1, max_iter),
                   delta,
                   eps * eps * delta0,
                   average_iteration_time / std::min(iter + 1, max_iter));
    }

    // calculate bias
    const real_type bias = b_back_value + QA_cost * sum(alpha) - (transposed{ q } * alpha);
    alpha.push_back(-sum(alpha));

    return std::make_pair(std::move(alpha), -bias);
}


template <typename T>
auto csvm<T>::calculate_w(const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &alpha) const -> std::vector<real_type> {
    const size_type num_data_points = A.size();
    const size_type num_features = A.front().size();

    // create w vector and fill with zeros
    std::vector<real_type> w(num_features, real_type{ 0.0 });

    // calculate the w vector
    #pragma omp parallel for default(none) shared(A, alpha, w) firstprivate(num_features, num_data_points)
    for (size_type feature_index = 0; feature_index < num_features; ++feature_index) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+: temp)
        for (size_type data_index = 0; data_index < num_data_points; ++data_index) {
            temp += alpha[data_index] * A[data_index][feature_index];
        }
        w[feature_index] = temp;
    }
    return w;
}

template <typename T>
auto csvm<T>::predict_values(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha,
                             const real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const -> std::vector<real_type> {
    using namespace plssvm::operators;

    std::vector<real_type> out(predict_points.size(), -rho);

    // use faster methode in case of the linear kernel function
    if (params.kernel_type == kernel_function_type::linear && w.empty()) {
        w = calculate_w(support_vectors, alpha);
    }

    #pragma omp parallel for default(none) shared(predict_points, support_vectors, alpha, w, params, out)
    for (typename std::vector<std::vector<real_type>>::size_type point_index = 0; point_index < predict_points.size(); ++point_index) {
        switch (params.kernel_type) {
            case kernel_function_type::linear:
                out[point_index] += transposed{ w } * predict_points[point_index];
                break;
            case kernel_function_type::polynomial:
            case kernel_function_type::rbf:
                {
                    real_type temp{ 0.0 };
                    #pragma omp simd reduction(+: temp)
                    for (size_type data_index = 0; data_index < support_vectors.size(); ++data_index) {
                        temp += alpha[data_index] * kernel_function(support_vectors[data_index], predict_points[point_index], params);
                    }
                    out[point_index] += temp;
                }
                break;
        }
    }
    return out;
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::openmp
