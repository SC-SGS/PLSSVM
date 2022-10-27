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
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>  // std::fill, std::all_of
#include <chrono>     // std::chrono
#include <iostream>   // std::cout, std::endl
#include <vector>     // std::vector

namespace plssvm::openmp {

csvm::csvm(const target_platform target, parameter params) :
    ::plssvm::csvm{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::cpu) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the OpenMP backend!", target) };
    }
    // the CPU target must be available
#if !defined(PLSSVM_HAS_CPU_TARGET)
    throw backend_exception{ "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!" };
#endif

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

template <typename real_type>
std::pair<std::vector<real_type>, real_type> csvm::solve_system_of_linear_equations_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, const real_type eps, const unsigned long long max_iter) const {
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
    run_device_kernel(params, q, r, alpha, A, QA_cost, real_type{ -1.0 });

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
        std::cout << fmt::format("Done in {}.", iteration_duration) << std::endl;
        average_iteration_time += iteration_duration;
    };

    unsigned long long iter = 0;
    for (; iter < max_iter; ++iter) {
        if (verbose) {
            std::cout << fmt::format("Start Iteration {} (max: {}) with current residuum {} (target: {}). ", iter + 1, max_iter, delta, eps * eps * delta0);
        }
        iteration_start_time = std::chrono::steady_clock::now();

        // Ad = A * d (q = A * d)
        std::fill(Ad.begin(), Ad.end(), real_type{ 0.0 });
        run_device_kernel(params, q, Ad, d, A, QA_cost, real_type{ 1.0 });

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        alpha += alpha_cd * d;

        if (iter % 50 == 49) {
            // (r = b - A * x)
            // r = b
            r = b;
            // r -= A * x
            run_device_kernel(params, q, r, alpha, A, QA_cost, real_type{ -1.0 });
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
        std::cout << fmt::format("Finished after {} iterations with a residuum of {} (target: {}) and an average iteration time of {}.",
                                 std::min(iter + 1, max_iter),
                                 delta,
                                 eps * eps * delta0,
                                 average_iteration_time / std::min(iter + 1, max_iter))
                  << std::endl;
    }

    // calculate bias
    const real_type bias = b_back_value + QA_cost * sum(alpha) - (transposed{ q } * alpha);
    alpha.push_back(-sum(alpha));

    return std::make_pair(std::move(alpha), -bias);
}

template std::pair<std::vector<float>, float> csvm::solve_system_of_linear_equations_impl(const detail::parameter<float> &, const std::vector<std::vector<float>> &, std::vector<float>, const float, const unsigned long long) const;
template std::pair<std::vector<double>, double> csvm::solve_system_of_linear_equations_impl(const detail::parameter<double> &, const std::vector<std::vector<double>> &, std::vector<double>, const double, const unsigned long long) const;

template <typename real_type>
std::vector<real_type> csvm::predict_values_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, const real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const {
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
            case kernel_function_type::rbf: {
                real_type temp{ 0.0 };
                #pragma omp simd reduction(+ : temp)
                for (typename std::vector<std::vector<real_type>>::size_type data_index = 0; data_index < support_vectors.size(); ++data_index) {
                    temp += alpha[data_index] * kernel_function(support_vectors[data_index], predict_points[point_index], params);
                }
                out[point_index] += temp;
            } break;
        }
    }
    return out;
}

template std::vector<float> csvm::predict_values_impl(const detail::parameter<float> &, const std::vector<std::vector<float>> &, const std::vector<float> &, float, std::vector<float> &, const std::vector<std::vector<float>> &) const;
template std::vector<double> csvm::predict_values_impl(const detail::parameter<double> &, const std::vector<std::vector<double>> &, const std::vector<double> &, double, std::vector<double> &, const std::vector<std::vector<double>> &) const;

template <typename real_type>
std::vector<real_type> csvm::generate_q(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &data) const {
    std::vector<real_type> q(data.size() - 1);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            device_kernel_q_linear(q, data);
            break;
        case kernel_function_type::polynomial:
            device_kernel_q_polynomial(q, data, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            device_kernel_q_rbf(q, data, params.gamma.value());
            break;
    }
    return q;
}
template std::vector<float> csvm::generate_q<float>(const detail::parameter<float> &, const std::vector<std::vector<float>> &) const;
template std::vector<double> csvm::generate_q<double>(const detail::parameter<double> &, const std::vector<std::vector<double>> &) const;

template <typename real_type>
std::vector<real_type> csvm::calculate_w(const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha) const {
    const typename std::vector<std::vector<real_type>>::size_type num_data_points = support_vectors.size();
    const typename std::vector<real_type>::size_type num_features = support_vectors.front().size();

    // create w vector and fill with zeros
    std::vector<real_type> w(num_features, real_type{ 0.0 });

    // calculate the w vector
    #pragma omp parallel for default(none) shared(support_vectors, alpha, w) firstprivate(num_features, num_data_points)
    for (typename std::vector<real_type>::size_type feature_index = 0; feature_index < num_features; ++feature_index) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (typename std::vector<std::vector<real_type>>::size_type data_index = 0; data_index < num_data_points; ++data_index) {
            temp = std::fma(alpha[data_index], support_vectors[data_index][feature_index], temp);
        }
        w[feature_index] = temp;
    }
    return w;
}

template std::vector<float> csvm::calculate_w(const std::vector<std::vector<float>> &, const std::vector<float> &) const;
template std::vector<double> csvm::calculate_w(const std::vector<std::vector<double>> &, const std::vector<double> &) const;

template <typename real_type>
void csvm::run_device_kernel(const detail::parameter<real_type> &params, const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type add) const {
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            openmp::device_kernel_linear(q, ret, d, data, QA_cost, 1 / params.cost, add);
            break;
        case kernel_function_type::polynomial:
            openmp::device_kernel_polynomial(q, ret, d, data, QA_cost, 1 / params.cost, add, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            openmp::device_kernel_rbf(q, ret, d, data, QA_cost, 1 / params.cost, add, params.gamma.value());
            break;
    }
}
template void csvm::run_device_kernel(const detail::parameter<float> &, const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, float, float) const;
template void csvm::run_device_kernel(const detail::parameter<double> &, const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, double, double) const;

}  // namespace plssvm::openmp
