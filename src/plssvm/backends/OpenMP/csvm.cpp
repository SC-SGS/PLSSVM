/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/csvm.hpp"

#include "plssvm/backends/OpenMP/exceptions.hpp"  // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/q_kernel.hpp"    // plssvm::openmp::device_kernel_q_linear, plssvm::openmp::device_kernel_q_polynomial, plssvm::openmp::device_kernel_q_rbf
#include "plssvm/backends/OpenMP/svm_kernel.hpp"  // plssvm::openmp::device_kernel_linear, plssvm::openmp::device_kernel_polynomial, plssvm::openmp::device_kernel_rbf
#include "plssvm/csvm.hpp"                        // plssvm::csvm
#include "plssvm/detail/assert.hpp"               // PLSSVM_ASSERT
#include "plssvm/detail/logger.hpp"               // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/operators.hpp"            // various operator overloads for std::vector and scalars
#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::tracking_entry
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "fmt/chrono.h"                           // directly print std::chrono literals with fmt
#include "fmt/core.h"                             // fmt::format
#include "fmt/ostream.h"                          // can use fmt using operator<< overloads

#include <algorithm>                              // std::fill, std::all_of, std::min
#include <chrono>                                 // std::chrono::{milliseconds, steady_clock, time_point, duration_cast}
#include <cmath>                                  // std::fma
#include <iostream>                               // std::cout, std::endl
#include <utility>                                // std::pair, std::make_pair, std::move
#include <vector>                                 // std::vector

namespace plssvm::openmp {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

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

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing OpenMP as backend with {} threads.\n\n", plssvm::detail::tracking_entry{ "backend", "num_threads", num_omp_threads });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::openmp }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "target_platform", plssvm::target_platform::cpu }));

    // update the target platform
    target_ = plssvm::target_platform::cpu;
}

template <typename real_type>
std::pair<std::vector<real_type>, real_type> csvm::solve_system_of_linear_equations_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, const real_type eps, const unsigned long long max_iter) const {
    PLSSVM_ASSERT(!A.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!A.front().empty(), "The data points must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(A.cbegin(), A.cend(), [&A](const std::vector<real_type> &data_point) { return data_point.size() == A.front().size(); }), "All data points must have the same number of features!");
    PLSSVM_ASSERT(A.size() == b.size(), "The number of data points in the matrix A ({}) and the values in the right hand side vector ({}) must be the same!", A.size(), b.size());
    PLSSVM_ASSERT(eps > real_type{ 0.0 }, "The stopping criterion in the CG algorithm must be greater than 0.0, but is {}!", eps);
    PLSSVM_ASSERT(max_iter > 0, "The number of CG iterations must be greater than 0!");

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
        const std::chrono::time_point iteration_end_time = std::chrono::steady_clock::now();
        const auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n", iteration_duration);
        average_iteration_time += iteration_duration;
    };

    unsigned long long iter = 0;
    for (; iter < max_iter; ++iter) {
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with current residuum {} (target: {}). ", iter + 1, max_iter, delta, eps * eps * delta0);
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
            output_iteration_duration();
            break;
        }

        // (beta = delta_new / delta_old)
        const real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;

        output_iteration_duration();
    }
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with a residuum of {} (target: {}) and an average iteration time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", std::min(iter + 1, max_iter) },
                detail::tracking_entry{ "cg", "max_iterations", max_iter },
                detail::tracking_entry{ "cg", "residuum", delta },
                detail::tracking_entry{ "cg", "target_residuum", eps * eps * delta0 },
                detail::tracking_entry{ "cg", "avg_iteration_time", average_iteration_time / std::min(iter + 1, max_iter) });
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n", std::min(iter + 1, max_iter));

    // calculate bias
    const real_type bias = b_back_value + QA_cost * sum(alpha) - (transposed{ q } * alpha);
    alpha.push_back(-sum(alpha));

    return std::make_pair(std::move(alpha), -bias);
}

template std::pair<std::vector<float>, float> csvm::solve_system_of_linear_equations_impl(const detail::parameter<float> &, const std::vector<std::vector<float>> &, std::vector<float>, const float, const unsigned long long) const;
template std::pair<std::vector<double>, double> csvm::solve_system_of_linear_equations_impl(const detail::parameter<double> &, const std::vector<std::vector<double>> &, std::vector<double>, const double, const unsigned long long) const;

template <typename real_type>
std::vector<real_type> csvm::predict_values_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, const real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const {
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors must not be empty!");
    PLSSVM_ASSERT(!support_vectors.front().empty(), "The support vectors must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(support_vectors.cbegin(), support_vectors.cend(), [&support_vectors](const std::vector<real_type> &data_point) { return data_point.size() == support_vectors.front().size(); }), "All support vectors must have the same number of features!");
    PLSSVM_ASSERT(support_vectors.size() == alpha.size(), "The number of support vectors ({}) and number of weights ({}) must be the same!", support_vectors.size(), alpha.size());
    PLSSVM_ASSERT(w.empty() || support_vectors.front().size() == w.size(), "Either w must be empty or contain exactly the same number of values ({}) as features are present ({})!", w.size(), support_vectors.front().size());
    PLSSVM_ASSERT(!predict_points.empty(), "The data points to predict must not be empty!");
    PLSSVM_ASSERT(!predict_points.front().empty(), "The data points to predict must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(predict_points.cbegin(), predict_points.cend(), [&predict_points](const std::vector<real_type> &data_point) { return data_point.size() == predict_points.front().size(); }), "All data points to predict must have the same number of features!");
    PLSSVM_ASSERT(support_vectors.front().size() == predict_points.front().size(), "The number of features in the support vectors ({}) must be the same as in the data points to predict ({})!", support_vectors.front().size(), predict_points.front().size());

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
    PLSSVM_ASSERT(!data.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!data.front().empty(), "The data points must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(data.cbegin(), data.cend(), [](const std::vector<real_type> &features) { return !features.empty(); }), "All data point must have exactly the same number of features!");

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
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors may not be empty!");
    PLSSVM_ASSERT(!support_vectors.front().empty(), "Each support vector must at least contain one feature!");
    PLSSVM_ASSERT(std::all_of(support_vectors.cbegin(), support_vectors.cend(), [](const std::vector<real_type> &features) { return !features.empty(); }), "All support vectors must have exactly the same number of features!");
    PLSSVM_ASSERT(!alpha.empty(), "The alpha array may not be empty!");
    PLSSVM_ASSERT(support_vectors.size() == alpha.size(), "The number of support vectors ({}) and weights ({}) must match!", support_vectors.size(), alpha.size());

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
    PLSSVM_ASSERT(!q.empty(), "The q array may not be empty!");
    PLSSVM_ASSERT(!ret.empty(), "The ret array may not be empty!");
    PLSSVM_ASSERT(!d.empty(), "The d array may not be empty!");
    PLSSVM_ASSERT(!data.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!data.front().empty(), "The data points must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(data.cbegin(), data.cend(), [](const std::vector<real_type> &features) { return !features.empty(); }), "All data point must have exactly the same number of features!");
    PLSSVM_ASSERT(add == real_type{ -1.0 } || add == real_type{ 1.0 }, "add must either by -1.0 or 1.0, but is {}!", add);

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
