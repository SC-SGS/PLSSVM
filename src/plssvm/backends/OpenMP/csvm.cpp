/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/csvm.hpp"

#include "plssvm/backends/OpenMP/exceptions.hpp"              // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/kernel_matrix_assembly.hpp"  // plssvm::openmp::linear_kernel_matrix_assembly, plssvm::openmp::polynomial_kernel_matrix_assembly, plssvm::openmp::rbf_kernel_matrix_assembly
#include "plssvm/backends/OpenMP/q_kernel.hpp"                // plssvm::openmp::device_kernel_q_linear, plssvm::openmp::device_kernel_q_polynomial, plssvm::openmp::device_kernel_q_rbf
#include "plssvm/csvm.hpp"                                    // plssvm::csvm
#include "plssvm/detail/assert.hpp"                           // PLSSVM_ASSERT
#include "plssvm/detail/logger.hpp"                           // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/operators.hpp"                        // various operator overloads for std::vector and scalars
#include "plssvm/detail/performance_tracker.hpp"              // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/kernel_function_types.hpp"                   // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                               // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                        // plssvm::target_platform

#include "fmt/chrono.h"   // directly print std::chrono literals with fmt
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>  // std::fill, std::all_of, std::min
#include <chrono>     // std::chrono::{milliseconds, steady_clock, time_point, duration_cast}
#include <cmath>      // std::fma
#include <utility>    // std::pair, std::make_pair, std::move
#include <vector>     // std::vector

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
                        "\nUsing OpenMP as backend with {} threads.\n\n",
                        plssvm::detail::tracking_entry{ "backend", "num_threads", num_omp_threads });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::openmp }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "target_platform", plssvm::target_platform::cpu }));

    // update the target platform
    target_ = plssvm::target_platform::cpu;
}

template <typename real_type>
std::pair<std::vector<std::vector<real_type>>, std::vector<real_type>> csvm::solve_system_of_linear_equations_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<std::vector<real_type>> B, const real_type eps, const unsigned long long max_iter) const {
    PLSSVM_ASSERT(!A.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!A.front().empty(), "The data points must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(A.cbegin(), A.cend(), [&A](const std::vector<real_type> &data_point) { return data_point.size() == A.front().size(); }), "All data points must have the same number of features!");
    PLSSVM_ASSERT(!B.empty(), "At least one right hand side must be given!");
    PLSSVM_ASSERT(std::all_of(B.cbegin(), B.cend(), [&A](const std::vector<real_type> &rhs) { return A.size() == rhs.size(); }), "The number of data points in the matrix A ({}) and the values in all right hand side vectors must be the same!", A.size());
    PLSSVM_ASSERT(eps > real_type{ 0.0 }, "The stopping criterion in the CG algorithm must be greater than 0.0, but is {}!", eps);
    PLSSVM_ASSERT(max_iter > 0, "The number of CG iterations must be greater than 0!");

    using namespace plssvm::operators;

    // create q_red vector
    const std::vector<real_type> q_red = this->generate_q(params, A);

    // calculate QA_costs
    const real_type QA_cost = kernel_function(A.back(), A.back(), params) + real_type{ 1.0 } / params.cost;

    // update b
    std::vector<real_type> b_back_value(B.size());
    for (std::size_t i = 0; i < B.size(); ++i) {
        b_back_value[i] = B[i].back();
        B[i].pop_back();
        B[i] -= b_back_value[i];
    }

    const typename std::vector<real_type>::size_type dept = B.front().size();

    // assemble explicit kernel matrix

    const std::vector<std::vector<real_type>> explicit_A = this->assemble_kernel_matrix(params, A, q_red, QA_cost);

    // CG

    std::vector<std::vector<real_type>> X(B.size(), std::vector<real_type>(dept, real_type{ 1.0 }));

    // sanity checks
    PLSSVM_ASSERT(dept == explicit_A.size(), "Sizes mismatch!: {} != {}", dept, explicit_A.size());

    std::vector<std::vector<real_type>> R(B);

    // R = B - AX
    #pragma omp parallel for collapse(2)
    for (std::size_t col = 0; col < B.size(); ++col) {
        for (std::size_t row = 0; row < dept; ++row) {
            R[col][row] -= transposed{ explicit_A[row] } * X[col];
        }
    }

    // delta = R.T * R
    std::vector<real_type> delta(B.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < B.size(); ++i) {
        delta[i] = transposed{ R[i] } * R[i];
    }
    const std::vector<real_type> delta0(delta);
    std::vector<std::vector<real_type>> Q(B.size(), std::vector<real_type>(dept));

    std::vector<std::vector<real_type>> D(R);

    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        const std::chrono::time_point iteration_end_time = std::chrono::steady_clock::now();
        const auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n",
                    iteration_duration);
        average_iteration_time += iteration_duration;
    };
    // get the index of the rhs that has the largest residual difference wrt to its target residual
    const auto residual_info = [&]() {
        real_type max_difference{ 0.0 };
        std::size_t idx{ 0 };
        for (std::size_t i = 0; i < delta.size(); ++i) {
            const real_type difference = delta[i] - (eps * eps * delta0[i]);
            if (difference > max_difference) {
                idx = i;
            }
        }
        return idx;
    };
    // get the number of rhs that have already been converged
    const auto num_converged = [&]() {
        std::vector<bool> converged(B.size(), false);
        for (std::size_t i = 0; i < B.size(); ++i) {
            // check if the rhs converged in the current iteration
            converged[i] = delta[i] <= eps * eps * delta0[i];
        }
        return static_cast<std::size_t>(std::count(converged.cbegin(), converged.cend(), true));
    };

    unsigned long long iter = 0;
    for (; iter < max_iter; ++iter) {
        const std::size_t max_residual_difference_idx = residual_info();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with {}/{} converged rhs (max residual {} with target residual {} for rhs {}). ",
                    iter + 1,
                    max_iter,
                    num_converged(),
                    B.size(),
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        iteration_start_time = std::chrono::steady_clock::now();

        // Q = A * D
        #pragma omp parallel for collapse(2)
        for (std::size_t col = 0; col < B.size(); ++col) {
            for (std::size_t row = 0; row < dept; ++row) {
                Q[col][row] = transposed{ explicit_A[row] } * D[col];
            }
        }

        // (alpha = delta_new / (D^T * Q))
        std::vector<real_type> alpha(B.size());
        #pragma omp parallel for
        for (std::size_t i = 0; i < B.size(); ++i) {
            alpha[i] = delta[i] / (transposed{ D[i] } * Q[i]);
        }

        // X = X + alpha * D)
        #pragma omp parallel for
        for (std::size_t i = 0; i < B.size(); ++i) {
            X[i] += alpha[i] * D[i];
        }

        if (iter % 50 == 49) {
            // R = B - A * X
            R = B;
            #pragma omp parallel for collapse(2)
            for (std::size_t i = 0; i < B.size(); ++i) {
                for (std::size_t row = 0; row < dept; ++row) {
                    R[i][row] -= transposed{ explicit_A[row] } * X[i];
                }
            }
        } else {
            // R = R - alpha * Q
            #pragma omp parallel for
            for (std::size_t i = 0; i < B.size(); ++i) {
                R[i] -= alpha[i] * Q[i];
            }
        }

        // delta = R^T * R
        const std::vector<real_type> delta_old = delta;
        #pragma omp parallel for
        for (std::size_t i = 0; i < B.size(); ++i) {
            delta[i] = transposed{ R[i] } * R[i];
        }

        // if we are exact enough stop CG iterations, i.e., if every rhs has converged
        if (num_converged() == B.size()) {
            output_iteration_duration();
            break;
        }

        // beta = delta_new / delta_old
        const std::vector<real_type> beta = delta / delta_old;
        // D = beta * D + R
        #pragma omp parallel for
        for (std::size_t i = 0; i < B.size(); ++i) {
            D[i] = beta[i] * D[i] + R[i];
        }

        output_iteration_duration();
    }
    const std::size_t max_residual_difference_idx = residual_info();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", std::min(iter + 1, max_iter) },
                detail::tracking_entry{ "cg", "max_iterations", max_iter },
                detail::tracking_entry{ "cg", "num_converged_rhs", num_converged() },
                detail::tracking_entry{ "cg", "num_rhs", B.size() },
                delta[max_residual_difference_idx],
                eps * eps * delta0[max_residual_difference_idx],
                max_residual_difference_idx,
                detail::tracking_entry{ "cg", "avg_iteration_time", average_iteration_time / std::min(iter + 1, max_iter) });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "residuals", delta }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "target_residuals", eps * eps * delta0 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "epsilon", eps }));
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n",
                std::min(iter + 1, max_iter));

    // calculate bias
    std::vector<real_type> bias(B.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < B.size(); ++i) {
        bias[i] = -(b_back_value[i] + QA_cost * sum(X[i]) - (transposed{ q_red } * X[i]));
        X[i].push_back(-sum(X[i]));
    }

    return std::make_pair(std::move(X), std::move(bias));
}

template std::pair<std::vector<std::vector<float>>, std::vector<float>> csvm::solve_system_of_linear_equations_impl(const detail::parameter<float> &, const std::vector<std::vector<float>> &, std::vector<std::vector<float>>, const float, const unsigned long long) const;
template std::pair<std::vector<std::vector<double>>, std::vector<double>> csvm::solve_system_of_linear_equations_impl(const detail::parameter<double> &, const std::vector<std::vector<double>> &, std::vector<std::vector<double>>, const double, const unsigned long long) const;

template <typename real_type>
std::vector<std::vector<real_type>> csvm::predict_values_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<std::vector<real_type>> &alpha, const std::vector<real_type> &rho, std::vector<std::vector<real_type>> &w, const std::vector<std::vector<real_type>> &predict_points) const {
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors must not be empty!");
    PLSSVM_ASSERT(!support_vectors.front().empty(), "The support vectors must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(support_vectors.cbegin(), support_vectors.cend(), [&support_vectors](const std::vector<real_type> &data_point) { return data_point.size() == support_vectors.front().size(); }), "All support vectors must have the same number of features!");
    PLSSVM_ASSERT(!alpha.empty(), "The alpha vectors (weights) must not be empty!");
    PLSSVM_ASSERT(!alpha.front().empty(), "The alpha vectors must contain at least one weight!");
    PLSSVM_ASSERT(std::all_of(alpha.cbegin(), alpha.cend(), [&alpha](const std::vector<real_type> &a) { return a.size() == alpha.front().size(); }), "All alpha vectors must have the same number of weights!");
    PLSSVM_ASSERT(support_vectors.size() == alpha.front().size(), "The number of support vectors ({}) and number of weights ({}) must be the same!", support_vectors.size(), alpha.front().size());
    PLSSVM_ASSERT(rho.size() == alpha.size(), "The number of rho values ({}) and the number of weights ({}) must be the same!", rho.size(), alpha.size());
    PLSSVM_ASSERT(w.empty() || support_vectors.front().size() == w.front().size(), "Either w must be empty or contain exactly the same number of values as features are present ({})!", support_vectors.front().size());
    PLSSVM_ASSERT(w.empty() || std::all_of(w.cbegin(), w.cend(), [&w](const std::vector<real_type> &vec) { return vec.size() == w.front().size(); }), "All w vectors must have the same number of values!");
    PLSSVM_ASSERT(w.empty() || alpha.size() == w.size(), "Either w must be empty or contain exactly the same number of vectors ({}) as the alpha vector ({})!", w.size(), alpha.size());
    PLSSVM_ASSERT(!predict_points.empty(), "The data points to predict must not be empty!");
    PLSSVM_ASSERT(!predict_points.front().empty(), "The data points to predict must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(predict_points.cbegin(), predict_points.cend(), [&predict_points](const std::vector<real_type> &data_point) { return data_point.size() == predict_points.front().size(); }), "All data points to predict must have the same number of features!");
    PLSSVM_ASSERT(support_vectors.front().size() == predict_points.front().size(), "The number of features in the support vectors ({}) must be the same as in the data points to predict ({})!", support_vectors.front().size(), predict_points.front().size());

    using namespace plssvm::operators;

    // num_predict_points x num_classes
    std::vector<std::vector<real_type>> out(predict_points.size(), std::vector<real_type>(alpha.size()));

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w.resize(alpha.size());
            #pragma omp parallel for
            for (std::size_t a = 0; a < alpha.size(); ++a) {
                w[a] = calculate_w(support_vectors, alpha[a]);
            }
        }
        // predict the values using the w vector
        #pragma omp parallel for collapse(2)
        for (std::size_t point_index = 0; point_index < predict_points.size(); ++point_index) {
            for (std::size_t a = 0; a < alpha.size(); ++a) {
                out[point_index][a] = transposed{ w[a] } * predict_points[point_index] - rho[a];
            }
        }
    } else {
        // "default" implementation for the other kernel functions

        // fill temporary matrix with all kernel function values
        std::vector<std::vector<real_type>> matr(predict_points.size(), std::vector<real_type>(support_vectors.size()));
        #pragma omp parallel for collapse(2)
        for (std::size_t point_index = 0; point_index < predict_points.size(); ++point_index) {
            for (std::size_t sv_index = 0; sv_index < support_vectors.size(); ++sv_index) {
                matr[point_index][sv_index] = kernel_function(support_vectors[sv_index], predict_points[point_index], params);
            }
        }
        // predict the values using the previously learned weights
        #pragma omp parallel for collapse(2)
        for (std::size_t point_index = 0; point_index < predict_points.size(); ++point_index) {
            for (std::size_t a = 0; a < alpha.size(); ++a) {
                real_type temp{ -rho[a] };
                #pragma omp simd reduction(+ : temp)
                for (std::size_t sv_index = 0; sv_index < support_vectors.size(); ++sv_index) {
                    temp += alpha[a][sv_index] * matr[point_index][sv_index];
                }
                out[point_index][a] = temp;
            }
        }
    }
    return out;
}

template std::vector<std::vector<float>> csvm::predict_values_impl(const detail::parameter<float> &, const std::vector<std::vector<float>> &, const std::vector<std::vector<float>> &, const std::vector<float> &, std::vector<std::vector<float>> &, const std::vector<std::vector<float>> &) const;
template std::vector<std::vector<double>> csvm::predict_values_impl(const detail::parameter<double> &, const std::vector<std::vector<double>> &, const std::vector<std::vector<double>> &, const std::vector<double> &, std::vector<std::vector<double>> &, const std::vector<std::vector<double>> &) const;

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
std::vector<std::vector<real_type>> csvm::assemble_kernel_matrix(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &data, const std::vector<real_type> &q, const real_type QA_cost) const {
    PLSSVM_ASSERT(!data.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!data.front().empty(), "The data points must contain at least one feature!");
    PLSSVM_ASSERT(std::all_of(data.cbegin(), data.cend(), [](const std::vector<real_type> &features) { return !features.empty(); }), "All data point must have exactly the same number of features!");
    PLSSVM_ASSERT(q.size() == data.size() - 1, "The q vector must have one entry less than the number of points in data!");

    const std::chrono::steady_clock::time_point assembly_start_time = std::chrono::steady_clock::now();
    std::vector<std::vector<real_type>> explicit_A(data.size() - 1, std::vector<real_type>(data.size() - 1));
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            openmp::linear_kernel_matrix_assembly(q, explicit_A, data, QA_cost, 1 / params.cost);
            break;
        case kernel_function_type::polynomial:
            openmp::polynomial_kernel_matrix_assembly(q, explicit_A, data, QA_cost, 1 / params.cost, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            openmp::rbf_kernel_matrix_assembly(q, explicit_A, data, QA_cost, 1 / params.cost, params.gamma.value());
            break;
    }
    const std::chrono::steady_clock::time_point assembly_end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Assembled the kernel matrix in {}.\n",
                detail::tracking_entry{ "cg", "kernel_matrix_assembly", std::chrono::duration_cast<std::chrono::milliseconds>(assembly_end_time - assembly_start_time) });

    PLSSVM_ASSERT(explicit_A.size() == q.size(), "The size of the kernel matrix must match the size of the q vector!");
    PLSSVM_ASSERT(std::all_of(explicit_A.cbegin(), explicit_A.cend(), [](const std::vector<real_type> &features) { return !features.empty(); }), "All data point in the kernel matrix must have exactly the same number of values!");
    PLSSVM_ASSERT(explicit_A.size() == explicit_A.front().size(), "The kernel matrix must be quadratic!");

    return explicit_A;
}

template std::vector<std::vector<float>> csvm::assemble_kernel_matrix(const detail::parameter<float> &, const std::vector<std::vector<float>> &, const std::vector<float> &, const float) const;
template std::vector<std::vector<double>> csvm::assemble_kernel_matrix(const detail::parameter<double> &, const std::vector<std::vector<double>> &, const std::vector<double> &, const double) const;

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

}  // namespace plssvm::openmp
