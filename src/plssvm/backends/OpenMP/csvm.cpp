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
#include "plssvm/detail/matrix.hpp"                           // plssvm::detail::aos_matrix
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
std::pair<detail::aos_matrix<real_type>, std::vector<real_type>> csvm::solve_system_of_linear_equations_impl(const detail::parameter<real_type> &params,
                                                                                                             const detail::aos_matrix<real_type> &A,
                                                                                                             detail::aos_matrix<real_type> B_in,
                                                                                                             const real_type eps,
                                                                                                             const unsigned long long max_iter) const {
    PLSSVM_ASSERT(!A.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!B_in.empty(), "At least one right hand side must be given!");
    PLSSVM_ASSERT(A.num_rows() == B_in.num_cols(), "The number of rows in A ({}) and B({}) must be the same!", A.num_rows(), B_in.num_cols());
    PLSSVM_ASSERT(eps > real_type{ 0.0 }, "The stopping criterion in the CG algorithm must be greater than 0.0, but is {}!", eps);
    PLSSVM_ASSERT(max_iter > 0, "The number of CG iterations must be greater than 0!");

    using namespace plssvm::operators;

    const std::size_t dept = A.num_rows() - 1;
    const std::size_t num_rhs = B_in.num_rows();

    // create q_red vector
    const std::vector<real_type> q_red = this->generate_q(params, A);

    // calculate QA_costs
    const real_type QA_cost = kernel_function(A, dept, A, dept, params) + real_type{ 1.0 } / params.cost;

    // update b
    std::vector<real_type> b_back_value(num_rhs);
    detail::aos_matrix<real_type> B{ num_rhs, dept };
    #pragma omp parallel for default(none) shared(B_in, B, b_back_value) firstprivate(dept, num_rhs)
    for (std::size_t row = 0; row < num_rhs; ++row) {
        b_back_value[row] = B_in(row, dept);
        for (std::size_t col = 0; col < dept; ++col) {
            B(row, col) = B_in(row, col) - b_back_value[row];
        }
    }

    // assemble explicit kernel matrix

    const detail::aos_matrix<real_type> explicit_A = this->assemble_kernel_matrix(params, A, q_red, QA_cost);
    PLSSVM_ASSERT(dept * dept == explicit_A.num_entries(), "Sizes mismatch!: {} != {}", dept * dept, explicit_A.num_entries());

    // CG

    detail::aos_matrix<real_type> X{ num_rhs, dept, real_type{ 1.0 } };

    detail::aos_matrix<real_type> R{ B };

    // R = B - A * X
    #pragma omp parallel for collapse(2) default(none) shared(explicit_A, R, X) firstprivate(num_rhs, dept)
    for (std::size_t rhs = 0; rhs < num_rhs; ++rhs) {
        for (std::size_t row = 0; row < dept; ++row) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < dept; ++dim) {
                temp += explicit_A(row, dim) * X(rhs, dim);
            }
            R(rhs, row) -= temp;
        }
    }

    // delta = R.T * R
    std::vector<real_type> delta(num_rhs);
    #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, dept)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (std::size_t j = 0; j < dept; ++j) {
            temp += R(i, j) * R(i, j);
        }
        delta[i] = temp;
    }
    const std::vector<real_type> delta0(delta);
    detail::aos_matrix<real_type> Q{ num_rhs, dept };

    detail::aos_matrix<real_type> D{ R };

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
        std::vector<bool> converged(delta.size(), false);
        for (std::size_t i = 0; i < delta.size(); ++i) {
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
                    num_rhs,
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        iteration_start_time = std::chrono::steady_clock::now();

        // Q = A * D
        #pragma omp parallel for collapse(2) default(none) shared(Q, explicit_A, D) firstprivate(num_rhs, dept)
        for (std::size_t rhs = 0; rhs < num_rhs; ++rhs) {
            for (std::size_t row = 0; row < dept; ++row) {
                real_type temp{ 0.0 };
                #pragma omp simd reduction(+ : temp)
                for (std::size_t dim = 0; dim < dept; ++dim) {
                    temp += explicit_A(row, dim) * D(rhs, dim);
                }
                Q(rhs, row) = temp;
            }
        }

        // (alpha = delta_new / (D^T * Q))
        std::vector<real_type> alpha(num_rhs);
        #pragma omp parallel for default(none) shared(D, Q, alpha, delta) firstprivate(num_rhs, dept)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < dept; ++dim) {
                temp += D(i, dim) * Q(i, dim);
            }
            alpha[i] = delta[i] / temp;
        }

        // X = X + alpha * D)
        #pragma omp parallel for collapse(2) default(none) shared(X, alpha, D) firstprivate(num_rhs, dept)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < dept; ++dim) {
                X(i, dim) += alpha[i] * D(i, dim);
            }
        }

        if (iter % 50 == 49) {
            // R = B - A * X
            R = B;
            #pragma omp parallel for collapse(2) default(none) shared(explicit_A, R, X) firstprivate(num_rhs, dept)
            for (std::size_t rhs = 0; rhs < num_rhs; ++rhs) {
                for (std::size_t row = 0; row < dept; ++row) {
                    real_type temp{ 0.0 };
                    #pragma omp simd reduction(+ : temp)
                    for (std::size_t dim = 0; dim < dept; ++dim) {
                        temp += explicit_A(row, dim) * X(rhs, dim);
                    }
                    R(rhs, row) -= temp;
                }
            }
        } else {
            // R = R - alpha * Q
            #pragma omp parallel for collapse(2) default(none) shared(R, alpha, Q) firstprivate(num_rhs, dept)
            for (std::size_t i = 0; i < num_rhs; ++i) {
                for (std::size_t dim = 0; dim < dept; ++dim) {
                    R(i, dim) -= alpha[i] * Q(i, dim);
                }
            }
        }

        // delta = R^T * R
        const std::vector<real_type> delta_old = delta;
        #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, dept)
        for (std::size_t col = 0; col < num_rhs; ++col) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t row = 0; row < dept; ++row) {
                temp += R(col, row) * R(col, row);
            }
            delta[col] = temp;
        }

        // if we are exact enough stop CG iterations, i.e., if every rhs has converged
        if (num_converged() == num_rhs) {
            output_iteration_duration();
            break;
        }

        // (beta = delta_new / delta_old)
        const std::vector<real_type> beta = delta / delta_old;
        // D = beta * D + R
        #pragma omp parallel for collapse(2) default(none) shared(D, beta, R) firstprivate(num_rhs, dept)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < dept; ++dim) {
                D(i, dim) = beta[i] * D(i, dim) + R(i, dim);
            }
        }

        output_iteration_duration();
    }
    const std::size_t max_residual_difference_idx = residual_info();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", std::min(iter + 1, max_iter) },
                detail::tracking_entry{ "cg", "max_iterations", max_iter },
                detail::tracking_entry{ "cg", "num_converged_rhs", num_converged() },
                detail::tracking_entry{ "cg", "num_rhs", num_rhs },
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
    detail::aos_matrix<real_type> X_ret{ num_rhs, A.num_rows() };
    std::vector<real_type> bias(num_rhs);
    #pragma omp parallel for default(none) shared(X, q_red, X_ret, bias, b_back_value) firstprivate(num_rhs, dept, QA_cost)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp_sum{ 0.0 };
        real_type temp_dot{ 0.0 };
        #pragma omp simd reduction(+ : temp_sum) reduction(+ : temp_dot)
        for (std::size_t dim = 0; dim < dept; ++dim) {
            temp_sum += X(i, dim);
            temp_dot += q_red[dim] * X(i, dim);

            X_ret(i, dim) = X(i, dim);
        }
        bias[i] = -(b_back_value[i] + QA_cost * temp_sum - temp_dot);
        X_ret(i, dept) = -temp_sum;
    }

    return std::make_pair(std::move(X_ret), std::move(bias));
}

template std::pair<detail::aos_matrix<float>, std::vector<float>> csvm::solve_system_of_linear_equations_impl(const detail::parameter<float> &, const detail::aos_matrix<float> &, detail::aos_matrix<float>, const float, const unsigned long long) const;
template std::pair<detail::aos_matrix<double>, std::vector<double>> csvm::solve_system_of_linear_equations_impl(const detail::parameter<double> &, const detail::aos_matrix<double> &, detail::aos_matrix<double>, const double, const unsigned long long) const;

template <typename real_type>
detail::aos_matrix<real_type> csvm::predict_values_impl(const detail::parameter<real_type> &params, const detail::aos_matrix<real_type> &support_vectors, const detail::aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, detail::aos_matrix<real_type> &w, const detail::aos_matrix<real_type> &predict_points) const {
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors must not be empty!");
    PLSSVM_ASSERT(!alpha.empty(), "The alpha vectors (weights) must not be empty!");
    PLSSVM_ASSERT(support_vectors.num_rows() == alpha.num_cols(), "The number of support vectors ({}) and number of weights ({}) must be the same!", support_vectors.num_rows(), alpha.num_cols());
    PLSSVM_ASSERT(rho.size() == alpha.num_rows(), "The number of rho values ({}) and the number of weights ({}) must be the same!", rho.size(), alpha.num_rows());
    PLSSVM_ASSERT(w.empty() || support_vectors.num_cols() == w.num_cols(), "Either w must be empty or contain exactly the same number of values as features are present ({})!", support_vectors.num_cols());
    PLSSVM_ASSERT(w.empty() || alpha.num_rows() == w.num_rows(), "Either w must be empty or contain exactly the same number of vectors ({}) as the alpha vector ({})!", w.num_rows(), alpha.num_rows());
    PLSSVM_ASSERT(!predict_points.empty(), "The data points to predict must not be empty!");
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "The number of features in the support vectors ({}) must be the same as in the data points to predict ({})!", support_vectors.num_cols(), predict_points.num_cols());

    using namespace plssvm::operators;

    // defined sizes
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    // num_predict_points x num_classes
    detail::aos_matrix<real_type> out{ num_predict_points, num_classes };

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w = detail::aos_matrix<real_type>{ num_classes, num_features };

            #pragma omp parallel for collapse(2) default(none) shared(w, support_vectors, alpha) firstprivate(num_classes, num_features, num_support_vectors)
            for (std::size_t a = 0; a < num_classes; ++a) {
                for (std::size_t dim = 0; dim < num_features; ++dim) {
                    real_type temp{ 0.0 };
                    #pragma omp simd reduction(+ : temp)
                    for (std::size_t idx = 0; idx < num_support_vectors; ++idx) {
                        temp = std::fma(alpha(a, idx), support_vectors(idx, dim), temp);
                    }
                    w(a, dim) = temp;
                }
            }
        }
        // predict the values using the w vector
        #pragma omp parallel for collapse(2) default(none) shared(out, w, rho, alpha, predict_points) firstprivate(num_classes, num_features, num_predict_points)
        for (std::size_t point_index = 0; point_index < num_predict_points; ++point_index) {
            for (std::size_t a = 0; a < num_classes; ++a) {
                real_type temp{ 0.0 };
                #pragma omp simd reduction(+ : temp)
                for (std::size_t dim = 0; dim < num_features; ++dim) {
                    temp += w(a, dim) * predict_points(point_index, dim);
                }
                out(point_index, a) = temp - rho[a];
            }
        }
    } else {
        // "default" implementation for the other kernel functions
        #pragma omp parallel for default(none) shared(alpha, support_vectors, predict_points, rho, params, out) firstprivate(num_predict_points, num_classes, num_support_vectors)
        for (std::size_t point_index = 0; point_index < num_predict_points; ++point_index) {
            for (std::size_t a = 0; a < num_classes; ++a) {
                out(point_index, a) -= rho[a];
            }
            for (std::size_t sv_index = 0; sv_index < num_support_vectors; ++sv_index) {
                const real_type kernel_func = kernel_function(support_vectors, sv_index, predict_points, point_index, params);
                for (std::size_t a = 0; a < num_classes; ++a) {
                    out(point_index, a) += alpha(a, sv_index) * kernel_func;
                }
            }
        }
    }
    return out;
}

template detail::aos_matrix<float> csvm::predict_values_impl(const detail::parameter<float> &, const detail::aos_matrix<float> &, const detail::aos_matrix<float> &, const std::vector<float> &, detail::aos_matrix<float> &, const detail::aos_matrix<float> &) const;
template detail::aos_matrix<double> csvm::predict_values_impl(const detail::parameter<double> &, const detail::aos_matrix<double> &, const detail::aos_matrix<double> &, const std::vector<double> &, detail::aos_matrix<double> &, const detail::aos_matrix<double> &) const;

template <typename real_type>
std::vector<real_type> csvm::generate_q(const detail::parameter<real_type> &params, const detail::aos_matrix<real_type> &data) const {
    PLSSVM_ASSERT(!data.empty(), "The data must not be empty!");

    std::vector<real_type> q(data.num_rows() - 1);
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

template std::vector<float> csvm::generate_q<float>(const detail::parameter<float> &, const detail::aos_matrix<float> &) const;
template std::vector<double> csvm::generate_q<double>(const detail::parameter<double> &, const detail::aos_matrix<double> &) const;

template <typename real_type>
detail::aos_matrix<real_type> csvm::assemble_kernel_matrix(const detail::parameter<real_type> &params, const detail::aos_matrix<real_type> &data, const std::vector<real_type> &q, const real_type QA_cost) const {
    PLSSVM_ASSERT(!data.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "The q vector must have one entry less than the number of points in data!");

    const std::chrono::steady_clock::time_point assembly_start_time = std::chrono::steady_clock::now();
    detail::aos_matrix<real_type> explicit_A{ data.num_rows() - 1, data.num_rows() - 1 };
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

    PLSSVM_ASSERT(explicit_A.num_rows() == q.size(), "The size of the kernel matrix must match the size of the q vector!");
    PLSSVM_ASSERT(explicit_A.num_rows() == explicit_A.num_cols(), "The kernel matrix must be quadratic!");

    return explicit_A;
}

template detail::aos_matrix<float> csvm::assemble_kernel_matrix(const detail::parameter<float> &, const detail::aos_matrix<float> &, const std::vector<float> &, const float) const;
template detail::aos_matrix<double> csvm::assemble_kernel_matrix(const detail::parameter<double> &, const detail::aos_matrix<double> &, const std::vector<double> &, const double) const;

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
