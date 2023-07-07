/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/csvm.hpp"

#include "plssvm/constants.hpp"                   // plssvm::real_type
#include "plssvm/detail/assert.hpp"               // PLSSVM_ASSERT
#include "plssvm/detail/logger.hpp"               // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/operators.hpp"            // plssvm operator overloads for vectors
#include "plssvm/detail/performance_tracker.hpp"  // PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking_entry
#include "plssvm/detail/simple_any.hpp"           // plssvm::detail::simple_any
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type, plssvm::kernel_function
#include "plssvm/matrix.hpp"                      // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/solver_types.hpp"                // plssvm::solver_type

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::count, std::min
#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <utility>    // std::move
#include <utility>    // std::pair, std::make_pair
#include <vector>     // std::vector

namespace plssvm {

void csvm::sanity_check_parameter() const {
    // kernel: valid kernel function
    if (params_.kernel_type != kernel_function_type::linear && params_.kernel_type != kernel_function_type::polynomial && params_.kernel_type != kernel_function_type::rbf) {
        throw invalid_parameter_exception{ fmt::format("Invalid kernel function {} given!", detail::to_underlying(params_.kernel_type)) };
    }

    // gamma: must be greater than 0 IF explicitly provided, but only in the polynomial and rbf kernel
    if ((params_.kernel_type == kernel_function_type::polynomial || params_.kernel_type == kernel_function_type::rbf) && !params_.gamma.is_default() && params_.gamma.value() <= real_type{ 0.0 }) {
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0.0, but is {}!", params_.gamma) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

aos_matrix<real_type> csvm::conjugate_gradients(const detail::simple_any &A, const aos_matrix<real_type> &B, const real_type eps, const unsigned long long max_cg_iter, const solver_type cg_solver) const {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(!B.empty(), "The right-hand sides may not be empty!");

    const std::size_t num_rows = B.num_cols();
    const std::size_t num_rhs = B.num_rows();

    //
    // perform Conjugate Gradients (CG) algorithm
    //

    aos_matrix<real_type> X{ num_rhs, num_rows, real_type{ 1.0 } };

    // R = B - A * X
    aos_matrix<real_type> R{ B };
    this->blas_gemm(cg_solver, real_type{ -1.0 }, A, X, real_type{ 1.0 }, R);

    // delta = R.T * R
    std::vector<real_type> delta(num_rhs);
    #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, num_rows)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (std::size_t j = 0; j < num_rows; ++j) {
            temp += R(i, j) * R(i, j);
        }
        delta[i] = temp;
    }
    const std::vector<real_type> delta0(delta);

    aos_matrix<real_type> D{ R };

    // TODO: look at functions
    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        const auto iteration_end_time = std::chrono::steady_clock::now();
        const auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n",
                    iteration_duration);
        average_iteration_time += iteration_duration;
    };
    // get the index of the rhs that has the largest residual difference wrt to its target residual
    const auto residual_info = [&]() {
        const real_type max_difference{ 0.0 };
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
    for (; iter < max_cg_iter; ++iter) {
        const std::size_t max_residual_difference_idx = residual_info();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with {}/{} converged rhs (max residual {} with target residual {} for rhs {}). ",
                    iter + 1,
                    max_cg_iter,
                    num_converged(),
                    num_rhs,
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        iteration_start_time = std::chrono::steady_clock::now();

        // Q = A * D
        aos_matrix<real_type> Q{ D.num_rows(), D.num_cols() };
        this->blas_gemm(cg_solver, real_type{ 1.0 }, A, D, real_type{ 0.0 }, Q);

        // alpha = delta_new / (D^T * Q))
        std::vector<real_type> alpha(num_rhs);
        #pragma omp parallel for default(none) shared(D, Q, alpha, delta) firstprivate(num_rhs, num_rows)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < num_rows; ++dim) {
                temp += D(i, dim) * Q(i, dim);
            }
            alpha[i] = delta[i] / temp;
        }

        // X = X + alpha * D
        #pragma omp parallel for collapse(2) default(none) shared(X, alpha, D) firstprivate(num_rhs, num_rows)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < num_rows; ++dim) {
                X(i, dim) += alpha[i] * D(i, dim);
            }
        }

        if (iter % 50 == 49) {
            // explicitly recalculate residual to remove accumulating floating point errors
            // R = B - A * X
            R = B;
            this->blas_gemm(cg_solver, real_type{ -1.0 }, A, X, real_type{ 1.0 }, R);
        } else {
            // R = R - alpha * Q
            #pragma omp parallel for collapse(2) default(none) shared(R, alpha, Q) firstprivate(num_rhs, num_rows)
            for (std::size_t i = 0; i < num_rhs; ++i) {
                for (std::size_t dim = 0; dim < num_rows; ++dim) {
                    R(i, dim) -= alpha[i] * Q(i, dim);
                }
            }
        }

        // delta = R^T * R
        const std::vector<real_type> delta_old = delta;
        #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, num_rows)
        for (std::size_t col = 0; col < num_rhs; ++col) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t row = 0; row < num_rows; ++row) {
                temp += R(col, row) * R(col, row);
            }
            delta[col] = temp;
        }

        // if we are exact enough stop CG iterations, i.e., if every rhs has converged
        if (num_converged() == num_rhs) {
            output_iteration_duration();
            break;
        }

        // beta = delta_new / delta_old
        const std::vector<real_type> beta = delta / delta_old;
        // D = beta * D + R
        #pragma omp parallel for collapse(2) default(none) shared(D, beta, R) firstprivate(num_rhs, num_rows)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < num_rows; ++dim) {
                D(i, dim) = beta[i] * D(i, dim) + R(i, dim);
            }
        }

        output_iteration_duration();
    }
    const std::size_t max_residual_difference_idx = residual_info();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", std::min(iter + 1, max_cg_iter) },
                detail::tracking_entry{ "cg", "max_iterations", max_cg_iter },
                detail::tracking_entry{ "cg", "num_converged_rhs", num_converged() },
                detail::tracking_entry{ "cg", "num_rhs", num_rhs },
                delta[max_residual_difference_idx],
                eps * eps * delta0[max_residual_difference_idx],
                max_residual_difference_idx,
                detail::tracking_entry{ "cg", "avg_iteration_time", average_iteration_time / std::min(iter + 1, max_cg_iter) });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "residuals", delta }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "target_residuals", eps * eps * delta0 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "epsilon", eps }));
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n",
                std::min(iter + 1, max_cg_iter));

    return X;
}

std::pair<std::vector<real_type>, real_type> csvm::perform_dimensional_reduction(const parameter &params, const aos_matrix<real_type> &A) const {
    const std::chrono::steady_clock::time_point dimension_reduction_start_time = std::chrono::steady_clock::now();

    const std::size_t num_rows_reduced = A.num_rows() - 1;

    // create q_red vector and calculate QA_costs
    std::vector<real_type> q_red(num_rows_reduced);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            #pragma omp parallel for default(none) shared(q_red, A) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::linear>(A, i, A, num_rows_reduced);
            }
            break;
        case kernel_function_type::polynomial:
            #pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::polynomial>(A, i, A, num_rows_reduced, params.degree.value(), params.gamma.value(), params.coef0.value());
            }
            break;
        case kernel_function_type::rbf:
            #pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::rbf>(A, i, A, num_rows_reduced, params.gamma.value());
            }
            break;
    }
    const real_type QA_cost = kernel_function(A, num_rows_reduced, A, num_rows_reduced, params) + real_type{ 1.0 } / params.cost;
    const std::chrono::steady_clock::time_point dimension_reduction_end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Performed dimensional reduction in {}.\n",
                detail::tracking_entry{ "cg", "dimensional_reduction", std::chrono::duration_cast<std::chrono::milliseconds>(dimension_reduction_end_time - dimension_reduction_start_time) });

    return std::make_pair(std::move(q_red), QA_cost);
}

}  // namespace plssvm