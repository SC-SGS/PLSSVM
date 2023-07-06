/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/csvm.hpp"

#include "plssvm/backend_types.hpp"                                       // plssvm::backend_type
#include "plssvm/backends/OpenMP/cg_explicit/kernel_matrix_assembly.hpp"  // plssvm::openmp::linear_kernel_matrix_assembly, plssvm::openmp::polynomial_kernel_matrix_assembly, plssvm::openmp::rbf_kernel_matrix_assembly
#include "plssvm/backends/OpenMP/exceptions.hpp"                          // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/q_kernel.hpp"                            // plssvm::openmp::device_kernel_q_linear, plssvm::openmp::device_kernel_q_polynomial, plssvm::openmp::device_kernel_q_rbf
#include "plssvm/constants.hpp"                                           // plssvm::real_type
#include "plssvm/csvm.hpp"                                                // plssvm::csvm
#include "plssvm/detail/assert.hpp"                                       // PLSSVM_ASSERT
#include "plssvm/detail/logger.hpp"                                       // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/operators.hpp"                                    // various operator overloads for std::vector and scalars
#include "plssvm/detail/performance_tracker.hpp"                          // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/kernel_function_types.hpp"                               // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                              // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                                           // plssvm::parameter
#include "plssvm/target_platforms.hpp"                                    // plssvm::target_platform

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

unsigned long long csvm::get_device_memory() const {
    return detail::get_system_memory();
}

//***************************************************//
//                        fit                        //
//***************************************************//

detail::simple_any csvm::setup_data_on_devices(const solver_type solver, const aos_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        return detail::simple_any{ &A };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assemblying the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

detail::simple_any csvm::assemble_kernel_matrix(const solver_type solver, const parameter &params, const detail::simple_any &data, const std::vector<real_type> &q_red, const real_type QA_cost) const {
    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    const aos_matrix<real_type> *data_ptr = data.get<const aos_matrix<real_type> *>();
    PLSSVM_ASSERT(data_ptr != nullptr, "The data_ptr must not be a nullptr!");

    const std::size_t num_rows_reduced = data_ptr->num_rows() - 1;
    PLSSVM_ASSERT(num_rows_reduced > 0, "At least one row must be given!");
    PLSSVM_ASSERT(data_ptr->num_rows() == num_rows_reduced + 1, "The number of rows in the data matrix must be {}, but is {}!", num_rows_reduced + 1, data_ptr->num_rows());

    if (solver == solver_type::cg_explicit) {
        aos_matrix<real_type> explicit_A{ num_rows_reduced, num_rows_reduced };
        switch (params.kernel_type) {
            case kernel_function_type::linear:
                openmp::linear_kernel_matrix_assembly(q_red, explicit_A, *data_ptr, QA_cost, 1 / params.cost);
                break;
            case kernel_function_type::polynomial:
                openmp::polynomial_kernel_matrix_assembly(q_red, explicit_A, *data_ptr, QA_cost, 1 / params.cost, params.degree.value(), params.gamma.value(), params.coef0.value());
                break;
            case kernel_function_type::rbf:
                openmp::rbf_kernel_matrix_assembly(q_red, explicit_A, *data_ptr, QA_cost, 1 / params.cost, params.gamma.value());
                break;
        }

        PLSSVM_ASSERT(explicit_A.num_rows() == num_rows_reduced && explicit_A.num_cols() == num_rows_reduced,
                      "The kernel matrix must be a quadratic matrix with shape {}x{}, but has a {}x{} shape!",
                      num_rows_reduced, num_rows_reduced, explicit_A.num_rows(), explicit_A.num_cols());

        return detail::simple_any{ std::move(explicit_A) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assemblying the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

void csvm::blas_gemm(const solver_type solver, const real_type alpha, const detail::simple_any &A, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) const {
    PLSSVM_ASSERT(!B.empty(), "The B matrix may not be empty!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix may not be empty!");
    PLSSVM_ASSERT(B.num_rows() == C.num_rows(), "The C matrix must have {} rows, but has {}!", B.num_rows(), C.num_rows());
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        const aos_matrix<real_type> &explicit_A = A.get<aos_matrix<real_type>>();
        PLSSVM_ASSERT(!explicit_A.empty(), "The A matrix may not be empty!");
        PLSSVM_ASSERT(explicit_A.num_rows() == C.num_cols(), "The C matrix must have {} columns, but has {}!", explicit_A.num_rows(), C.num_cols());

        const std::size_t num_rhs = B.num_rows();
        const std::size_t num_rows = B.num_cols();

        // ret = explicit_A * other
        #pragma omp parallel for collapse(2) default(none) shared(explicit_A, B, C) firstprivate(num_rhs, num_rows, alpha, beta)
        for (std::size_t rhs = 0; rhs < num_rhs; ++rhs) {
            for (std::size_t row = 0; row < num_rows; ++row) {
                real_type temp{ 0.0 };
                #pragma omp simd reduction(+ : temp)
                for (std::size_t dim = 0; dim < num_rows; ++dim) {
                    temp += explicit_A(row, dim) * B(rhs, dim);
                }
                C(rhs, row) = alpha * temp + beta * C(rhs, row);
            }
        }
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("The GEMM calculation using the {} CG variation is currently not implemented!", solver) };
    }
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

aos_matrix<real_type> csvm::predict_values(const parameter &params, const aos_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, aos_matrix<real_type> &w, const aos_matrix<real_type> &predict_points) const {
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
    aos_matrix<real_type> out{ num_predict_points, num_classes };

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w = aos_matrix<real_type>{ num_classes, num_features };

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

}  // namespace plssvm::openmp
