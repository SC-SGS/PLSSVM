/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                   // plssvm::backend_type
#include "plssvm/backends/stdpar/detail/utility.hpp"                                  // plssvm::stdpar::detail::{get_stdpar_implementation}
#include "plssvm/backends/stdpar/exceptions.hpp"                                      // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/implementation_types.hpp"                            // plssvm::stdpar::implementation_type
#include "plssvm/backends/stdpar/kernel/cg_explicit/blas.hpp"                         // plssvm::stdpar::detail::device_kernel_symm
#include "plssvm/backends/stdpar/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::stdpar::detail::device_kernel_assembly
#include "plssvm/backends/stdpar/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::stdpar::detail::device_kernel_assembly_symm
#include "plssvm/backends/stdpar/kernel/predict_kernel.hpp"                           // plssvm::stdpar::detail::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                       // plssvm::real_type
#include "plssvm/csvm.hpp"                                                            // plssvm::csvm
#include "plssvm/detail/assert.hpp"                                                   // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"                                        // plssvm::detail::{data_distribution, triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/logging.hpp"                                                  // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                              // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"                                            // plssvm::detail::{move_only_any, move_only_any_cast}
#include "plssvm/detail/performance_tracker.hpp"                                      // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/utility.hpp"                                                  // plssvm::detail::get_system_memory
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                                          // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/parameter.hpp"                                                       // plssvm::parameter
#include "plssvm/shape.hpp"                                                           // plssvm::shape
#include "plssvm/solver_types.hpp"                                                    // plssvm::solver_type
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                                // plssvm::verbosity_level

#include "fmt/core.h"  // fmt::format

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    #include "sycl/sycl.hpp"
#endif

#include <cmath>    // std::fma
#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple
#include <utility>  // std::pair, std::make_pair, std::move
#include <vector>   // std::vector

namespace plssvm::stdpar {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } { }

csvm::csvm(const target_platform target, parameter params) :
    ::plssvm::csvm{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check whether the requested target platform has been enabled
    switch (target) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
    }

    // TODO: correct target platform selection
    if (target == target_platform::automatic) {
        target_ = determine_default_target_platform();
    } else {
        target_ = target;
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing stdpar ({}) as backend.\n\n",
                        plssvm::detail::tracking_entry{ "dependencies", "stdpar_implementation", detail::get_stdpar_implementation() });

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    // AdaptiveCpp's stdpar per default uses the sycl default device
    ::sycl::device default_device{};
    // TODO: check whether the requested target_platform equals the default_selector
    // if not -> throw excpetion with export ACPP_VISIBILITY_MASK="omp"
#endif

    // print found stdpar devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} stdpar device(s) for the target platform {}:\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", this->num_available_devices() },  // TODO: more than one device?
                        plssvm::detail::tracking_entry{ "backend", "target_platform", target_ });

    // TODO: get used device names and types?
    // std::vector<std::string> device_names;
    // device_names.reserve(devices_.size());
    // for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
    //     const std::string device_name = devices_[device].impl->sycl_queue.get_device().template get_info<::sycl::info::device::name>();
    //     plssvm::detail::log(verbosity_level::full,
    //                         "  [{}, {}]\n",
    //                         device,
    //                         device_name);
    //     device_names.emplace_back(device_name);
    // }
    // PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "device", device_names }));
    // plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
    //                     "\n");
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    // TODO: use correct values
    return { ::plssvm::detail::get_system_memory() };
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    // TODO: use correct values
    return this->get_device_memory();
}

implementation_type csvm::get_implementation_type() const noexcept {
#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    return implementation_type::adaptivecpp;
#endif
}

//***************************************************//
//                        fit                        //
//***************************************************//

std::vector<::plssvm::detail::move_only_any> csvm::assemble_kernel_matrix(const solver_type solver, const parameter &params, const soa_matrix<real_type> &A, const std::vector<real_type> &q_red, const real_type QA_cost) const {
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices must not be empty!");
    PLSSVM_ASSERT(A.is_padded(), "The matrix to setup on the devices must be padded!");
    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector must not be empty!");
    PLSSVM_ASSERT(q_red.size() == A.num_rows() - 1, "The q_red size ({}) mismatches the number of data points after dimensional reduction ({})!", q_red.size(), A.num_rows() - 1);

    std::vector<::plssvm::detail::move_only_any> kernel_matrices_parts(this->num_available_devices());
    const real_type cost = real_type{ 1.0 } / params.cost;

    switch (solver) {
        case solver_type::automatic:
            // unreachable
            break;
        case solver_type::cg_explicit:
            {
                const plssvm::detail::triangular_data_distribution dist{ A.num_rows() - 1, this->num_available_devices() };
                std::vector<real_type> kernel_matrix(dist.calculate_explicit_kernel_matrix_num_entries_padded(0));  // only explicitly store the upper triangular matrix
                switch (params.kernel_type) {
                    case kernel_function_type::linear:
                        detail::device_kernel_assembly<kernel_function_type::linear>(q_red, kernel_matrix, A, QA_cost, cost);
                        break;
                    case kernel_function_type::polynomial:
                        detail::device_kernel_assembly<kernel_function_type::polynomial>(q_red, kernel_matrix, A, QA_cost, cost, params.degree, std::get<real_type>(params.gamma), params.coef0);
                        break;
                    case kernel_function_type::rbf:
                        detail::device_kernel_assembly<kernel_function_type::rbf>(q_red, kernel_matrix, A, QA_cost, cost, std::get<real_type>(params.gamma));
                        break;
                    case kernel_function_type::sigmoid:
                        detail::device_kernel_assembly<kernel_function_type::sigmoid>(q_red, kernel_matrix, A, QA_cost, cost, std::get<real_type>(params.gamma), params.coef0);
                        break;
                    case kernel_function_type::laplacian:
                        detail::device_kernel_assembly<kernel_function_type::laplacian>(q_red, kernel_matrix, A, QA_cost, cost, std::get<real_type>(params.gamma));
                        break;
                    case kernel_function_type::chi_squared:
                        detail::device_kernel_assembly<kernel_function_type::chi_squared>(q_red, kernel_matrix, A, QA_cost, cost, std::get<real_type>(params.gamma));
                        break;
                }

                kernel_matrices_parts[0] = ::plssvm::detail::move_only_any{ std::move(kernel_matrix) };
            }
            break;
        case solver_type::cg_implicit:
            {
                // simply return data since in implicit we don't assembly the kernel matrix here!
                kernel_matrices_parts[0] = ::plssvm::detail::move_only_any{ std::make_tuple(std::move(A), params, std::move(q_red), QA_cost) };
            }
            break;
    }

    return kernel_matrices_parts;
}

void csvm::blas_level_3(const solver_type solver, const real_type alpha, const std::vector<::plssvm::detail::move_only_any> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const {
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");
    PLSSVM_ASSERT(A.size() == 1, "Not enough kernel matrix parts ({}) for the available number of devices (1)!", A.size());
    PLSSVM_ASSERT(!B.empty(), "The B matrix must not be empty!");
    PLSSVM_ASSERT(B.is_padded(), "The B matrix must be padded!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix must not be empty!");
    PLSSVM_ASSERT(C.is_padded(), "The C matrix must be padded!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The B ({}) and C ({}) matrices must have the same shape!", B.shape(), C.shape());
    PLSSVM_ASSERT(B.padding() == C.padding(), "The B ({}) and C ({}) matrices must have the same padding!", B.padding(), C.padding());

    switch (solver) {
        case solver_type::automatic:
            // unreachable
            break;
        case solver_type::cg_explicit:
            {
                const std::size_t num_rhs = B.shape().x;
                const std::size_t num_rows = B.shape().y;

                const auto &explicit_A = ::plssvm::detail::move_only_any_cast<const std::vector<real_type> &>(A.front());
                PLSSVM_ASSERT(!explicit_A.empty(), "The A matrix must not be empty!");

                detail::device_kernel_symm(num_rows, num_rhs, alpha, explicit_A, B, beta, C);
            }
            break;
        case solver_type::cg_implicit:
            {
                const auto &[matr_A, params, q_red, QA_cost] = ::plssvm::detail::move_only_any_cast<const std::tuple<soa_matrix<real_type>, parameter, std::vector<real_type>, real_type> &>(A.front());
                PLSSVM_ASSERT(!matr_A.empty(), "The A matrix must not be empty!");
                PLSSVM_ASSERT(!q_red.empty(), "The q_red vector must not be empty!");
                const real_type cost = real_type{ 1.0 } / params.cost;

                switch (params.kernel_type) {
                    case kernel_function_type::linear:
                        detail::device_kernel_assembly_symm<kernel_function_type::linear>(alpha, q_red, matr_A, QA_cost, cost, B, beta, C);
                        break;
                    case kernel_function_type::polynomial:
                        detail::device_kernel_assembly_symm<kernel_function_type::polynomial>(alpha, q_red, matr_A, QA_cost, cost, B, beta, C, params.degree, std::get<real_type>(params.gamma), params.coef0);
                        break;
                    case kernel_function_type::rbf:
                        detail::device_kernel_assembly_symm<kernel_function_type::rbf>(alpha, q_red, matr_A, QA_cost, cost, B, beta, C, std::get<real_type>(params.gamma));
                        break;
                    case kernel_function_type::sigmoid:
                        detail::device_kernel_assembly_symm<kernel_function_type::sigmoid>(alpha, q_red, matr_A, QA_cost, cost, B, beta, C, std::get<real_type>(params.gamma), params.coef0);
                        break;
                    case kernel_function_type::laplacian:
                        detail::device_kernel_assembly_symm<kernel_function_type::laplacian>(alpha, q_red, matr_A, QA_cost, cost, B, beta, C, std::get<real_type>(params.gamma));
                        break;
                    case kernel_function_type::chi_squared:
                        detail::device_kernel_assembly_symm<kernel_function_type::chi_squared>(alpha, q_red, matr_A, QA_cost, cost, B, beta, C, std::get<real_type>(params.gamma));
                        break;
                }
            }
            break;
    }
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

aos_matrix<real_type> csvm::predict_values(const parameter &params,
                                           const soa_matrix<real_type> &support_vectors,
                                           const aos_matrix<real_type> &alpha,
                                           const std::vector<real_type> &rho,
                                           soa_matrix<real_type> &w,
                                           const soa_matrix<real_type> &predict_points) const {
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors must not be empty!");
    PLSSVM_ASSERT(support_vectors.is_padded(), "The support vectors must be padded!");
    PLSSVM_ASSERT(!alpha.empty(), "The alpha vectors (weights) must not be empty!");
    PLSSVM_ASSERT(alpha.is_padded(), "The alpha vectors (weights) must be padded!");
    PLSSVM_ASSERT(support_vectors.num_rows() == alpha.num_cols(), "The number of support vectors ({}) and number of weights ({}) must be the same!", support_vectors.num_rows(), alpha.num_cols());
    PLSSVM_ASSERT(rho.size() == alpha.num_rows(), "The number of rho values ({}) and the number of weight vectors ({}) must be the same!", rho.size(), alpha.num_rows());
    PLSSVM_ASSERT(w.empty() || w.is_padded(), "Either w must be empty or must be padded!");
    PLSSVM_ASSERT(w.empty() || support_vectors.num_cols() == w.num_cols(), "Either w must be empty or contain exactly the same number of values ({}) as features are present ({})!", w.num_cols(), support_vectors.num_cols());
    PLSSVM_ASSERT(w.empty() || alpha.num_rows() == w.num_rows(), "Either w must be empty or contain exactly the same number of vectors ({}) as the alpha vector ({})!", w.num_rows(), alpha.num_rows());
    PLSSVM_ASSERT(!predict_points.empty(), "The data points to predict must not be empty!");
    PLSSVM_ASSERT(predict_points.is_padded(), "The data points to predict must be padded!");
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "The number of features in the support vectors ({}) must be the same as in the data points to predict ({})!", support_vectors.num_cols(), predict_points.num_cols());

    // defined sizes
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    // num_predict_points x num_classes
    aos_matrix<real_type> out{ plssvm::shape{ num_predict_points, num_classes }, real_type{ 0.0 }, plssvm::shape{ PADDING_SIZE, PADDING_SIZE } };

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w = soa_matrix<real_type>{ plssvm::shape{ num_classes, num_features }, plssvm::shape{ PADDING_SIZE, PADDING_SIZE } };
            detail::device_kernel_w_linear(w, alpha, support_vectors);
        }
    }

    // call the predict kernels
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            // predict the values using the w vector
            detail::device_kernel_predict_linear(out, w, rho, predict_points);
            break;
        case kernel_function_type::polynomial:
            detail::device_kernel_predict<kernel_function_type::polynomial>(out, alpha, rho, support_vectors, predict_points, params.degree, std::get<real_type>(params.gamma), params.coef0);
            break;
        case kernel_function_type::rbf:
            detail::device_kernel_predict<kernel_function_type::rbf>(out, alpha, rho, support_vectors, predict_points, std::get<real_type>(params.gamma));
            break;
        case kernel_function_type::sigmoid:
            detail::device_kernel_predict<kernel_function_type::sigmoid>(out, alpha, rho, support_vectors, predict_points, std::get<real_type>(params.gamma), params.coef0);
            break;
        case kernel_function_type::laplacian:
            detail::device_kernel_predict<kernel_function_type::laplacian>(out, alpha, rho, support_vectors, predict_points, std::get<real_type>(params.gamma));
            break;
        case kernel_function_type::chi_squared:
            detail::device_kernel_predict<kernel_function_type::chi_squared>(out, alpha, rho, support_vectors, predict_points, std::get<real_type>(params.gamma));
            break;
    }

    return out;
}

}  // namespace plssvm::stdpar
