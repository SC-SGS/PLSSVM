/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                   // plssvm::backend_type
#include "plssvm/backends/OpenMP/detail/utility.hpp"                                  // plssvm::openmp::detail::{get_num_threads, get_openmp_version}
#include "plssvm/backends/OpenMP/exceptions.hpp"                                      // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/kernel/cg_explicit/blas.hpp"                         // plssvm::openmp::detail::device_kernel_symm
#include "plssvm/backends/OpenMP/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::openmp::detail::device_kernel_assembly
#include "plssvm/backends/OpenMP/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::openmp::detail::device_kernel_assembly_symm
#include "plssvm/backends/OpenMP/kernel/predict_kernel.hpp"                           // plssvm::openmp::detail::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                       // plssvm::real_type
#include "plssvm/detail/assert.hpp"                                                   // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"                                        // plssvm::detail::triangular_data_distribution
#include "plssvm/detail/logging/mpi_log_untracked.hpp"                                // plssvm::detail::log_untracked
#include "plssvm/detail/make_unique_for_overwrite.hpp"                                // plssvm::detail::{make_unique_for_overwrite, parallel_zero_memset}
#include "plssvm/detail/memory_size.hpp"                                              // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"                                            // plssvm::detail::{move_only_any, move_only_any_cast}
#include "plssvm/detail/tracking/performance_tracker.hpp"                             // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/utility.hpp"                                                  // plssvm::detail::get_system_memory
#include "plssvm/gamma.hpp"                                                           // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                                          // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/mpi/communicator.hpp"                                                // plssvm::mpi::communicator
#include "plssvm/mpi/detail/information.hpp"                                          // plssvm::mpi::detail::gather_and_print_csvm_information
#include "plssvm/parameter.hpp"                                                       // plssvm::parameter
#include "plssvm/shape.hpp"                                                           // plssvm::shape
#include "plssvm/solver_types.hpp"                                                    // plssvm::solver_type
#include "plssvm/svm/csvm.hpp"                                                        // plssvm::csvm
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                                // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format

#include <chrono>   // std::chrono::{steady_clock, duration_cast}
#include <cmath>    // std::fma
#include <cstddef>  // std::size_t
#include <cstring>  // std::memset
#include <tuple>    // std::tuple, std::make_tuple
#include <utility>  // std::pair, std::make_pair, std::move
#include <variant>  // std::get
#include <vector>   // std::vector

namespace plssvm::openmp {

csvm::csvm(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::cpu) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the OpenMP backend!", target) };
    }
    // the CPU target must be available
#if !defined(PLSSVM_HAS_CPU_TARGET)
    throw backend_exception{ "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!" };
#endif

    // update the target platform
    target_ = plssvm::target_platform::cpu;

    if (comm_.size() > 1) {
        mpi::detail::gather_and_print_csvm_information(comm_, plssvm::backend_type::openmp, target_);
    } else {
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "\nUsing OpenMP ({}) as backend with {} thread(s).\n",
                                      detail::get_openmp_version(),
                                      detail::get_num_threads());
    }

    plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "\n");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "openmp_version", detail::get_openmp_version() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "backend", plssvm::backend_type::openmp }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "num_threads", detail::get_num_threads() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", this->num_available_devices() }));
}

csvm::~csvm() = default;

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    return { ::plssvm::detail::get_system_memory() };
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    return this->get_device_memory();
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

    // update the data distribution: only the upper triangular kernel matrix is used
    // note: account for the dimensional reduction
    data_distribution_ = std::make_unique<::plssvm::detail::triangular_data_distribution>(comm_, A.num_rows() - 1, this->num_available_devices());
    // get the triangular data distribution
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);

    std::vector<::plssvm::detail::move_only_any> kernel_matrices_parts(this->num_available_devices());
    const real_type cost = real_type{ 1.0 } / params.cost;

    if (dist.place_specific_num_rows(0) > std::size_t{ 0 }) {
        switch (solver) {
            case solver_type::automatic:
                // unreachable
                break;
            case solver_type::cg_explicit:
                {
                    // calculate the number of data points this device is responsible for
                    const std::size_t device_specific_num_rows = dist.place_specific_num_rows(0);

                    // get the offset of the data points this device is responsible for
                    const std::size_t row_offset = dist.place_row_offset(0);

                    // get the number of kernel matrix entries
                    const std::size_t num_entries = dist.calculate_explicit_kernel_matrix_num_entries_padded(0);

                    // only explicitly store the upper triangular matrix
                    auto kernel_matrix = ::plssvm::detail::make_unique_for_overwrite<real_type[]>(num_entries);
                    // initialize kernel matrix to all zeros in parallel
                    ::plssvm::detail::parallel_zero_memset(kernel_matrix.get(), num_entries);

                    const auto start = std::chrono::steady_clock::now();
                    switch (params.kernel_type) {
                        case kernel_function_type::linear:
                            detail::device_kernel_assembly<kernel_function_type::linear>(kernel_matrix.get(), A, device_specific_num_rows, row_offset, q_red, QA_cost, cost);
                            break;
                        case kernel_function_type::polynomial:
                            detail::device_kernel_assembly<kernel_function_type::polynomial>(kernel_matrix.get(), A, device_specific_num_rows, row_offset, q_red, QA_cost, cost, params.degree, std::get<real_type>(params.gamma), params.coef0);
                            break;
                        case kernel_function_type::rbf:
                            detail::device_kernel_assembly<kernel_function_type::rbf>(kernel_matrix.get(), A, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<real_type>(params.gamma));
                            break;
                        case kernel_function_type::sigmoid:
                            detail::device_kernel_assembly<kernel_function_type::sigmoid>(kernel_matrix.get(), A, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<real_type>(params.gamma), params.coef0);
                            break;
                        case kernel_function_type::laplacian:
                            detail::device_kernel_assembly<kernel_function_type::laplacian>(kernel_matrix.get(), A, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<real_type>(params.gamma));
                            break;
                        case kernel_function_type::chi_squared:
                            detail::device_kernel_assembly<kernel_function_type::chi_squared>(kernel_matrix.get(), A, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<real_type>(params.gamma));
                            break;
                    }
                    const auto end = std::chrono::steady_clock::now();
                    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "kernel_matrix", "kernel_matrix_assembly_kernel", duration }));

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

    using namespace operators;

    // get the triangular data distribution
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);

    // check whether the current device is responsible for at least one data point!
    if (dist.place_specific_num_rows(0) > std::size_t{ 0 }) {
        if (!comm_.is_main_rank()) {
            // MPI rank 0 always touches all values in C -> other MPI ranks do not need C
            std::memset(C.data(), 0, C.size_padded() * sizeof(real_type));
        }

        // calculate the number of data points this device is responsible for
        const std::size_t device_specific_num_rows = dist.place_specific_num_rows(0);
        // get the offset of the data points this device is responsible for
        const std::size_t row_offset = dist.place_row_offset(0);

        const std::size_t num_rhs = B.shape().x;
        const std::size_t num_rows = B.shape().y;

        switch (solver) {
            case solver_type::automatic:
                // unreachable
                break;
            case solver_type::cg_explicit:
                {
                    const auto &explicit_A = ::plssvm::detail::move_only_any_cast<const std::unique_ptr<real_type[]> &>(A.front());
                    PLSSVM_ASSERT(explicit_A != nullptr, "The A matrix must not be empty!");

                    const auto start = std::chrono::steady_clock::now();

                    detail::device_kernel_symm(num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, explicit_A.get(), B, beta, C);

                    const std::size_t num_mirror_rows = num_rows - row_offset - device_specific_num_rows;
                    if (num_mirror_rows > std::size_t{ 0 }) {
                        detail::device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, explicit_A.get(), B, beta, C);
                    }

                    const auto end = std::chrono::steady_clock::now();
                    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "cg", "blas_level_3_times_kernel", duration }));
                }
                break;
            case solver_type::cg_implicit:
                {
                    const auto &[matr_A, params, q_red, QA_cost] = ::plssvm::detail::move_only_any_cast<const std::tuple<soa_matrix<real_type>, parameter, std::vector<real_type>, real_type> &>(A.front());
                    PLSSVM_ASSERT(!matr_A.empty(), "The A matrix must not be empty!");
                    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector must not be empty!");
                    const real_type cost = real_type{ 1.0 } / params.cost;

                    if (comm_.is_main_rank()) {
                        // we do not perform the beta scale in C in the cg_implicit device kernel
                        // -> calculate it using a separate kernel (always on device 0 and MPI rank 0!)
                        C *= beta;
                    }

                    const auto start = std::chrono::steady_clock::now();
                    switch (params.kernel_type) {
                        case kernel_function_type::linear:
                            detail::device_kernel_assembly_symm<kernel_function_type::linear>(alpha, q_red, matr_A, device_specific_num_rows, row_offset, QA_cost, cost, B, C);
                            break;
                        case kernel_function_type::polynomial:
                            detail::device_kernel_assembly_symm<kernel_function_type::polynomial>(alpha, q_red, matr_A, device_specific_num_rows, row_offset, QA_cost, cost, B, C, params.degree, std::get<real_type>(params.gamma), params.coef0);
                            break;
                        case kernel_function_type::rbf:
                            detail::device_kernel_assembly_symm<kernel_function_type::rbf>(alpha, q_red, matr_A, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<real_type>(params.gamma));
                            break;
                        case kernel_function_type::sigmoid:
                            detail::device_kernel_assembly_symm<kernel_function_type::sigmoid>(alpha, q_red, matr_A, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<real_type>(params.gamma), params.coef0);
                            break;
                        case kernel_function_type::laplacian:
                            detail::device_kernel_assembly_symm<kernel_function_type::laplacian>(alpha, q_red, matr_A, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<real_type>(params.gamma));
                            break;
                        case kernel_function_type::chi_squared:
                            detail::device_kernel_assembly_symm<kernel_function_type::chi_squared>(alpha, q_red, matr_A, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<real_type>(params.gamma));
                            break;
                    }
                    const auto end = std::chrono::steady_clock::now();
                    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "cg", "blas_level_3_times_kernel", duration }));
                }
                break;
        }
    }
    // restore padding entries by setting them to zero
    C.restore_padding();
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
    const std::size_t num_sv = support_vectors.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    // num_predict_points x num_classes
    aos_matrix<real_type> out{ plssvm::shape{ num_predict_points, num_classes }, real_type{ 0.0 }, plssvm::shape{ PADDING_SIZE, PADDING_SIZE } };

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // update the data distribution to account for the support vectors
            data_distribution_ = std::make_unique<::plssvm::detail::rectangular_data_distribution>(comm_, num_sv, this->num_available_devices());

            // fill w vector
            w = soa_matrix<real_type>{ plssvm::shape{ num_classes, num_features }, plssvm::shape{ PADDING_SIZE, PADDING_SIZE } };

            if (data_distribution_->place_specific_num_rows(0) > std::size_t{ 0 }) {
                const std::size_t device_specific_num_sv = data_distribution_->place_specific_num_rows(0);
                const std::size_t sv_offset = data_distribution_->place_row_offset(0);

                const auto start = std::chrono::steady_clock::now();

                detail::device_kernel_w_linear(w, alpha, support_vectors, device_specific_num_sv, sv_offset);

                const auto end = std::chrono::steady_clock::now();
                [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "predict_values", "w_kernel", duration }));
            }
            // restore padding entries by setting them to zero
            w.restore_padding();

            // reduce w on all MPI ranks
            comm_.allreduce_inplace(w);
        }
    }

    data_distribution_ = std::make_unique<::plssvm::detail::rectangular_data_distribution>(comm_, num_predict_points, this->num_available_devices());
    const std::size_t device_specific_num_predict_points = data_distribution_->place_specific_num_rows(0);
    const std::size_t row_offset = data_distribution_->place_row_offset(0);

    if (data_distribution_->place_specific_num_rows(0) > std::size_t{ 0 }) {
        const auto start = std::chrono::steady_clock::now();
        // call the predict kernels
        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // predict the values using the w vector
                detail::device_kernel_predict_linear(out, w, rho, predict_points, device_specific_num_predict_points, row_offset);
                break;
            case kernel_function_type::polynomial:
                detail::device_kernel_predict<kernel_function_type::polynomial>(out, alpha, rho, support_vectors, predict_points, device_specific_num_predict_points, row_offset, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::device_kernel_predict<kernel_function_type::rbf>(out, alpha, rho, support_vectors, predict_points, device_specific_num_predict_points, row_offset, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::device_kernel_predict<kernel_function_type::sigmoid>(out, alpha, rho, support_vectors, predict_points, device_specific_num_predict_points, row_offset, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::device_kernel_predict<kernel_function_type::laplacian>(out, alpha, rho, support_vectors, predict_points, device_specific_num_predict_points, row_offset, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::device_kernel_predict<kernel_function_type::chi_squared>(out, alpha, rho, support_vectors, predict_points, device_specific_num_predict_points, row_offset, std::get<real_type>(params.gamma));
                break;
        }
        const auto end = std::chrono::steady_clock::now();
        [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "predict_values", "predict_kernel", duration }));
    }

    // restore padding entries by setting them to zero
    out.restore_padding();
    return out;
}

}  // namespace plssvm::openmp
