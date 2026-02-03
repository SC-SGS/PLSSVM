/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them.
 */

#ifndef PLSSVM_SVM_CSVM_HPP_
#define PLSSVM_SVM_CSVM_HPP_
#pragma once

#include "plssvm/constants.hpp"                            // plssvm::real_type, plssvm::PADDING_SIZE, plssvm::DEFAULT_EPSILON
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"             // plssvm::detail::{data_distribution, triangular_data_distribution}
#include "plssvm/detail/igor_utility.hpp"                  // plssvm::detail::{get_value_from_named_parameter, has_only_parameter_named_args_v}
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/logging/mpi_log_untracked.hpp"     // plssvm::detail::log_untracked
#include "plssvm/detail/memory_size.hpp"                   // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"                 // plssvm::detail::move_only_any
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/type_traits.hpp"                   // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::{check_local_memory_usage, get_system_memory}
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::invalid_parameter_exception
#include "plssvm/matrix.hpp"                               // plssvm::aos_matrix
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/mpi/detail/information.hpp"               // plssvm::mpi::detail::gather_and_print_solver_information
#include "plssvm/parameter.hpp"                            // plssvm::parameter
#include "plssvm/shape.hpp"                                // plssvm::shape
#include "plssvm/solver_types.hpp"                         // plssvm::solver_type
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "fmt/format.h"   // fmt::format
#include "fmt/ranges.h"   // fmt::join
#include "igor/igor.hpp"  // igor::parser

#include <algorithm>    // std::max
#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>      // std::size_t
#include <memory>       // std::unique_ptr
#include <optional>     // std::optional
#include <ratio>        // std::milli
#include <string>       // std::string
#include <tuple>        // std::tie, std::tuple, std::make_tuple
#include <type_traits>  // std::enable_if_t, std::false_type
#include <utility>      // std::pair, std::forward, std::move
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Base class for all C-SVM backends.
 * @details This class implements all features shared between all C-SVM backends.
 */
class csvm {
  public:
    /**
     * @brief Default constructor.
     * @details Needed due to multiple-inheritance.
     */
    csvm() = default;
    /**
     * @brief Construct a C-SVM using the SVM parameter @p params.
     * @details Uses the default SVM parameter if none are provided.
     * @param[in] comm the used MPI communicator
     * @param[in] params the SVM parameter
     */
    explicit csvm(mpi::communicator comm, parameter params = {});
    /**
     * @brief Construct a C-SVM forwarding all parameters @p args to the plssvm::parameter constructor.
     * @tparam Args the type of the (named-)parameters
     * @param[in] comm the used MPI communicator
     * @param[in] args the parameters used to construct a plssvm::parameter
     */
    template <typename... Args>
    explicit csvm(mpi::communicator comm, Args &&...args);

    /**
     * @brief Delete copy-constructor since a C-SVM is a move-only type.
     */
    csvm(const csvm &) = delete;
    /**
     * @brief Default move-constructor since a virtual destructor has been declared.
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @brief Delete copy-assignment operator since a C-SVM is a move-only type.
     * @return `*this`
     */
    csvm &operator=(const csvm &) = delete;
    /**
     * @brief Default move-assignment operator since a virtual destructor has been declared.
     * @return `*this`
     */
    csvm &operator=(csvm &&) noexcept = default;
    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~csvm() = default;

    /**
     * @brief Return the target platform (i.e, CPU or GPU including the vendor) this SVM runs on.
     * @return the target platform (`[[nodiscard]]`)
     */
    [[nodiscard]] target_platform get_target_platform() const noexcept { return target_; }

    /**
     * @brief Return the number of available devices.
     * @return the number of available devices (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::size_t num_available_devices() const noexcept = 0;

    /**
     * @brief Return the currently used SVM parameter.
     * @return the SVM parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] parameter get_params() const noexcept { return params_; }

    /**
     * @brief Override the old SVM parameter with the new plssvm::parameter @p params.
     * @param[in] params the new SVM parameter to use
     */
    void set_params(parameter params) noexcept { params_ = params; }

    /**
     * @brief Override the old SVM parameter with the new ones given as named parameters in @p named_args.
     * @tparam Args the type of the named-parameters
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_parameter_named_args_v<Args...>)>
    void set_params(Args &&...named_args);

    /**
     * @brief Get the associated MPI communicator.
     * @return the MPI communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] const mpi::communicator &communicator() const noexcept {
        return comm_;
    }

  protected:
    //*************************************************************************************************************************************//
    //                        pure virtual functions, must be implemented for all subclasses; doing the actual work                        //
    //*************************************************************************************************************************************//
    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @brief Calculate the total available device memory for all devices based on the used backend.
     * @return the total device memory per device (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<detail::memory_size> get_device_memory() const = 0;
    /**
     * @brief Return the maximum allocation size possible in a single allocation for all devices.
     * @return the maximum (single) allocation size per device (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<detail::memory_size> get_max_mem_alloc_size() const = 0;
    /**
     * @brief Calculate the total available local memory for all devices based on the used backend.
     * @details If the backend has no notion of local memory, returns a std::nullopt.
     * @return the total local memory per device (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<std::optional<detail::memory_size>> get_local_memory() const = 0;

    /**
     * @brief Explicitly assemble the kernel matrix using potentially multiple devices. Backend specific!
     * @param[in] solver the used solver type, determines the return type
     * @param[in] params the parameters used to assemble the kernel matrix (e.g., the used kernel function)
     * @param[in] A the data to assemble the kernel matrix from
     * @param[in] q_red the vector used in the dimensional reduction
     * @param[in] QA_cost the value used in the dimensional reduction
     * @return based on the used solver type (e.g., cg_explicit -> kernel matrix fully stored on the device (distributed across the devices); cg_implicit -> "nothing") (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<detail::move_only_any> assemble_kernel_matrix(solver_type solver, const parameter &params, const soa_matrix<real_type> &A, const std::vector<real_type> &q_red, real_type QA_cost) const = 0;

    /**
     * @brief Perform a BLAS level 3 matrix-matrix multiplication: `C = alpha * A * B + beta * C`.
     * @param[in] solver the used solver type, determines the type of @p A
     * @param[in] alpha the value to scale the result of the matrix-matrix multiplication
     * @param[in] A a matrix depending on the used solver type (e.g., cg_explicit -> the kernel matrix fully stored on the device (distributed across the devices); cg_implicit -> the input data set used to implicitly perform the matrix-matrix multiplication)
     * @param[in] B the other matrix to multiply the kernel matrix with
     * @param[in] beta the value to scale the matrix o add with
     * @param[in,out] C the result matrix and the matrix to add (inplace)
     */
    virtual void blas_level_3(solver_type solver, real_type alpha, const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const = 0;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] params the SVM parameters used in the respective kernel functions
     * @param[in] support_vectors the previously learned support vectors
     * @param[in] alpha the alpha values (weights) associated with the support vectors and classes
     * @param[in] rho the rho values for each class determined after training the model
     * @param[in,out] w the normal vectors to speedup prediction in case of the linear kernel function, an empty vector in case of the polynomial or rbf kernel
     * @param[in] predict_points the points to predict
     * @throws plssvm::exception any exception thrown by the backend's implementation
     * @return a vector filled with the predictions (not the actual labels!) (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual aos_matrix<real_type> predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const = 0;

    /**
     * @brief Solve the system of linear equations `K * X = B` where `K` is the kernel matrix assembled from @p A using the @p params with potentially multiple right-hand sides.
     * @tparam Args the type of the potential additional parameters
     * @param[in] A the data used to create the kernel matrix
     * @param[in] B the right-hand sides
     * @param[in] params the parameter to create the kernel matrix
     * @param[in] named_args additional parameters for the respective algorithm used to solve the system of linear equations
     * @return the result matrix `X`, the respective biases, and the number of iterations necessary for each right-hand side to converge (`[[nodiscard]]`)
     */
    template <typename... Args>
    [[nodiscard]] std::tuple<aos_matrix<real_type>, std::vector<real_type>, std::vector<unsigned long long>> solve_lssvm_system_of_linear_equations(const soa_matrix<real_type> &A, const aos_matrix<real_type> &B, const parameter &params, Args &&...named_args) const;
    /**
     * @brief Solve the system of linear equations `AX = B` where `A` is the kernel matrix using the Conjugate Gradients (CG) algorithm.
     * @param[in] A the kernel matrix; potentially distributed across multiple devices
     * @param[in] B the right-hand sides
     * @param[in] eps the termination criterion for the CG algorithm
     * @param[in] max_cg_iter the maximum number of CG iterations
     * @param[in] cg_solver the variation of the CG algorithm to use, i.e., how the kernel matrix is assembled (currently: explicit, streaming, implicit)
     * @return the result matrix `X` and the number of CG iterations necessary for each right-hand side to converge (`[[nodiscard]]`)
     */
    [[nodiscard]] std::pair<soa_matrix<real_type>, std::vector<unsigned long long>> conjugate_gradients(const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, real_type eps, unsigned long long max_cg_iter, solver_type cg_solver) const;
    /**
     * @brief Perform a dimensional reduction for the kernel matrix.
     * @details Reduces the resulting dimension by `2` compared to the original LS-SVM formulation.
     * @param[in] params the parameter used for the kernel matrix
     * @param[in] A the data used for the kernel matrix
     * @return the reduction vector `q_red` and the bottom-right value `QA_cost` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::pair<std::vector<real_type>, real_type> perform_dimensional_reduction(const parameter &params, const soa_matrix<real_type> &A) const;

    /**
     * @copydoc plssvm::csvm::blas_level_3
     * @details Small wrapper around the virtual `plssvm::csvm::blas_level_3` function to easily track its execution time.
     * @returns the duration of the BLAS routine in milliseconds (`[[nodiscard]]`)
     */
    [[nodiscard]] std::chrono::duration<long, std::milli> run_blas_level_3(solver_type solver, real_type alpha, const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const;

    /**
     * @copydoc plssvm::csvm::predict_values
     * @details Small wrapper around the virtual `plssvm::csvm::predict_values` function to easily track its execution time.
     */
    [[nodiscard]] aos_matrix<real_type> run_predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const;

    /// The SVM parameter (e.g., cost, degree, gamma, coef0) currently in use.
    parameter params_;
    /// The target platform of this SVM.
    target_platform target_{ plssvm::target_platform::automatic };
    /// The data distribution on the available devices.
    mutable std::unique_ptr<detail::data_distribution> data_distribution_;
    /// The used MPI communicator.
    mpi::communicator comm_;
};

inline csvm::csvm(mpi::communicator comm, parameter params) :
    params_{ params },
    comm_{ std::move(comm) } {
}

template <typename... Args>
csvm::csvm(mpi::communicator comm, Args &&...named_args) :
    params_{ std::forward<Args>(named_args)... },
    comm_{ std::move(comm) } {
}

template <typename... Args, std::enable_if_t<detail::has_only_parameter_named_args_v<Args...>, bool>>
void csvm::set_params(Args &&...named_args) {
    static_assert(sizeof...(Args) > 0, "At least one named parameter mus be given when calling set_params()!");

    // update the parameters
    params_.set_named_arguments(std::forward<Args>(named_args)...);
}

//*************************************************************************************************************************************//
//                                                       private member functions                                                      //
//*************************************************************************************************************************************//

template <typename... Args>
std::tuple<aos_matrix<real_type>, std::vector<real_type>, std::vector<unsigned long long>> csvm::solve_lssvm_system_of_linear_equations(const soa_matrix<real_type> &A, const aos_matrix<real_type> &B, const parameter &params, Args &&...named_args) const {
    PLSSVM_ASSERT(!A.empty(), "The A matrix must not be empty!");
    PLSSVM_ASSERT(A.is_padded(), "The A matrix must be padded!");
    PLSSVM_ASSERT((A.padding() == shape{ PADDING_SIZE, PADDING_SIZE }),
                  "The provided matrix must be padded with {}, but is padded with {}!",
                  shape{ PADDING_SIZE, PADDING_SIZE },
                  A.padding());
    PLSSVM_ASSERT(!B.empty(), "The B matrix must not be empty!");
    PLSSVM_ASSERT(A.num_rows() == B.num_cols(), "The number of data points in A ({}) and B ({}) must be the same!", A.num_rows(), B.num_cols());

    const igor::parser parser{ std::forward<Args>(named_args)... };

    // set default values
    auto used_epsilon{ DEFAULT_EPSILON };
    // NOTE: account for later dimensional reduction
    unsigned long long used_max_iter{ A.num_rows() - 1 };  // NOLINT: can be modified in compile-time if later on
    solver_type used_solver{ solver_type::automatic };

    // compile time check: only named parameters are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(epsilon, max_iter, classification, solver), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(epsilon)) {
        // get the value of the provided named parameter
        used_epsilon = detail::get_value_from_named_parameter<real_type>(parser, epsilon);
        // check if value makes sense
        if (used_epsilon <= real_type{ 0.0 }) {
            throw invalid_parameter_exception{ fmt::format("epsilon must be less than 0.0, but is {}!", used_epsilon) };
        }
    }
    if constexpr (parser.has(max_iter)) {
        // get the value of the provided named parameter
        used_max_iter = detail::get_value_from_named_parameter<unsigned long long>(parser, max_iter);
        // check if value makes sense
        if (used_max_iter == 0) {
            throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", used_max_iter) };
        }
    }
    if constexpr (parser.has(solver)) {
        // get the value of the provided parameter
        used_solver = detail::get_value_from_named_parameter<solver_type>(parser, solver);
    }

    const std::size_t num_rows = A.num_rows();
    const std::size_t num_features = A.num_cols();
    const std::size_t num_rows_reduced = num_rows - 1;
    const std::size_t num_rhs = B.num_rows();

    // determine the used local memory and check whether it exceeds the maximum necessary value!
    detail::check_local_memory_usage(this->get_local_memory());

    // determine the correct solver type, if the automatic solver type has been provided
    if (used_solver == solver_type::automatic) {
        using namespace detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace

        // define used safety margin constants
        constexpr detail::memory_size minimal_safety_margin = 512_MiB;
        constexpr long double percentual_safety_margin = 0.05L;
        const auto reduce_total_memory = [=](const detail::memory_size total_memory) {
            if (total_memory < minimal_safety_margin) {
                throw kernel_launch_resources{ fmt::format("At least {} of memory must be available, but available are only {}!", minimal_safety_margin, total_memory) };
            }
            return total_memory - std::max(total_memory * percentual_safety_margin, minimal_safety_margin);
        };

        // get the total and usable system memory (i.e., RAM)
        const detail::memory_size total_system_memory = detail::get_system_memory();
        const detail::memory_size usable_system_memory = reduce_total_memory(total_system_memory);

        // get the total and usable device memory per device (i.e., VRAM)
        const std::vector<detail::memory_size> total_device_memory_per_device = this->get_device_memory();
        const std::vector<detail::memory_size> usable_device_memory_per_device = [&]() {
            std::vector<detail::memory_size> res(total_device_memory_per_device.size());
            for (std::size_t device_id = 0; device_id < res.size(); ++device_id) {
                res[device_id] = reduce_total_memory(total_device_memory_per_device[device_id]);
            }
            return res;
        }();

        // calculate the maximum total memory needed for the explicit and implicit kernel matrix per device
        const detail::triangular_data_distribution data_distribution{ comm_, num_rows_reduced, this->num_available_devices() };
        const std::vector<detail::memory_size> total_memory_needed_explicit_per_device = data_distribution.calculate_maximum_explicit_kernel_matrix_memory_needed_per_place(num_features, num_rhs);
        const std::vector<detail::memory_size> total_memory_needed_implicit_per_device = data_distribution.calculate_maximum_implicit_kernel_matrix_memory_needed_per_place(num_features, num_rhs);

        // format a vector differentiating between it containing only a single entry or multiple
        const auto format_vector = [](const auto &vec) -> std::string {
            if (vec.size() == 1) {
                return fmt::format("{}", vec.front());
            }
            return fmt::format("[{}]", fmt::join(vec, ", "));
        };

        if (comm_.size() <= 1) {
            // output the necessary information on the console, full output only if a single MPI rank is used
            detail::log(verbosity_level::full,
                        comm_,
                        "Determining the solver type based on the available memory:\n"
                        "  - total system memory: {2}\n"
                        "  - usable system memory (with safety margin of min({0} %, {1}): {3}\n"
                        "  - total device memory: {4}\n"
                        "  - usable device memory (with safety margin of min({0} %, {1}): {5}\n"
                        "  - maximum memory needed (cg_explicit): {6}\n"
                        "  - maximum memory needed (cg_implicit): {7}\n",
                        static_cast<double>(percentual_safety_margin * 100.0L),  // NOLINT: convert float to percent by multiplying it with 100
                        minimal_safety_margin,
                        detail::tracking::tracking_entry{ "resource_constraints", "system_memory", total_system_memory },
                        detail::tracking::tracking_entry{ "resource_constraints", "usable_system_memory_with_safety_margin", usable_system_memory },
                        format_vector(total_device_memory_per_device),
                        format_vector(usable_device_memory_per_device),
                        format_vector(total_memory_needed_explicit_per_device),
                        format_vector(total_memory_needed_implicit_per_device));
        }
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "device_memory", total_device_memory_per_device }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "usable_device_memory_with_safety_margin", usable_device_memory_per_device }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "needed_device_memory_cg_explicit", total_memory_needed_explicit_per_device }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "needed_device_memory_cg_implicit", total_memory_needed_implicit_per_device }));

        // helper function to check whether ALL devices fulfill the requested memory constraint for the specific solver type
        const auto check_sizes = [](const auto &needed_memory_per_device, const auto &memory_constraint) -> std::vector<std::size_t> {
            PLSSVM_ASSERT(needed_memory_per_device.size() == memory_constraint.size(), "Can't check sizes due to device number mismatch! {} != {}", needed_memory_per_device.size(), memory_constraint.size());
            std::vector<std::size_t> failed_constraints{};
            for (std::size_t device_id = 0; device_id < needed_memory_per_device.size(); ++device_id) {
                if (needed_memory_per_device[device_id] > memory_constraint[device_id]) {
                    failed_constraints.push_back(device_id);
                }
            }
            return failed_constraints;
        };

        // select solver type based on the available memory
        // check whether the explicit (partial) kernel matrix can fit into each device memory
        if (const std::vector<std::size_t> failed_cg_explicit_constraints = check_sizes(total_memory_needed_explicit_per_device, usable_device_memory_per_device); failed_cg_explicit_constraints.empty()) {
            // use the explicit solver type
            used_solver = solver_type::cg_explicit;
        } else {
            if (comm_.size() <= 1) {
                // output only if a single MPI rank is used
                detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "Cannot use cg_explicit due to memory constraints on device(s) {}!\n",
                                      format_vector(failed_cg_explicit_constraints));
            }

            // check whether there is enough memory available for cg_implicit
            if (const std::vector<std::size_t> failed_cg_implicit_constraints = check_sizes(total_memory_needed_implicit_per_device, usable_device_memory_per_device); failed_cg_implicit_constraints.empty()) {
                // use the implicit solver type
                used_solver = solver_type::cg_implicit;
            } else {
                // not enough device memory available for the implicit case
                throw kernel_launch_resources{ fmt::format("Not enough device memory available on device(s) {} even for the cg_implicit solver!", format_vector(failed_cg_implicit_constraints)) };
            }
        }

        // enforce max mem alloc size if requested
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        // get the maximum possible memory allocation size per device
        const std::vector<detail::memory_size> max_mem_alloc_size_per_device = this->get_max_mem_alloc_size();

        // get the maximum single allocation size per device
        const std::vector<detail::memory_size> max_single_allocation_cg_explicit_size_per_device = data_distribution.calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place(num_features, num_rhs);
        const std::vector<detail::memory_size> max_single_allocation_cg_implicit_size_per_device = data_distribution.calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place(num_features, num_rhs);

        // output the maximum memory allocation size per device
        if (comm_.size() <= 1) {
            // output only if a single MPI rank is used
            detail::log_untracked(verbosity_level::full,
                                  comm_,
                                  "  - maximum supported single memory allocation size: {}\n"
                                  "  - maximum needed single memory allocation size (cg_explicit): {}\n"
                                  "  - maximum needed single memory allocation size (cg_implicit): {}\n",
                                  format_vector(max_mem_alloc_size_per_device),
                                  format_vector(max_single_allocation_cg_explicit_size_per_device),
                                  format_vector(max_single_allocation_cg_implicit_size_per_device));
        }
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "device_max_single_mem_alloc_size", max_mem_alloc_size_per_device }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "device_max_mem_alloc_size_cg_explicit", max_single_allocation_cg_explicit_size_per_device }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "device_max_mem_alloc_size_cg_implicit", max_single_allocation_cg_implicit_size_per_device }));

        // check whether the maximum single memory allocation sizes per device can be satisfied
        // check whether the maximum single cg_explicit memory allocation size can be satisfied
        if (const std::vector<std::size_t> failed_cg_explicit_constraints = check_sizes(max_single_allocation_cg_explicit_size_per_device, max_mem_alloc_size_per_device);
            used_solver == solver_type::cg_explicit && !failed_cg_explicit_constraints.empty()) {
            // max mem alloc size constraints not fulfilled
            if (comm_.size() <= 1) {
                // output only if a single MPI rank is used
                detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "Cannot use cg_explicit due to maximum single memory allocation constraints on device(s) {}! Falling back to cg_implicit.\n",
                                      format_vector(failed_cg_explicit_constraints));
            }
            // can't use cg_explicit
            used_solver = solver_type::cg_implicit;
        }
        if (const std::vector<std::size_t> failed_cg_implicit_constraints = check_sizes(max_single_allocation_cg_implicit_size_per_device, max_mem_alloc_size_per_device);
            used_solver == solver_type::cg_implicit && !failed_cg_implicit_constraints.empty()) {
            // can't fulfill maximum single memory allocation size even for cg_implicit
            if (comm_.size() <= 1) {
                // output only if a single MPI rank is used
                plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                                              comm_,
                                              "WARNING: if you are sure that the guaranteed maximum memory allocation size can be safely ignored on your device, "
                                              "this check can be disabled via \"-DPLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE=OFF\" during the CMake configuration!\n");
            }
            throw kernel_launch_resources{ fmt::format("Can't fulfill maximum single memory allocation constraint for device(s) {} even for the cg_implicit solver!", format_vector(failed_cg_implicit_constraints)) };
        }
#endif
    }

    if (comm_.size() <= 1) {
        // output only if a single MPI rank is used
        detail::log_untracked(verbosity_level::full,
                              comm_,
                              "Using {} as solver for AX=B.\n\n",
                              used_solver);
    } else {
        // multiple MPI ranks are used -> output used solver type in a more condensed way
        mpi::detail::gather_and_print_solver_information(comm_, used_solver);
    }
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "solver", "solver_type", used_solver }));

    // perform dimensional reduction
    // note: structured binding is rejected by clang HIP compiler!
    std::vector<real_type> q_red{};
    real_type QA_cost{};
    std::tie(q_red, QA_cost) = this->perform_dimensional_reduction(params, A);

    // update right-hand sides (B)
    std::vector<real_type> b_back_value(num_rhs);
    soa_matrix<real_type> B_red{ shape{ num_rhs, num_rows_reduced } };
#pragma omp parallel for default(none) shared(B, B_red, b_back_value) firstprivate(num_rhs, num_rows_reduced)
    for (std::size_t row = 0; row < num_rhs; ++row) {
        b_back_value[row] = B(row, num_rows_reduced);
        for (std::size_t col = 0; col < num_rows_reduced; ++col) {
            B_red(row, col) = B(row, col) - b_back_value[row];
        }
    }

    // assemble explicit kernel matrix
    const std::chrono::steady_clock::time_point assembly_start_time = std::chrono::steady_clock::now();
    const std::vector<detail::move_only_any> kernel_matrix = this->assemble_kernel_matrix(used_solver, params, A, q_red, QA_cost);
    const std::chrono::steady_clock::time_point assembly_end_time = std::chrono::steady_clock::now();
    const auto assembly_duration = std::chrono::duration_cast<std::chrono::milliseconds>(assembly_end_time - assembly_start_time);

    if (used_solver != solver_type::cg_implicit) {
        if (comm_.size() > 1) {
            // gather kernel matrix assembly runtimes from each MPI rank
            const std::vector<std::chrono::milliseconds> durations = comm_.gather(assembly_duration);

            detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "Assembled the kernel matrix in {} ({}).\n",
                                  *std::max_element(durations.cbegin(), durations.cend()),
                                  fmt::join(durations, "|"));

        } else {
            detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "Assembled the kernel matrix in {}.\n",
                                  assembly_duration);
        }
    }
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "kernel_matrix", "kernel_matrix_assembly", assembly_duration }));

    // choose the correct algorithm based on the (provided) solver type -> currently only CG available
    soa_matrix<real_type> X{};
    std::vector<unsigned long long> num_iter{};
    std::tie(X, num_iter) = this->conjugate_gradients(kernel_matrix, B_red, used_epsilon, used_max_iter, used_solver);

    // calculate bias and undo dimensional reduction
    aos_matrix<real_type> X_ret{ shape{ num_rhs, A.num_rows() }, shape{ PADDING_SIZE, PADDING_SIZE } };
    std::vector<real_type> bias(num_rhs);
#pragma omp parallel for default(none) shared(X, q_red, X_ret, bias, b_back_value) firstprivate(num_rhs, num_rows_reduced, QA_cost)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp_sum{ 0.0 };
        real_type temp_dot{ 0.0 };
#pragma omp simd reduction(+ : temp_sum) reduction(+ : temp_dot)
        for (std::size_t dim = 0; dim < num_rows_reduced; ++dim) {
            temp_sum += X(i, dim);
            temp_dot += q_red[dim] * X(i, dim);

            X_ret(i, dim) = X(i, dim);
        }
        bias[i] = -(b_back_value[i] + (QA_cost * temp_sum) - temp_dot);
        X_ret(i, num_rows_reduced) = -temp_sum;
    }

    return std::make_tuple(std::move(X_ret), std::move(bias), num_iter);
}

/// @cond Doxygen_suppress
namespace detail {

/**
 * @brief Sets the `value` to `false` since the given type @p T is either not a C-SVM or the C-SVM using the requested backend isn't available.
 * @tparam T the type of the C-SVM
 */
template <typename T, typename Enable = void>
struct csvm_backend_exists : std::false_type { };

}  // namespace detail

/// @endcond

/**
 * @brief Sets the value of the `value` member to `true` if @p T is a C-SVM using an available backend. Ignores any top-level const, volatile, and reference qualifiers.
 * @tparam T the type of the C-SVM
 */
template <typename T>
struct csvm_backend_exists : detail::csvm_backend_exists<detail::remove_cvref_t<T>> { };

/**
 * @brief Sets the value of the `value` member to `true` if @p T is a C-SVM using an available backend. Ignores any top-level const, volatile, and reference qualifiers.
 */
template <typename T>
constexpr bool csvm_backend_exists_v = csvm_backend_exists<T>::value;

}  // namespace plssvm

#endif  // PLSSVM_SVM_CSVM_HPP_
