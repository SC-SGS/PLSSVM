/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the stdpar backend.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_CSVM_HPP_
#define PLSSVM_BACKENDS_STDPAR_CSVM_HPP_
#pragma once

#include "plssvm/constants.hpp"             // plssvm::real_type
#include "plssvm/csvm.hpp"                  // plssvm::csvm, plssvm::detail::csvm_backend_exists
#include "plssvm/detail/memory_size.hpp"    // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"  // plssvm::detail::move_only_any
#include "plssvm/detail/type_traits.hpp"    // PLSSVM_REQUIRES
#include "plssvm/matrix.hpp"                // plssvm::aos_matrix
#include "plssvm/parameter.hpp"             // plssvm::parameter, plssvm::detail::has_only_parameter_named_args_v
#include "plssvm/solver_types.hpp"          // plssvm::solver_type
#include "plssvm/target_platforms.hpp"      // plssvm::target_platform

#include <cstddef>      // std::size_t
#include <type_traits>  // std::true_type
#include <utility>      // std::forward, std::pair
#include <vector>       // std::vector

namespace plssvm {

namespace stdpar {

/**
 * @brief A C-SVM implementation using stdpar as backend.
 */
class csvm : public ::plssvm::csvm {
  public:
    /**
     * @brief Construct a new C-SVM using the stdpar backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::stdpar::backend_exception if the requested target is not available
     * @throws plssvm::stdpar::backend_exception if no device for the requested target was found
     */
    explicit csvm(parameter params = {});
    /**
     * @brief Construct a new C-SVM using the stdpar backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::stdpar::backend_exception if the requested target is not available
     * @throws plssvm::stdpar::backend_exception if no device for the requested target was found
     */
    explicit csvm(target_platform target, parameter params = {});

    /**
     * @brief Construct a new C-SVM using the stdpar backend and the optionally provided @p named_args.
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::stdpar::backend_exception if the requested target is not available
     * @throws plssvm::stdpar::backend_exception if no device for the requested target was found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(Args &&...named_args) :
        ::plssvm::csvm{ std::forward<Args>(named_args)... } {
        // the default target is the automatic one
        this->init(plssvm::target_platform::automatic);
    }

    /**
     * @brief Construct a new C-SVM using the stdpar backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::stdpar::backend_exception if the requested target is not available
     * @throws plssvm::stdpar::backend_exception if no device for the requested target was found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(const target_platform target, Args &&...named_args) :
        ::plssvm::csvm{ std::forward<Args>(named_args)... } {
        this->init(target);
    }

    /**
     * @copydoc plssvm::csvm::csvm(const plssvm::csvm &)
     */
    csvm(const csvm &) = delete;
    /**
     * @copydoc plssvm::csvm::csvm(plssvm::csvm &&) noexcept
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @copydoc plssvm::csvm::operator=(const plssvm::csvm &)
     */
    csvm &operator=(const csvm &) = delete;
    /**
     * @copydoc plssvm::csvm::operator=(plssvm::csvm &&) noexcept
     */
    csvm &operator=(csvm &&) noexcept = default;
    /**
     * @brief Default destructor since the copy and move constructors and copy- and move-assignment operators are defined.
     */
    ~csvm() override = default;

    /**
     * @copydoc plssvm::csvm::num_available_devices
     * @note We currently only support one device for C++ standard parallelism.
     */
    [[nodiscard]] std::size_t num_available_devices() const noexcept override {
        return 1;
    }

  protected:
    /**
     * @copydoc plssvm::csvm::get_device_memory
     */
    [[nodiscard]] std::vector<::plssvm::detail::memory_size> get_device_memory() const final;
    /**
     * @copydoc plssvm::csvm::get_max_mem_alloc_size
     */
    [[nodiscard]] std::vector<::plssvm::detail::memory_size> get_max_mem_alloc_size() const final;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix
     */
    [[nodiscard]] std::vector<::plssvm::detail::move_only_any> assemble_kernel_matrix(solver_type solver, const parameter &params, const soa_matrix<real_type> &A, const std::vector<real_type> &q_red, real_type QA_cost) const final;
    /**
     * @copydoc plssvm::csvm::blas_level_3
     */
    void blas_level_3(solver_type solver, real_type alpha, const std::vector<::plssvm::detail::move_only_any> &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const final;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<real_type> predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const final;

  private:
    /**
     * @brief Initializes the stdpar backend and performs some sanity checks.
     * @param[in] target the target platform to use
     * @throws plssvm::stdpar::backend_exception if the requested target is not available
     * @throws plssvm::stdpar::backend_exception if no device for the requested target was found
     */
    void init(target_platform target);
};

}  // namespace stdpar

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the OpenMP backend are available.
 */
template <>
struct csvm_backend_exists<stdpar::csvm> : std::true_type { };

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_STDPAR_CSVM_HPP_
