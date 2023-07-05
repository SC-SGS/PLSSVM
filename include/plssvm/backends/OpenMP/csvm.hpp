/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_CSVM_HPP_
#define PLSSVM_BACKENDS_OPENMP_CSVM_HPP_
#pragma once

#include "plssvm/csvm.hpp"                // plssvm::csvm
#include "plssvm/detail/simple_any.hpp"   // plssvm::detail::simple_any
#include "plssvm/detail/type_traits.hpp"  // PLSSVM_REQUIRES
#include "plssvm/matrix.hpp"              // plssvm::aos_matrix
#include "plssvm/parameter.hpp"           // plssvm::parameter, plssvm::detail::{parameter, has_only_parameter_named_args_v}
#include "plssvm/target_platforms.hpp"    // plssvm::target_platform


#include <type_traits>                    // std::true_type
#include <utility>                        // std::forward, std::pair
#include <vector>                         // std::vector
#include <variant>

namespace plssvm {

namespace openmp {

/**
 * @brief A C-SVM implementation using OpenMP as backend.
 */
class csvm : public ::plssvm::csvm {
  public:
    /**
     * @brief Construct a new C-SVM using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    explicit csvm(parameter params = {});
    /**
     * @brief Construct a new C-SVM using the OpenMP backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    explicit csvm(target_platform target, parameter params = {});

    /**
     * @brief Construct a new C-SVM using the OpenMP backend and the optionally provided @p named_args.
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(Args &&...named_args) :
        ::plssvm::csvm{ std::forward<Args>(named_args)... } {
        // the default target is the automatic one
        this->init(plssvm::target_platform::automatic);
    }
    /**
     * @brief Construct a new C-SVM using the OpenMP backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_parameter_named_args_v<Args...>)>
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

  protected:
    /**
     * @copydoc plssvm::csvm::get_device_memory
     */
    unsigned long long get_device_memory() const final;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::setup_data_on_devices
     */
    [[nodiscard]] detail::simple_any setup_data_on_devices(const solver_type solver, const aos_matrix<float> &A) const final { return this->setup_data_on_devices_impl(solver, A); }
    /**
     * @copydoc plssvm::csvm::setup_data_on_devices
     */
    [[nodiscard]] detail::simple_any setup_data_on_devices(const solver_type solver, const aos_matrix<double> &A) const final { return this->setup_data_on_devices_impl(solver, A); }
    /**
     * @copydoc plssvm::csvm::setup_data_on_devices
     */
    template <typename real_type>
    [[nodiscard]] detail::simple_any setup_data_on_devices_impl(solver_type solver, const aos_matrix<real_type> &A) const;

    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix
     */
    [[nodiscard]] detail::simple_any assemble_kernel_matrix(const solver_type solver, const detail::parameter<float> &params, const detail::simple_any &data, const std::vector<float> &q_red, const float QA_cost) const final { return this->assemble_kernel_matrix_impl(solver, params, data, q_red, QA_cost); }
    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix
     */
    [[nodiscard]] detail::simple_any assemble_kernel_matrix(const solver_type solver, const detail::parameter<double> &params, const detail::simple_any &data, const std::vector<double> &q_red, const double QA_cost) const final { return this->assemble_kernel_matrix_impl(solver, params, data, q_red, QA_cost); }
    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix
     */
    template <typename real_type>
    [[nodiscard]] detail::simple_any assemble_kernel_matrix_impl(solver_type solver, const detail::parameter<real_type> &params, const detail::simple_any &data, const std::vector<real_type> &q_red, real_type QA_cost) const;

    /**
     * @copydoc plssvm::csvm::blas_gemm
     */
    void blas_gemm(const solver_type solver, const float alpha, const detail::simple_any &A, const aos_matrix<float> &B, const float beta, aos_matrix<float> &C) const final { this->blas_gemm_impl(solver, alpha, A, B, beta, C); }
    /**
     * @copydoc plssvm::csvm::blas_gemm
     */
    void blas_gemm(const solver_type solver, const double alpha, const detail::simple_any &A, const aos_matrix<double> &B, const double beta, aos_matrix<double> &C) const final { this->blas_gemm_impl(solver, alpha, A, B, beta, C); }
    /**
     * @copydoc plssvm::csvm::blas_gemm
     */
    template <typename real_type>
    void blas_gemm_impl(solver_type solver, real_type alpha, const detail::simple_any &A, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C) const;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<float> predict_values(const detail::parameter<float> &params, const aos_matrix<float> &support_vectors, const aos_matrix<float> &alpha, const std::vector<float> &rho, aos_matrix<float> &w, const aos_matrix<float> &predict_points) const override { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<double> predict_values(const detail::parameter<double> &params, const aos_matrix<double> &support_vectors, const aos_matrix<double> &alpha, const std::vector<double> &rho, aos_matrix<double> &w, const aos_matrix<double> &predict_points) const override { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    template <typename real_type>
    [[nodiscard]] aos_matrix<real_type> predict_values_impl(const detail::parameter<real_type> &params, const aos_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, aos_matrix<real_type> &w, const aos_matrix<real_type> &predict_points) const;

    private:
    /**
     * @brief Initializes the OpenMP backend and performs some sanity checks.
     * @param[in] target the target platform to use
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    void init(target_platform target);
};

}  // namespace openmp

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the OpenMP backend are available.
 */
template <>
struct csvm_backend_exists<openmp::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_OPENMP_CSVM_HPP_