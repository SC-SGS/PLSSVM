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
#include "plssvm/detail/type_traits.hpp"  // PLSSVM_REQUIRES
#include "plssvm/matrix.hpp"              // plssvm::aos_matrix
#include "plssvm/parameter.hpp"           // plssvm::parameter, plssvm::detail::{parameter, has_only_parameter_named_args_v}
#include "plssvm/target_platforms.hpp"    // plssvm::target_platform

#include "plssvm/detail/simple_any.hpp"

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


    detail::simple_any  setup_data_on_devices(const aos_matrix<float> &A) override { return this->setup_data_on_devices_impl(A); }
    detail::simple_any  setup_data_on_devices(const aos_matrix<double> &A) override { return this->setup_data_on_devices_impl(A); }
    template <typename real_type>
    detail::simple_any  setup_data_on_devices_impl(const aos_matrix<real_type> &A);

    detail::simple_any assemble_kernel_matrix_explicit(const detail::parameter<float> &params, const detail::simple_any &data, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<float> &q_red, float QA_cost) override { return this->assemble_kernel_matrix_explicit_impl(params, data, num_rows_reduced, num_features, q_red, QA_cost); }
    detail::simple_any assemble_kernel_matrix_explicit(const detail::parameter<double> &params, const detail::simple_any &data, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<double> &q_red, double QA_cost) override { return this->assemble_kernel_matrix_explicit_impl(params, data, num_rows_reduced, num_features, q_red, QA_cost); }
    template <typename real_type>
    detail::simple_any assemble_kernel_matrix_explicit_impl(const detail::parameter<real_type> &params, const detail::simple_any &data, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<real_type> &q_red, real_type QA_cost);

    [[nodiscard]] aos_matrix<float> kernel_matrix_matmul_explicit(const detail::simple_any &explicit_kernel_matrix, const aos_matrix<float> &vec) override { return this->kernel_matrix_matmul_explicit_impl(explicit_kernel_matrix, vec); }
    [[nodiscard]] aos_matrix<double> kernel_matrix_matmul_explicit(const detail::simple_any &explicit_kernel_matrix, const aos_matrix<double> &vec) override { return this->kernel_matrix_matmul_explicit_impl(explicit_kernel_matrix, vec); }
    template <typename real_type>
    [[nodiscard]] aos_matrix<real_type> kernel_matrix_matmul_explicit_impl(const detail::simple_any &explicit_kernel_matrix, const aos_matrix<real_type> &vec);

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