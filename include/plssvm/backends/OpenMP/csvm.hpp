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

#include "plssvm/csvm.hpp"                   // plssvm::csvm
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include <type_traits>  // std::true_type
#include <vector>       // std::vector

namespace plssvm {

namespace openmp {

/**
 * @brief A C-SVM implementation using OpenMP as backend.
 * @tparam T the type of the data
 */
class csvm : public ::plssvm::csvm {
  protected:
    // protected for test mock class
    /// The template base type of the OpenMP C-SVM class.
    using base_type = ::plssvm::csvm;

  public:
    /// The type of the data. Must be either `float` or `double`.
    //    using typename base_type::real_type;
    using typename base_type::size_type;

    explicit csvm(parameter params = {}) :
        csvm{ plssvm::target_platform::automatic, params } {}
    /**
     * @brief Construct a new C-SVM using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    explicit csvm(target_platform target, parameter params = {});
    template <typename... Args>
    csvm(const target_platform target, const kernel_function_type kernel, Args &&...named_args) :
        base_type{ kernel, std::forward<Args>(named_args)... } {
        this->init(target);
    }
    template <typename... Args>
    explicit csvm(const kernel_function_type kernel, Args &&...named_args) :
        base_type{ kernel, std::forward<Args>(named_args)... } {
        // the default target is the automatic one
        this->init(plssvm::target_platform::automatic);
    }

  private:
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    [[nodiscard]] std::pair<std::vector<float>, float> solve_system_of_linear_equations(const detail::parameter<float> &params, const std::vector<std::vector<float>> &A, std::vector<float> b, float eps, size_type max_iter) const override { return this->solve_system_of_linear_equations_impl(params, A, b, eps, max_iter); }
    [[nodiscard]] std::pair<std::vector<double>, double> solve_system_of_linear_equations(const detail::parameter<double> &params, const std::vector<std::vector<double>> &A, std::vector<double> b, double eps, size_type max_iter) const override { return this->solve_system_of_linear_equations_impl(params, A, b, eps, max_iter); }

    template <typename real_type>
    [[nodiscard]] std::pair<std::vector<real_type>, real_type> solve_system_of_linear_equations_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, real_type eps, size_type max_iter) const;

    /**
     * @copydoc plssvm::csvm::predict_values_impl
     */
    [[nodiscard]] std::vector<float> predict_values(const detail::parameter<float> &params, const std::vector<std::vector<float>> &support_vectors, const std::vector<float> &alpha, float rho, std::vector<float> &w, const std::vector<std::vector<float>> &predict_points) const override { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    [[nodiscard]] std::vector<double> predict_values(const detail::parameter<double> &params, const std::vector<std::vector<double>> &support_vectors, const std::vector<double> &alpha, double rho, std::vector<double> &w, const std::vector<std::vector<double>> &predict_points) const override { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }

    template <typename real_type>
    [[nodiscard]] std::vector<real_type> predict_values_impl(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const;

    /**
     * @copydoc plssvm::csvm::generate_q
     */
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> generate_q(const detail::parameter<real_type> &params, const std::vector<std::vector<real_type>> &data) const;
    /**
     * @copydoc plssvm::csvm::calculate_w
     */
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> calculate_w(const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &alpha) const;

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the CPU using OpenMP.
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    template <typename real_type>
    void run_device_kernel(const detail::parameter<real_type> &params, const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type add) const;

    /**
     * @brief Initializes the OpenMP backend and performs some sanity checks.
     * @param[in] target the platform to run on (must be `plssvm::target_platfrom::cpu` for the OpenMP backend).
     */
    void init(target_platform target);
};

}  // namespace openmp

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the OpenMP are available.
 */
template <>
struct csvm_backend_exists<openmp::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_OPENMP_CSVM_HPP_