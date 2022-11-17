/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_CSVM_HPP_
#define PLSSVM_BACKENDS_HIP_CSVM_HPP_
#pragma once

#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"  // plssvm::hip::detail::device_ptr
#include "plssvm/backends/gpu_csvm.hpp"                   // plssvm::detail::gpu_csvm
#include "plssvm/kernel_function_types.hpp"               // plssvm::kernel_type
#include "plssvm/parameter.hpp"                           // plssvm::parameter
#include "plssvm/target_platforms.hpp"                    // plssvm::target_platform

#include <cstddef>      // std::size_t
#include <type_traits>  // std::true_type
#include <utility>      // std::forward

namespace plssvm {

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace hip {

/**
 * @brief A C-SVM implementation using HIP as backend.
 */
class csvm : public ::plssvm::detail::gpu_csvm<::plssvm::hip::detail::device_ptr, int> {
  protected:
    // protected for the test mock class
    /// The template base type of the HIP C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<::plssvm::hip::detail::device_ptr, int>;

    using base_type::devices_;

  public:
    using base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the HIP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit csvm(parameter params = {}) :
        csvm{ plssvm::target_platform::automatic, params } {}
    /**
     * @brief Construct a new C-SVM using the HIP backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] params struct encapsulating all possible SVM parameters
     */
    explicit csvm(target_platform target, parameter params = {}) :
        base_type{ params } {
        this->init(target);
    }

    /**
     * @brief Construct a new C-SVM using the HIP backend and the optionally provided @p named_args.
     * @param[in] named_args the additional optional named arguments
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(Args &&...named_args) :
        csvm{ plssvm::target_platform::automatic, std::forward<Args>(named_args)... } {}

    /**
     * @brief Construct a new C-SVM using the HIP backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] named_args the additional optional named arguments
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(const target_platform target, Args &&...named_args) :
        base_type{ std::forward<Args>(named_args)... } {
        this->init(target);
    }

    /**
     * @brief Wait for all operations on all HIP devices to finish.
     * @details Terminates the program, if any exception is thrown.
     */
    ~csvm() override;

  protected:
    /**
     * @copydoc plssvm::detail::gpu_csvm::device_synchronize
     */
    void device_synchronize(const queue_type &queue) const final;

    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    void run_q_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<float> &params, device_ptr_type<float> &q_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const final { this->run_q_kernel_impl(device, range, params, q_d, data_d, data_last_d, num_data_points_padded, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    void run_q_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<double> &params, device_ptr_type<double> &q_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const final { this->run_q_kernel_impl(device, range, params, q_d, data_d, data_last_d, num_data_points_padded, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    template <typename real_type>
    void run_q_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &q_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<float> &params, const device_ptr_type<float> &q_d, device_ptr_type<float> &r_d, const device_ptr_type<float> &x_d, const device_ptr_type<float> &data_d, float QA_cost, float add, std::size_t num_data_points_padded, std::size_t num_features) const final { this->run_svm_kernel_impl(device, range, params, q_d, r_d, x_d, data_d, QA_cost, add, num_data_points_padded, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<double> &params, const device_ptr_type<double> &q_d, device_ptr_type<double> &r_d, const device_ptr_type<double> &x_d, const device_ptr_type<double> &data_d, double QA_cost, double add, std::size_t num_data_points_padded, std::size_t num_features) const final { this->run_svm_kernel_impl(device, range, params, q_d, r_d, x_d, data_d, QA_cost, add, num_data_points_padded, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    template <typename real_type>
    void run_svm_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, real_type QA_cost, real_type add, std::size_t num_data_points_padded, std::size_t num_features) const;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<float> &w_d, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_data_points, std::size_t num_features) const final { this->run_w_kernel_impl(device, range, w_d, alpha_d, data_d, data_last_d, num_data_points, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<double> &w_d, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_data_points, std::size_t num_features) const final { this->run_w_kernel_impl(device, range, w_d, alpha_d, data_d, data_last_d, num_data_points, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    template <typename real_type>
    void run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, std::size_t num_data_points, std::size_t num_features) const;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<float> &params, device_ptr_type<float> &out_d, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &point_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_support_vectors, std::size_t num_predict_points, std::size_t num_features) const final { this->run_predict_kernel_impl(range, params, out_d, alpha_d, point_d, data_d, data_last_d, num_support_vectors, num_predict_points, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<double> &params, device_ptr_type<double> &out_d, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &point_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_support_vectors, std::size_t num_predict_points, std::size_t num_features) const final { this->run_predict_kernel_impl(range, params, out_d, alpha_d, point_d, data_d, data_last_d, num_support_vectors, num_predict_points, num_features); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    template <typename real_type>
    void run_predict_kernel_impl(const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &out_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &point_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, std::size_t num_support_vectors, std::size_t num_predict_points, std::size_t num_features) const;

  private:
    /**
     * @brief Initialize all important states related to the HIP backend.
     * @param[in] target the target platform to use
     */
    void init(target_platform target);
};

}  // namespace hip

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the HIP are available.
 */
template <>
struct csvm_backend_exists<hip::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_HIP_CSVM_HPP_
