/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a base C-SVM used for the different SYCL backends using Open SYCL as SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_OPENSYCL_CSVM_HPP_
#define PLSSVM_BACKENDS_SYCL_OPENSYCL_CSVM_HPP_
#pragma once

#include "plssvm/backends/SYCL/OpenSYCL/detail/device_ptr.hpp"
#include "plssvm/backends/SYCL/OpenSYCL/detail/queue.hpp"  // plssvm::opensycl::detail::queue (PImpl)

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/backends/gpu_csvm.hpp"                     // plssvm::detail::gpu_csvm
#include "plssvm/detail/type_traits.hpp"                    // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t
#include "plssvm/parameter.hpp"                             // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "igor/igor.hpp"  // igor::parser

#include <cstddef>      // std::size_t
#include <type_traits>  // std::is_same_v, std::true_type
#include <utility>      // std::forward

namespace plssvm {

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace opensycl {

/**
 * @brief A C-SVM implementation using Open SYCL (formerly hipSYCL) as SYCL backend.
 */
class csvm : public ::plssvm::detail::gpu_csvm<detail::device_ptr, detail::queue> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the Open SYCL SYCL C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<detail::device_ptr, detail::queue>;

    using base_type::devices_;

  public:
    using base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the SYCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::opensycl::backend_exception if the requested target is not available
     * @throws plssvm::opensycl::backend_exception if no device for the requested target was found
     */
    explicit csvm(parameter params = {});
    /**
     * @brief Construct a new C-SVM using the SYCL backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::opensycl::backend_exception if the requested target is not available
     * @throws plssvm::opensycl::backend_exception if no device for the requested target was found
     */
    explicit csvm(target_platform target, parameter params = {});

    /**
     * @brief Construct a new C-SVM using the SYCL backend and the optionally provided @p named_args.
     * @details Additionally sets the SYCL specific kernel invocation type.
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::opensycl::backend_exception if the requested target is not available
     * @throws plssvm::opensycl::backend_exception if no device for the requested target was found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_sycl_parameter_named_args_v<Args...>)>
    explicit csvm(Args &&...named_args) :
        csvm{ plssvm::target_platform::automatic, std::forward<Args>(named_args)... } {}
    /**
     * @brief Construct a new C-SVM using the SYCL backend on the @p target platform and the optionally provided @p named_args.
     * @details Additionally sets the SYCL specific kernel invocation type.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::opensycl::backend_exception if the requested target is not available
     * @throws plssvm::opensycl::backend_exception if no device for the requested target was found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_sycl_parameter_named_args_v<Args...>)>
    explicit csvm(const target_platform target, Args &&...named_args) :
        base_type{ named_args... } {
        // check igor parameter
        igor::parser parser{ std::forward<Args>(named_args)... };

        // check whether a specific SYCL kernel invocation type has been requested
        if constexpr (parser.has(sycl_kernel_invocation_type)) {
            // compile time check: the value must have the correct type
            static_assert(std::is_same_v<::plssvm::detail::remove_cvref_t<decltype(parser(sycl_kernel_invocation_type))>, sycl::kernel_invocation_type>, "Provided sycl_kernel_invocation_type must be convertible to a plssvm::sycl::kernel_invocation_type!");
            invocation_type_ = static_cast<sycl::kernel_invocation_type>(parser(sycl_kernel_invocation_type));
        }
        this->init(target);
    }

    /**
     * @brief Wait for all operations in all [`sycl::queue`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:interface.queue.class) to finish.
     * @details Terminates the program, if any asynchronous exception is thrown.
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
     * @brief Initialize all important states related to the SYCL backend.
     * @param[in] target the target platform to use
     * @throws plssvm::opensycl::backend_exception if the requested target is not available
     * @throws plssvm::opensycl::backend_exception if no device for the requested target was found
     */
    void init(target_platform target);

    /// The SYCL kernel invocation type for the svm kernel. Either nd_range or hierarchical.
    sycl::kernel_invocation_type invocation_type_{ sycl::kernel_invocation_type::automatic };
};
}  // namespace hipsycl

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the SYCL backend with Open SYCL as SYCL implementation are available.
 */
template <>
struct csvm_backend_exists<opensycl::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_SYCL_OPENSYCL_CSVM_HPP_