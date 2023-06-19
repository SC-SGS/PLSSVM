/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_CSVM_HPP_
#define PLSSVM_BACKENDS_CUDA_CSVM_HPP_
#pragma once

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/gpu_csvm.hpp"                // plssvm::detail::gpu_csvm
#include "plssvm/parameter.hpp"                        // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include <cstddef>                                     // std::size_t
#include <type_traits>                                 // std::true_type
#include <utility>                                     // std::forward

namespace plssvm {

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace cuda {

/**
 * @brief A C-SVM implementation using CUDA as backend.
 */
class csvm : public ::plssvm::detail::gpu_csvm<detail::device_ptr, int> {
  protected:
    // protected for the test mock class
    /// The template base type of the CUDA C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<detail::device_ptr, int>;

    using base_type::devices_;

  public:
    using base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the CUDA backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    explicit csvm(parameter params = {});
    /**
     * @brief Construct a new C-SVM using the CUDA backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    explicit csvm(target_platform target, parameter params = {});

    /**
     * @brief Construct a new C-SVM using the CUDA backend and the optionally provided @p named_args.
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(Args &&...named_args) :
        csvm{ plssvm::target_platform::automatic, std::forward<Args>(named_args)... } {}
    /**
     * @brief Construct a new C-SVM using the CUDA backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvm(const target_platform target, Args &&...named_args) :
        base_type{ std::forward<Args>(named_args)... } {
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
     * @brief Wait for all operations on all CUDA devices to finish.
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

    device_ptr_type<float> assemble_kernel_matrix(const ::plssvm::detail::parameter<float> &params, const device_ptr_type<float> &data_d, const std::vector<float> &q, float QA_cost, const std::size_t num_data_points, const std::size_t num_features) const final { return this->assemble_kernel_matrix_impl(params, data_d, q, QA_cost, num_data_points, num_features); }
    device_ptr_type<double> assemble_kernel_matrix(const ::plssvm::detail::parameter<double> &params, const device_ptr_type<double> &data_d, const std::vector<double> &q, double QA_cost, const std::size_t num_data_points, const std::size_t num_features) const final { return this->assemble_kernel_matrix_impl(params, data_d, q, QA_cost, num_data_points, num_features); }
    template <typename real_type>
    device_ptr_type<real_type> assemble_kernel_matrix_impl(const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &data_d, const std::vector<real_type> &q, const real_type QA_cost, const std::size_t num_data_points, const std::size_t num_features) const;

    void cg_blas_matmul(std::size_t m, std::size_t n, std::size_t k, float alpha, const device_ptr_type<float> &A, const device_ptr_type<float> &B, device_ptr_type<float> &RET) const final { this->cg_blas_matmul_impl(m, n, k, alpha, A, B, RET); }
    void cg_blas_matmul(std::size_t m, std::size_t n, std::size_t k, double alpha, const device_ptr_type<double> &A, const device_ptr_type<double> &B, device_ptr_type<double> &RET) const final { this->cg_blas_matmul_impl(m, n, k, alpha, A, B, RET); }
    template <typename real_type>
    void cg_blas_matmul_impl(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type<real_type> &A, const device_ptr_type<real_type> &B, device_ptr_type<real_type> &RET) const;

    void cg_blas_matmul2(std::size_t m, std::size_t n, std::size_t k, float alpha, const device_ptr_type<float> &A, const device_ptr_type<float> &B, float beta, const device_ptr_type<float> &C, device_ptr_type<float> &RET) const final { this->cg_blas_matmul_impl2(m, n, k, alpha, A, B, beta, C, RET); }
    void cg_blas_matmul2(std::size_t m, std::size_t n, std::size_t k, double alpha, const device_ptr_type<double> &A, const device_ptr_type<double> &B, double beta, const device_ptr_type<double> &C, device_ptr_type<double> &RET) const final { this->cg_blas_matmul_impl2(m, n, k, alpha, A, B, beta, C, RET); }
    template <typename real_type>
    void cg_blas_matmul_impl2(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type<real_type> &A, const device_ptr_type<real_type> &B, real_type beta, const device_ptr_type<real_type> &C, device_ptr_type<real_type> &RET) const;


  private:
    /**
     * @brief Initialize all important states related to the CUDA backend.
     * @param[in] target the target platform to use
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    void init(target_platform target);
};

}  // namespace cuda

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the CUDA backend are available.
 */
template <>
struct csvm_backend_exists<cuda::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_CUDA_CSVM_HPP_