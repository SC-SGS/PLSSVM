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
#include "plssvm/detail/simple_any.hpp"                // plssvm::detail::simple_any
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
     * @copydoc plssvm::csvm::get_device_memory
     */
    unsigned long long get_device_memory() const final;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_assemble_kernel_matrix_explicit
     */
    [[nodiscard]] device_ptr_type<float> run_assemble_kernel_matrix_explicit(const ::plssvm::detail::parameter<float> &params, const device_ptr_type<float> & data_d, const device_ptr_type<float> &q_red_d, float QA_cost) const final { return this->run_assemble_kernel_matrix_explicit_impl(params, data_d, q_red_d, QA_cost); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_assemble_kernel_matrix_explicit
     */
    [[nodiscard]] device_ptr_type<double> run_assemble_kernel_matrix_explicit(const ::plssvm::detail::parameter<double> &params, const device_ptr_type<double> &data_d, const device_ptr_type<double> &q_red_d, double QA_cost) const final { return this->run_assemble_kernel_matrix_explicit_impl(params, data_d, q_red_d, QA_cost); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_assemble_kernel_matrix_explicit
     */
    template <typename real_type>
    [[nodiscard]] device_ptr_type<real_type> run_assemble_kernel_matrix_explicit_impl(const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &q_red_d, real_type QA_cost) const;

    /**
     * @copydoc plssvm::detail::gpu_csvm::run_gemm_kernel_explicit
     */
    void run_gemm_kernel_explicit(std::size_t m, std::size_t n, std::size_t k, float alpha, const device_ptr_type<float> &A_d, const device_ptr_type<float> &B_d, const float beta, device_ptr_type<float> &C_d) const final { this->run_gemm_kernel_explicit_impl(m, n, k, alpha, A_d, B_d, beta, C_d); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_gemm_kernel_explicit
     */
    void run_gemm_kernel_explicit(std::size_t m, std::size_t n, std::size_t k, double alpha, const device_ptr_type<double> &A_d, const device_ptr_type<double> &B_d, const double beta, device_ptr_type<double> &C_d) const final { this->run_gemm_kernel_explicit_impl(m, n, k, alpha, A_d, B_d, beta, C_d); }
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_gemm_kernel_explicit
     */
    template <typename real_type>
    void run_gemm_kernel_explicit_impl(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type<real_type> &A_d, const device_ptr_type<real_type> &B_d, const real_type beta, device_ptr_type<real_type> &C_d) const;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    device_ptr_type<float> run_predict_kernel(const ::plssvm::detail::parameter<float> &params, const device_ptr_type<float> &w_d, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &rho_d, const device_ptr_type<float> &sv_d, const device_ptr_type<float> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const final { return this->run_predict_kernel_impl(params, w_d, alpha_d, rho_d, sv_d, predict_points_d, num_classes, num_sv, num_predict_points, num_features); }
    device_ptr_type<double> run_predict_kernel(const ::plssvm::detail::parameter<double> &params, const device_ptr_type<double> &w_d, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &rho_d, const device_ptr_type<double> &sv_d, const device_ptr_type<double> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const final { return this->run_predict_kernel_impl(params, w_d, alpha_d, rho_d, sv_d, predict_points_d, num_classes, num_sv, num_predict_points, num_features); }
    template <typename real_type>
    device_ptr_type<real_type> run_predict_kernel_impl(const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &rho_d, const device_ptr_type<real_type> &sv_d, const device_ptr_type<real_type> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const;

    device_ptr_type<float> run_w_kernel(const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const final { return this->run_w_kernel_impl(alpha_d, sv_d, num_classes, num_sv, num_features); }
    device_ptr_type<double> run_w_kernel(const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const final { return this->run_w_kernel_impl(alpha_d, sv_d, num_classes, num_sv, num_features); }
    template <typename real_type>
    device_ptr_type<real_type> run_w_kernel_impl(const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const;

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