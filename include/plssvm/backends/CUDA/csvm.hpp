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
#include <variant>

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


    device_ptr_type<float> run_predict_kernel(const ::plssvm::detail::parameter<float> &params, const device_ptr_type<float> &w_d, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &rho_d, const device_ptr_type<float> &sv_d, const device_ptr_type<float> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const final { return this->run_predict_kernel_impl(params, w_d, alpha_d, rho_d, sv_d, predict_points_d, num_classes, num_sv, num_predict_points, num_features); }
    device_ptr_type<double> run_predict_kernel(const ::plssvm::detail::parameter<double> &params, const device_ptr_type<double> &w_d, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &rho_d, const device_ptr_type<double> &sv_d, const device_ptr_type<double> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const final { return this->run_predict_kernel_impl(params, w_d, alpha_d, rho_d, sv_d, predict_points_d, num_classes, num_sv, num_predict_points, num_features); }
    template <typename real_type>
    device_ptr_type<real_type> run_predict_kernel_impl(const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &rho_d, const device_ptr_type<real_type> &sv_d, const device_ptr_type<real_type> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const;

    device_ptr_type<float> run_w_kernel(const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const final { return this->run_w_kernel_impl(alpha_d, sv_d, num_classes, num_sv, num_features); }
    device_ptr_type<double> run_w_kernel(const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const final { return this->run_w_kernel_impl(alpha_d, sv_d, num_classes, num_sv, num_features); }
    template <typename real_type>
    device_ptr_type<real_type> run_w_kernel_impl(const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const;


    void setup_data_on_devices(const aos_matrix<float> &A) override { this->setup_data_on_devices_impl(A); }
    void setup_data_on_devices(const aos_matrix<double> &A) override { this->setup_data_on_devices_impl(A); }
    template <typename real_type>
    void setup_data_on_devices_impl(const aos_matrix<real_type> &A);

    [[nodiscard]] std::vector<float> generate_q2(const ::plssvm::detail::parameter<float> &params, const std::size_t num_rows_reduced, const std::size_t num_features) override { return this->generate_q2_impl(params, num_rows_reduced, num_features); }
    [[nodiscard]] std::vector<double> generate_q2(const ::plssvm::detail::parameter<double> &params, const std::size_t num_rows_reduced, const std::size_t num_features) override { return this->generate_q2_impl(params, num_rows_reduced, num_features); }
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> generate_q2_impl(const ::plssvm::detail::parameter<real_type> &params, const std::size_t num_rows_reduced, const std::size_t num_features);

    void assemble_kernel_matrix_explicit(const ::plssvm::detail::parameter<float> &params, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<float> &q_red, float QA_cost) override { this->assemble_kernel_matrix_explicit_impl(params, num_rows_reduced, num_features, q_red, QA_cost); }
    void assemble_kernel_matrix_explicit(const ::plssvm::detail::parameter<double> &params, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<double> &q_red, double QA_cost) override { this->assemble_kernel_matrix_explicit_impl(params, num_rows_reduced, num_features, q_red, QA_cost); }
    template <typename real_type>
    void assemble_kernel_matrix_explicit_impl(const ::plssvm::detail::parameter<real_type> &params, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<real_type> &q_red, real_type QA_cost);

    [[nodiscard]] aos_matrix<float> kernel_matrix_matmul_explicit(const aos_matrix<float> &vec) override { return this->kernel_matrix_matmul_explicit_impl(vec); }
    [[nodiscard]] aos_matrix<double> kernel_matrix_matmul_explicit(const aos_matrix<double> &vec) override { return this->kernel_matrix_matmul_explicit_impl(vec); }
    template <typename real_type>
    [[nodiscard]] aos_matrix<real_type> kernel_matrix_matmul_explicit_impl(const aos_matrix<real_type> &vec);

    void clear_data_on_devices(float) override { this->clear_data_on_devices_impl(float{}); }
    void clear_data_on_devices(double) override { this->clear_data_on_devices_impl(double{}); }
    template <typename real_type>
    void clear_data_on_devices_impl(real_type);


  private:
    /**
     * @brief Initialize all important states related to the CUDA backend.
     * @param[in] target the target platform to use
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    void init(target_platform target);

    // the input data set
    std::variant<device_ptr_type<float>, device_ptr_type<double>> data_d_;
    std::variant<device_ptr_type<float>, device_ptr_type<double>> data_last_d_;
    std::variant<device_ptr_type<float>, device_ptr_type<double>> q_d_;

    // the explicit kernel matrix
    std::variant<device_ptr_type<float>, device_ptr_type<double>> explicit_kernel_matrix_;
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