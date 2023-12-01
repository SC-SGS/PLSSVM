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
#include "plssvm/constants.hpp"                        // plssvm::real_type
#include "plssvm/detail/memory_size.hpp"               // plssvm::detail::memory_size
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include <cstddef>      // std::size_t
#include <type_traits>  // std::true_type
#include <utility>      // std::forward

namespace plssvm {

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
     * @brief Initialize all important states related to the CUDA backend.
     * @param[in] target the target platform to use
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA capable devices could be found
     */
    void init(target_platform target);

    /**
     * @copydoc plssvm::detail::gpu_csvm::device_synchronize
     */
    void device_synchronize(const queue_type &queue) const final;
    /**
     * @copydoc plssvm::csvm::get_device_memory
     */
    [[nodiscard]] ::plssvm::detail::memory_size get_device_memory() const final;
    /**
     * @copydoc plssvm::csvm::get_max_mem_alloc_size
     */
    [[nodiscard]] ::plssvm::detail::memory_size get_max_mem_alloc_size() const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::get_max_work_group_size
     */
    [[nodiscard]] std::size_t get_max_work_group_size() const final;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_assemble_kernel_matrix_explicit
     */
    [[nodiscard]] device_ptr_type run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_blas_level_3_kernel_explicit
     */
    void run_blas_level_3_kernel_explicit(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const final;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    [[nodiscard]] device_ptr_type run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    [[nodiscard]] device_ptr_type run_predict_kernel(const parameter &params, const device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const final;
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