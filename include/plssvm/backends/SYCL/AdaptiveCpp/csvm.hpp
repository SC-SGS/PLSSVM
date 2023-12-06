/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a base C-SVM used for the different SYCL backends using AdaptiveCpp as SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_ADAPTIVECPP_CSVM_HPP_
#define PLSSVM_BACKENDS_SYCL_ADAPTIVECPP_CSVM_HPP_
#pragma once

#include "plssvm/backends/SYCL/AdaptiveCpp/detail/device_ptr.hpp"  // plssvm::adaptivecpp::detail::device_ptr
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/queue.hpp"       // plssvm::adaptivecpp::detail::queue (PImpl)

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/backends/gpu_csvm.hpp"                     // plssvm::detail::gpu_csvm
#include "plssvm/constants.hpp"                             // plssvm::real_type
#include "plssvm/detail/memory_size.hpp"                    // plssvm::detail::memory_size
#include "plssvm/detail/type_traits.hpp"                    // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t
#include "plssvm/parameter.hpp"                             // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "igor/igor.hpp"  // igor::parser

#include <cstddef>      // std::size_t
#include <type_traits>  // std::is_same_v, std::true_type
#include <utility>      // std::forward

namespace plssvm {

namespace adaptivecpp {

/**
 * @brief A C-SVM implementation using AdaptiveCpp as SYCL backend.
 */
class csvm : public ::plssvm::detail::gpu_csvm<detail::device_ptr, detail::queue> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the AdaptiveCpp SYCL C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<detail::device_ptr, detail::queue>;

    using base_type::devices_;

  public:
    using base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the SYCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::adaptivecpp::backend_exception if the requested target is not available
     * @throws plssvm::adaptivecpp::backend_exception if no device for the requested target was found
     */
    explicit csvm(parameter params = {});
    /**
     * @brief Construct a new C-SVM using the SYCL backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVM
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::adaptivecpp::backend_exception if the requested target is not available
     * @throws plssvm::adaptivecpp::backend_exception if no device for the requested target was found
     */
    explicit csvm(target_platform target, parameter params = {});

    /**
     * @brief Construct a new C-SVM using the SYCL backend and the optionally provided @p named_args.
     * @details Additionally sets the SYCL specific kernel invocation type.
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::adaptivecpp::backend_exception if the requested target is not available
     * @throws plssvm::adaptivecpp::backend_exception if no device for the requested target was found
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
     * @throws plssvm::adaptivecpp::backend_exception if the requested target is not available
     * @throws plssvm::adaptivecpp::backend_exception if no device for the requested target was found
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_sycl_parameter_named_args_v<Args...>)>
    explicit csvm(const target_platform target, Args &&...named_args) :
        base_type{ named_args... } {
        // check igor parameter
        igor::parser parser{ std::forward<Args>(named_args)... };

        // check whether a specific SYCL kernel invocation type has been requested
        if constexpr (parser.has(sycl_kernel_invocation_type)) {
            // compile time check: the value must have the correct type
            invocation_type_ = ::plssvm::detail::get_value_from_named_parameter<sycl::kernel_invocation_type>(parser, sycl_kernel_invocation_type);
        }
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
     * @brief Wait for all operations in all [`sycl::queue`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:interface.queue.class) to finish.
     * @details Terminates the program, if any asynchronous exception is thrown.
     */
    ~csvm() override;

    /**
     * @brief Return the kernel invocation type used in this SYCL SVM.
     * @return the SYCL kernel invocation type (`[[nodiscard]]`)
     */
    [[nodiscard]] sycl::kernel_invocation_type get_kernel_invocation_type() const noexcept { return invocation_type_; }

  protected:
    /**
     * @brief Initialize all important states related to the SYCL backend.
     * @param[in] target the target platform to use
     * @throws plssvm::adaptivecpp::backend_exception if the requested target is not available
     * @throws plssvm::adaptivecpp::backend_exception if no device for the requested target was found
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

    /// The SYCL kernel invocation type for the svm kernel.
    sycl::kernel_invocation_type invocation_type_{ sycl::kernel_invocation_type::automatic };
};

}  // namespace adaptivecpp

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the SYCL backend with AdaptiveCpp as SYCL implementation are available.
 */
template <>
struct csvm_backend_exists<adaptivecpp::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_SYCL_ADAPTIVECPP_CSVM_HPP_