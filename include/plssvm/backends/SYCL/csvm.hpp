/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the SYCL backend.
 */

#pragma once

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/constants.hpp"  // forward declaration and namespace alias
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/device_ptr.hpp" // plssvm::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::detail::device_ptr
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"                          // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/backends/gpu_csvm.hpp"                                             // plssvm::detail::gpu_csvm

#include <memory> // std::unique_ptr

namespace plssvm {

using namespace sycl_generic;

// forward declare parameter class
template <typename T>
class parameter;

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace @PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@ {

/**
 * @brief A C-SVM implementation using SYCL as backend.
 * @details If DPC++ is available, this class also exists in the `plssvm::dpcpp` namespace.
 *          If hipSYCL is available, this class also exists in the `plssvm::hipsycl` namespace.
 * @tparam T the type of the data
 */
template <typename T>
    class csvm : public ::plssvm::detail::gpu_csvm<T, detail::device_ptr<T>, std::unique_ptr<detail::sycl::queue>> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the SYCL C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<T, detail::device_ptr<T>, std::unique_ptr<detail::sycl::queue>>;

    using base_type::coef0_;
    using base_type::cost_;
    using base_type::degree_;
    using base_type::dept_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;
    using base_type::target_;

    using base_type::data_d_;
    using base_type::data_last_d_;
    using base_type::devices_;
    using base_type::num_cols_;
    using base_type::num_rows_;

  public:
    using typename base_type::real_type;
    using typename base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the SYCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::sycl::backend_exception if the requested plssvm::target_platform isn't available
     * @throws plssvm::sycl::backend_exception if no possible OpenCL devices could be found
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Wait for all operations in all [`sycl::queue`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:interface.queue.class) to finish.
     * @details Terminates the program, if any asynchronous exception is thrown.
     */
    ~csvm() override;

  protected:
    /**
     * @copydoc plssvm::detail::gpu_csvm::device_synchronize
     */
    void device_synchronize(queue_type &queue) final;

    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    void run_q_kernel(std::size_t device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, device_ptr_type &q_d, std::size_t num_features) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, std::size_t num_features) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(std::size_t device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, std::size_t num_features) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, std::size_t num_predict_points) final;

  private:
    /// The SYCL kernel invocation type for the svm kernel. Either nd_range or hierarchical.
    kernel_invocation_type invocation_type_;
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace @PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@
}  // namespace plssvm