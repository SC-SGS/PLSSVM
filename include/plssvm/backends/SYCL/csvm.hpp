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

#include "plssvm/backends/SYCL/detail/device_ptr.hpp"  // plssvm::sycl::detail::device_ptr
#include "plssvm/backends/gpu_csvm.hpp"                // plssvm::detail::gpu_csvm
#include "plssvm/detail/execution_range.hpp"           // plssvm::detail::execution_range
#include "plssvm/parameter.hpp"                        // plssvm::parameter

#include "sycl/sycl.hpp"  // sycl::queue

namespace plssvm::sycl {

/**
 * @brief The C-SVM class using the SYCL backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : public ::plssvm::detail::gpu_csvm<T, ::plssvm::sycl::detail::device_ptr<T>, ::sycl::queue> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the SYCL C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<T, ::plssvm::sycl::detail::device_ptr<T>, ::sycl::queue>;

    using base_type::coef0_;
    using base_type::cost_;
    using base_type::degree_;
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
    using base_type::w_d_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /// The type of the SYCL device pointer.
    using device_ptr_type = ::plssvm::sycl::detail::device_ptr<real_type>;
    /// The type of the SYCL device queue.
    using queue_type = ::sycl::queue;

    /**
     * @brief Construct a new C-SVM using the SYCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Wait for all operations in all [`sycl::queue`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:interface.queue.class) to finish.
     * @details Terminates the program, if any asynchronous exceptions are thrown.
     */
    ~csvm() override;

  protected:
    void device_synchronize(queue_type &queue) final;

    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    void run_q_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, device_ptr_type &q_d, const int col_range) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const int first_feature, const int last_feature) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(const ::plssvm::detail::execution_range<size_type> &range, const device_ptr_type &alpha_d) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range<size_type> &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const size_type num_predict_points) final;
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::sycl