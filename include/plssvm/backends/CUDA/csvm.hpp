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

#pragma once

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/gpu_csvm.hpp"                // plssvm::detail::gpu_csvm

#include <cstddef>  // std::size_t

namespace plssvm {

// forward declare parameter class
template <typename T>
class parameter;

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace cuda {

/**
 * @brief A C-SVM implementation using CUDA as backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : public ::plssvm::detail::gpu_csvm<T, ::plssvm::cuda::detail::device_ptr<T>, int> {
  protected:
    // protected for the test mock class
    /// The template base type of the CUDA C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<T, ::plssvm::cuda::detail::device_ptr<T>, int>;

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

    using base_type::boundary_size_;
    using base_type::dept_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;

    /// The type of the CUDA device pointer.
    using device_ptr_type = typename base_type::device_ptr_type;
    /// The type of the CUDA device queue.
    using queue_type = typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the CUDA backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::cuda::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::gpu_nvidia
     * @throws plssvm::cuda::backend_exception if the plssvm::target_platform::gpu_nvidia target isn't available
     * @throws plssvm::cuda::backend_exception if no CUDA devices could be found
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Wait for all operations on all CUDA devices to finish.
     * @details Terminates the program, if any exception is thrown.
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
    void run_q_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &q_d, std::size_t num_features) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, real_type add, std::size_t num_features) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, std::size_t num_features) final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, std::size_t num_predict_points) final;
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace cuda
}  // namespace plssvm
