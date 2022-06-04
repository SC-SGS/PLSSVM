/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the OpenCL backend.
 */

#pragma once

#include "plssvm/kernel_types.hpp"
#include "plssvm/target_platforms.hpp"
#include "plssvm/parameter.hpp"
#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/context.hpp"        // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"     // plssvm::opencl::detail::device_ptr
#include "plssvm/backends/gpu_csvm.hpp"                     // plssvm::detail::gpu_csvm

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm {

// forward declare parameter class
//template <typename T>
//class parameter;

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace opencl {

/**
 * @brief A C-SVM implementation using OpenCL as backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : public ::plssvm::detail::gpu_csvm<T, ::plssvm::opencl::detail::device_ptr<T>, ::plssvm::opencl::detail::command_queue> {
  protected:
    // protected for test MOCK class
    /// The template base type of the OpenCL C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<T, ::plssvm::opencl::detail::device_ptr<T>, ::plssvm::opencl::detail::command_queue>;

    using base_type::devices_;

  public:
    using typename base_type::real_type;
    using typename base_type::size_type;
    using typename base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the OpenCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::opencl::backend_exception if the requested plssvm::target_platform isn't available
     * @throws plssvm::opencl::backend_exception if no possible OpenCL devices could be found
     */
    explicit csvm(target_platform target, parameter<real_type> params = {});

    template <typename... Args>
    csvm(target_platform target, kernel_type kernel, Args&&... named_args) : base_type{ kernel, std::forward<Args>(named_args)... } {
        this->init(target, kernel);
    }

    /**
     * @brief Wait for all operations on all devices to finish.
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
    void run_q_kernel(size_type device, const ::plssvm::detail::execution_range &range, const parameter<real_type> &params, device_ptr_type &q_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_data_points_padded, size_type num_features) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(size_type device, const ::plssvm::detail::execution_range &range, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const device_ptr_type &data_d, real_type QA_cost, real_type add, size_type num_data_points_padded, size_type num_features) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(size_type device, const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_data_points, size_type num_features) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range &range, const parameter<real_type> &params, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_support_vectors, size_type num_predict_points, size_type num_features) const final;

    /// The available OpenCL contexts for the current target platform with the associated devices.
    std::vector<detail::context> contexts_{};

  private:
    void init(target_platform target, kernel_type kernel);
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace opencl
}  // namespace plssvm