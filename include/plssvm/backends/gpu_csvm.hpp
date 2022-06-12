/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends using a GPU. Used for code duplication reduction.
 */

#pragma once

#include "plssvm/csvm.hpp"       // plssvm::csvm
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <vector>   // std::vector

namespace plssvm::detail {

// forward declare execution_range class
class execution_range;

/**
 * @brief A C-SVM implementation for all GPU backends to reduce code duplication.
 * @details Implements all virtual functions defined in plssvm::csvm. The GPU backends only have to implement the actual kernel launches.
 * @tparam T the type of the data
 * @tparam device_ptr_t the type of the device pointer (dependent on the used backend)
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 */
template <typename T, typename device_ptr_t, typename queue_t>
class gpu_csvm : public csvm<T> {
  protected:
    /// The template base type of the C-SVM class.
    using base_type = ::plssvm::csvm<T>;

  public:
    using typename base_type::real_type;
    using typename base_type::size_type;
    /// The type of the device pointer (dependent on the used backend).
    using device_ptr_type = device_ptr_t;
    /// The type of the device queue (dependent on the used backend).
    using queue_type = queue_t;

    /**
     * @brief Construct a new C-SVM using any GPU backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     */
    explicit gpu_csvm(parameter<real_type> params = {});
    template <typename... Args>
    explicit gpu_csvm(kernel_type kernel, Args&&... args) : base_type{ kernel, std::forward<Args>(args)... } { }

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~gpu_csvm() = default;

    [[nodiscard]] size_type num_available_devices() const noexcept {
        return devices_.size();
    }

  protected:
    /**
     * @copydoc plssvm::csvm::setup_data_on_device
     */
    [[nodiscard]] std::tuple<std::vector<device_ptr_type>, std::vector<device_ptr_type>, std::vector<size_type>> setup_data_on_device(const std::vector<std::vector<real_type>> &data, size_type num_data_points, size_type num_features, size_type boundary_size, size_type num_used_devices) const;
    /**
     * @copydoc plssvm::csvm::generate_q
     */
    [[nodiscard]] std::vector<real_type> generate_q(const parameter<real_type> &params, const std::vector<device_ptr_type> &data_d, const std::vector<device_ptr_type> &data_last_d, size_type num_data_points, const std::vector<size_type> &feature_ranges, size_type boundary_size, size_type num_used_devices) const;
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    [[nodiscard]] std::pair<std::vector<real_type>, real_type> solve_system_of_linear_equations(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, real_type eps, size_type max_iter) const final;
    /**
     * @copydoc plssvm::csvm::update_w
     */
    [[nodiscard]] std::vector<real_type> calculate_w(const std::vector<device_ptr_type> &data_d, const std::vector<device_ptr_type> &data_last_d, const std::vector<device_ptr_type> &alpha_d, size_type num_data_points, const std::vector<size_type> &feature_ranges, size_type num_used_devices) const;

    [[nodiscard]] std::vector<real_type> predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const final;


    /**
     * @brief Run the SVM kernel on the GPU denoted by the @p device ID.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[in,out] r_d the result vector
     * @param[in] x_d the right-hand side of the equation
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(size_type device, const std::vector<size_type> &feature_ranges, const parameter<real_type> &params, device_ptr_type &r_d, const device_ptr_type &data_d, const device_ptr_type &q_d, real_type QA_cost, const device_ptr_type &x_d, real_type add, size_type dept, size_type boundary_size) const;
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each device.
     * @param[in,out] buffer_d the data to gather
     * @param[in,out] buffer the reduced data
     */
    void device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) const;

    [[nodiscard]] size_type select_num_used_devices(kernel_type kernel, size_type num_features) const noexcept;

    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Synchronize the device denoted by @p queue.
     * @param[in,out] queue the queue denoting the device to synchronize
     */
    virtual void device_synchronize(const queue_type &queue) const = 0;
    /**
     * @brief Run the GPU kernel filling the `q` vector.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[out] q_d the `q` vector to fill
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_q_kernel(size_type device, const detail::execution_range &range, const parameter<real_type> &params, device_ptr_type &q_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_data_points_padded, size_type num_features) const = 0;
    /**
     * @brief Run the main GPU kernel used in the CG algorithm.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[in] q_d the `q` vector
     * @param[in,out] r_d the result vector
     * @param[in] x_d the right-hand side of the equation
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] num_features number of features used for the calculation in the @p device
     */
    virtual void run_svm_kernel(size_type device, const detail::execution_range &range, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const device_ptr_type &data_d, real_type QA_cost, real_type add, size_type num_data_points_padded, size_type num_features) const = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) the calculate the `w` vector used to speed up the prediction when using the linear kernel function.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the
     * @param[out] w_d the `w` vector to fill, used to speed up the prediction when using the linear kernel
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_w_kernel(size_type device, const detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_data_points, size_type num_features) const = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) to predict the new data points @p point_d.
     * @param[in] range the execution range used to launch the kernel
     * @param[out] out_d the calculated prediction
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] point_d the data points to predict
     * @param[in] num_predict_points the number of data points to predict
     */
    virtual void run_predict_kernel(const detail::execution_range &range, const parameter<real_type> &params, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_support_vectors, size_type num_predict_points, size_type num_features) const = 0;


    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
};

}  // namespace plssvm::detail