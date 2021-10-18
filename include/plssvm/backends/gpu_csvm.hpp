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

#include "plssvm/csvm.hpp"                    // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"  // plssvm::detail::execution_range
#include "plssvm/parameter.hpp"               // plssvm::parameter

#include <vector>  // std::vector

namespace plssvm::detail {

/**
 * @brief A C-SVM class for all GPU backends to reduce code duplication. Implements all virtual functions defined in `plssvm::csvm`.
 * @tparam T the type of the data
 * @tparam device_ptr_t the type of the device pointer (dependent on the used backend)
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 */
template <typename T, typename device_ptr_t, typename queue_t>
class gpu_csvm : public csvm<T> {
  protected:
    /// The template base type of the C-SVM class.
    using base_type = ::plssvm::csvm<T>;

    using base_type::alpha_ptr_;
    using base_type::bias_;
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_ptr_;
    using base_type::degree_;
    using base_type::epsilon_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;
    using base_type::target_;
    using base_type::value_ptr_;
    using base_type::w_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;

    /// The type of the device pointer (dependent on the used backend).
    using device_ptr_type = device_ptr_t;
    /// The type of the device queue (dependent on the used backend).
    using queue_type = queue_t;

    /**
     * @brief Construct a new C-SVM using any GPU backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit gpu_csvm(const parameter<T> &params);

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~gpu_csvm() = default;

    //*************************************************************************************************************************************//
    //                                                functions inherited from plssvm::csvm                                                //
    //*************************************************************************************************************************************//
    /**
     * @copydoc plssvm::csvm::predict(const std::vector<std::vector<real_type>>&)
     */
    [[nodiscard]] std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) final;

  protected:
    /**
     * @copydoc plssvm::csvm::setup_data_on_device
     */
    void setup_data_on_device() final;
    /**
     * @copydoc plssvm::csvm::generate_q
     */
    [[nodiscard]] std::vector<real_type> generate_q() final;
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, std::size_t imax, real_type eps, const std::vector<real_type> &q) final;

    /**
     * @copydoc plssvm::csvm::update_w
     */
    void update_w() final;

  protected:
    /**
     * @brief Run the SVM kernel the @p device.
     * @param[in] device the OpenCL device to run the kernel on
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[in,out] r_d the result vector
     * @param[in] x_d the `x` vector
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(std::size_t device, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, real_type add);
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each devices.
     * @param[in,out] buffer_d the data to gather
     * @param[in,out] buffer the reduces data
     */
    void device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer);

    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Synchronize the device denoted by @p queue.
     * @param[in,out] queue the queue denoting the device to synchronize
     */
    virtual void device_synchronize(queue_type &queue) = 0;
    /**
     * @brief Run the GPU kernel filling the `q_` vector.
     * @param[in] device the device on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[out] q_d the `q` vector to fill
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_q_kernel(std::size_t device, const detail::execution_range<std::size_t> &range, device_ptr_type &q_d, std::size_t num_features) = 0;
    /**
     * @brief Run the main GPU kernel used in the CG algorithm.
     * @param[in] device the device on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[in] q_d the `q` vector
     * @param[in,out] r_d the result vector
     * @param[in] x_d the right-hand side of the equation
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] num_features number of features used for the calculation in the @p device
     */
    virtual void run_svm_kernel(std::size_t device, const detail::execution_range<std::size_t> &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, real_type add, std::size_t num_features) = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) the calculate the `w` vector used to speedup the prediction when using the linear kernel function.
     * @param[in] device the device on which the kernel should be executed
     * @param[in] range the execution range used to launch the
     * @param[out] w_d the `w` vector to fill, used to speed-up the prediction when using the linear kernel
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_w_kernel(std::size_t device, const detail::execution_range<std::size_t> &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, std::size_t num_features) = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) to predict the new data points @p point_d.
     * @param[in] range the execution range used to launch the kernel
     * @param[out] out_d the calculated prediction
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] point_d the data points to predict
     * @param[in] num_predict_points the number of data points to predict
     */
    virtual void run_predict_kernel(const detail::execution_range<std::size_t> &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, std::size_t num_predict_points) = 0;

    //*************************************************************************************************************************************//
    //                                             internal variables specific to GPU backends                                             //
    //*************************************************************************************************************************************//
    /// The number of data points excluding the last data point.
    std::size_t dept_{};
    /// The boundary size used to remove boundary condition checks inside the kernels.
    std::size_t boundary_size_{};
    /// The number of rows to calculate including the boundary values.v
    std::size_t num_rows_{};
    /// The number of columns in the data matrix (= the number of features per data point).
    std::size_t num_cols_{};

    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
    /// The data saved across all devices.
    std::vector<device_ptr_type> data_d_{};
    /// The last row of the data matrix.
    std::vector<device_ptr_type> data_last_d_{};
};

}  // namespace plssvm::detail