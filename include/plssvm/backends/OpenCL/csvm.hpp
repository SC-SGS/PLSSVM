/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines a C-SVM using the OpenCL backend.
 */

#pragma once

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"     // plssvm::opencl::detail::device_ptr
#include "plssvm/backends/OpenCL/detail/kernel.hpp"         // plssvm::opencl::detail::kernel
#include "plssvm/csvm.hpp"                                  // plssvm::csvm
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/target_platform.hpp"                       // plssvm::target_platform

#include "CL/cl.h"  // cl_command_queue, cl_kernel

#include <vector>  // std::vector

namespace plssvm::opencl {

/**
 * @brief The C-SVM class using the OpenCL backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : public ::plssvm::csvm<T> {
  protected:
    // protected for test MOCK class
    /// The template base type of the OpenCL C-SVM class.
    using base_type = ::plssvm::csvm<T>;
    using base_type::alpha_ptr_;
    using base_type::bias_;
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_ptr_;
    using base_type::degree_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;
    using base_type::target_;
    using base_type::w_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /**
     * @brief Construct a new C-SVM using the OpenCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Wait for all operations on all devices to finish.
     * @details Terminates the program, if any exceptions are thrown.
     */
    ~csvm() override;

    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] points the data points to predict
     * @return a `std::vector<real_type>` filled with negative values for each prediction for a data point with the negative class and positive values otherwise ([[nodiscard]])
     */
    [[nodiscard]] virtual std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) override;

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    /**
     * @brief updates the `w_` vector to the current data and alpha values.
     */
    virtual void update_w();

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the OpenCL @p device.
     * @param[in] device the OpenCL device to run the kernel on
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[in,out] r_d the result vector
     * @param[in] x_d the `x` vector
     * @param[in] data_d the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(size_type device, const detail::device_ptr<real_type> &q_d, detail::device_ptr<real_type> &r_d, const detail::device_ptr<real_type> &x_d, const detail::device_ptr<real_type> &data_d, real_type add);
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each devices.
     * @param[in,out] buffer_d the data to gather
     * @param[in,out] buffer the reduces data
     */
    void device_reduction(std::vector<detail::device_ptr<real_type>> &buffer_d, std::vector<real_type> &buffer);

    /// The available/used OpenCL devices.
    std::vector<detail::command_queue> devices_{};
    /// The number of data points excluding the last data point.
    size_type dept_{};
    /// The boundary size used to remove boundary condition checks inside the kernels.
    size_type boundary_size_{};
    /// The number of rows to calculate including the boundary values.
    int num_rows_{};
    /// The number of columns in the data matrix (= the number of features per data point).
    int num_cols_{};
    /// The data saved across all devices.
    std::vector<detail::device_ptr<real_type>> data_d_{};
    /// The last row of the data matrix.
    std::vector<detail::device_ptr<real_type>> data_last_d_{};
    /// TODO:
    detail::device_ptr<real_type> w_d_{};

    /// OpenCL kernel for the generate q function compiled for each device.
    std::vector<detail::kernel> q_kernel_{};
    /// OpenCL kernel for the svm kernel function compiled for each device.
    std::vector<detail::kernel> svm_kernel_{};

    /// TODO
    std::vector<detail::kernel> kernel_w_kernel_{};
    std::vector<detail::kernel> predict_kernel_{};
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::opencl