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
    using base_type::alpha_;
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_;
    using base_type::degree_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;
    using base_type::target_;

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
     * @brief Construct an new C-SVM using the OpenCL backend explicitly specifying all necessary parameters.
     * @param[in] target the target platform
     * @param[in] kernel the type of the kernel function
     * @param[in] degree parameter used in the polynomial kernel function
     * @param[in] gamma parameter used in the polynomial and rbf kernel functions
     * @param[in] coef0 parameter use din the polynomial kernel function
     * @param[in] cost parameter of the C-SVM
     * @param[in] epsilon error tolerance in the CG algorithm
     * @param[in] print_info if `true` additional information will be printed during execution
     */
    csvm(target_platform target, kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    /**
     * @brief Wait for all operations on all devices to finish.
     * @details Terminates the program, if any exceptions are thrown.
     */
    ~csvm() override;

    // std::vector<real_type> predict(real_type *, size_type, size_type) override;  // TODO: implement

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    void load_w() override {}  // TODO: implement correctly

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the OpenCL @p device.
     * @param[in] device the OpenCL device to run the kernel on
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[inout] r_d the result vector
     * @param[in] x_d the `x` vector
     * @param[in] data_d the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(size_type device, const detail::device_ptr<real_type> &q_d, detail::device_ptr<real_type> &r_d, const detail::device_ptr<real_type> &x_d, const detail::device_ptr<real_type> &data_d, int add);
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each devices.
     * @param[inout] buffer_d the data to gather
     * @param[inout] buffer the reduces data
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
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::opencl