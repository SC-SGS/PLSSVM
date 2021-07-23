/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief MOCK class for the C-SVM class using the OpenCL backend.
 */
#pragma once

#include "plssvm/backends/OpenCL/DevicePtrOpenCL.hpp"  // opencl::DevicePtrOpenCL
#include "plssvm/backends/OpenCL/csvm.hpp"             // plssvm::opencl::csvm
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the OpenCL CSVM.
 * @tparam T the type of the data
 */
template <typename T>
class mock_opencl_csvm : public plssvm::opencl::csvm<T> {
    using base_type = plssvm::opencl::csvm<T>;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit mock_opencl_csvm(const plssvm::parameter<T> &params) :
        base_type{ params } {}
    explicit mock_opencl_csvm(const plssvm::kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
        base_type{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // make non-virtual functions publicly visible
    using base_type::generate_q;
    using base_type::learn;
    using base_type::manager;
    using base_type::setup_data_on_device;

    // getter for internal variables
    std::vector<opencl::DevicePtrOpenCL<real_type>> &get_device_data() { return data_cl; }

  private:
    using base_type::data_cl;
};
