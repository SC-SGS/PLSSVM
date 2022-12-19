/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the OpenCL backend.
 */

#ifndef PLSSVM_TESTS_BACKENDS_OPENCL_MOCK_OPENCL_CSVM_HPP_
#define PLSSVM_TESTS_BACKENDS_OPENCL_MOCK_OPENCL_CSVM_HPP_
#pragma once

#include "plssvm/backends/OpenCL/csvm.hpp"  // plssvm::opencl::csvm
#include "plssvm/parameter.hpp"             // plssvm::parameter

/**
 * @brief GTest mock class for the OpenCL CSVM.
 */
class mock_opencl_csvm : public plssvm::opencl::csvm {
    using base_type = plssvm::opencl::csvm;

  public:
    using base_type::device_ptr_type;

    // constructors necessary to compile the correct kernel functions
    mock_opencl_csvm() = default;
    explicit mock_opencl_csvm(plssvm::parameter params) :
        base_type{ params } {}

    // make protected member functions public
    using base_type::calculate_w;
    using base_type::device_reduction;
    using base_type::generate_q;
    using base_type::predict_values;
    using base_type::run_device_kernel;
    using base_type::select_num_used_devices;
    using base_type::setup_data_on_device;
    using base_type::solve_system_of_linear_equations;

    using base_type::devices_;
};

#endif  // PLSSVM_TESTS_BACKENDS_OPENCL_MOCK_OPENCL_CSVM_HPP_