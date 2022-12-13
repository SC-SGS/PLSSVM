/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the SYCL backend with hipSYCL as SYCL implementation.
 */

#ifndef PLSSVM_TESTS_BACKENDS_SYCL_HIPSYCL_MOCK_HIPSYCL_CSVM_HPP_
#define PLSSVM_TESTS_BACKENDS_SYCL_HIPSYCL_MOCK_HIPSYCL_CSVM_HPP_
#pragma once

#include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"  // plssvm::hipsycl::csvm

/**
 * @brief GTest mock class for the SYCL CSVM using hipSYCL as SYCL implementation.
 */
class mock_hipsycl_csvm final : public plssvm::hipsycl::csvm {
    using base_type = plssvm::hipsycl::csvm;

  public:
    using base_type::device_ptr_type;

    mock_hipsycl_csvm() = default;
    explicit mock_hipsycl_csvm(plssvm::parameter params) :
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

#endif  // PLSSVM_TESTS_BACKENDS_SYCL_HIPSYCL_MOCK_HIPSYCL_CSVM_HPP_