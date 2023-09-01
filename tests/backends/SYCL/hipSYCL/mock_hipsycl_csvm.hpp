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

    template <typename... Args>
    explicit mock_hipsycl_csvm(Args &&...args) :
        base_type{ std::forward<Args>(args)... } {}

    // make protected member functions public
    using base_type::assemble_kernel_matrix;
    using base_type::blas_level_3;
    using base_type::device_reduction;
    using base_type::predict_values;
    using base_type::select_num_used_devices;
    using base_type::setup_data_on_devices;
    //    using base_type::device_synchronize;
    //    using base_type::run_assemble_kernel_matrix_explicit;
    //    using base_type::run_blas_level_3_kernel_explicit;
    //    using base_type::run_w_kernel;
    //    using base_type::run_predict_kernel;

    using base_type::devices_;
};

#endif  // PLSSVM_TESTS_BACKENDS_SYCL_HIPSYCL_MOCK_HIPSYCL_CSVM_HPP_