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

/**
 * @brief GTest mock class for the OpenCL CSVM.
 */
class mock_opencl_csvm : public plssvm::opencl::csvm {
    using base_type = plssvm::opencl::csvm;

  public:
    using base_type::device_ptr_type;

    template <typename... Args>
    explicit mock_opencl_csvm(Args &&...args) :
        base_type{ std::forward<Args>(args)... } {}

    // make protected member functions public
    using base_type::assemble_kernel_matrix;
    using base_type::blas_level_3;
    using base_type::get_device_memory;
    using base_type::get_max_work_group_size;
    using base_type::setup_data_on_devices;

    using base_type::predict_values;

    using base_type::conjugate_gradients;
    using base_type::perform_dimensional_reduction;
    using base_type::run_blas_level_3;
    using base_type::solve_lssvm_system_of_linear_equations;

    using base_type::get_max_mem_alloc_size;

    using base_type::run_assemble_kernel_matrix_explicit;
    using base_type::run_blas_level_3_kernel_explicit;
    using base_type::run_predict_kernel;
    using base_type::run_w_kernel;

    using base_type::devices_;
};

#endif  // PLSSVM_TESTS_BACKENDS_OPENCL_MOCK_OPENCL_CSVM_HPP_