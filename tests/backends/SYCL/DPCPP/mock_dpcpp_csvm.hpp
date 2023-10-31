/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the SYCL backend with DPC++ as SYCL implementation.
 */

#ifndef PLSSVM_TESTS_BACKENDS_SYCL_DPCPP_MOCK_DPCPP_CSVM_HPP_
#define PLSSVM_TESTS_BACKENDS_SYCL_DPCPP_MOCK_DPCPP_CSVM_HPP_
#pragma once

#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"  // plssvm::dpcpp::csvm

/**
 * @brief GTest mock class for the SYCL CSVM using DPC++ as SYCL implementation.
 */
class mock_dpcpp_csvm final : public plssvm::dpcpp::csvm {
    using base_type = plssvm::dpcpp::csvm;

  public:
    using base_type::device_ptr_type;

    template <typename... Args>
    explicit mock_dpcpp_csvm(Args &&...args) :
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

#endif  // PLSSVM_TESTS_BACKENDS_SYCL_DPCPP_MOCK_DPCPP_CSVM_HPP_