/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the SYCL backend with AdaptiveCpp as SYCL implementation.
 */

#ifndef PLSSVM_TESTS_BACKENDS_SYCL_ADAPTIVECPP_MOCK_ADAPTIVECPP_CSVM_HPP_
#define PLSSVM_TESTS_BACKENDS_SYCL_ADAPTIVECPP_MOCK_ADAPTIVECPP_CSVM_HPP_
#pragma once

#include "plssvm/backends/execution_range.hpp"        // plssvm::detail::dim_type
#include "plssvm/backends/SYCL/AdaptiveCpp/csvm.hpp"  // plssvm::adaptivecpp::csvm

#include "gmock/gmock.h"  // MOCK_METHOD, ON_CALL, ::testing::Return

#include <cstddef>  // std::size_t
#include <utility>  // std::forward

/**
 * @brief GTest mock class for the SYCL CSVM using AdaptiveCpp as SYCL implementation.
 * @tparam mock_grid_size `true` if the `plssvm::adaptivecpp::csvm::get_max_grid_size()` function should be mocked, otherwise `false`
 */
template <bool mock_grid_size>
class mock_adaptivecpp_csvm final : public plssvm::adaptivecpp::csvm {
    using base_type = plssvm::adaptivecpp::csvm;

  public:
    using base_type::device_ptr_type;

    template <typename... Args>
    explicit mock_adaptivecpp_csvm(Args &&...args) :
        base_type{ std::forward<Args>(args)... } {
        this->fake_functions();
    }

    MOCK_METHOD((plssvm::detail::dim_type), get_max_grid_size, (const std::size_t), (const, override));

    // make protected member functions public
    using base_type::assemble_kernel_matrix;
    using base_type::blas_level_3;
    using base_type::get_device_memory;
    using base_type::get_max_work_group_size;
    using base_type::num_available_devices;

    using base_type::predict_values;

    using base_type::conjugate_gradients;
    using base_type::perform_dimensional_reduction;
    using base_type::run_assemble_kernel_matrix_implicit_blas_level_3;
    using base_type::run_blas_level_3;
    using base_type::solve_lssvm_system_of_linear_equations;

    using base_type::get_max_mem_alloc_size;

    using base_type::run_assemble_kernel_matrix_explicit;
    using base_type::run_blas_level_3_kernel_explicit;
    using base_type::run_inplace_matrix_addition;
    using base_type::run_inplace_matrix_scale;
    using base_type::run_predict_kernel;
    using base_type::run_w_kernel;

    using base_type::data_distribution_;
    using base_type::devices_;

  private:
    /*
     * @brief Fake the plssvm::adaptivecpp::csvm::get_max_grid_size() function if requested.
     */
    void fake_functions() const {
        if constexpr (mock_grid_size) {
            // mock the function using hardcoded maximum grid sizes
            ON_CALL(*this, get_max_grid_size).WillByDefault(::testing::Return(plssvm::detail::dim_type{ std::size_t{ 4 }, std::size_t{ 4 }, std::size_t{ 4 } }));
        } else {
            // use the actual real implementation otherwise
            ON_CALL(*this, get_max_grid_size).WillByDefault([this](const std::size_t device_id) { return base_type::get_max_grid_size(device_id); });
        }
    }
};

#endif  // PLSSVM_TESTS_BACKENDS_SYCL_ADAPTIVECPP_MOCK_ADAPTIVECPP_CSVM_HPP_
