/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic tests for all GPU backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_GPU_CSVM_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_GPU_CSVM_TESTS_HPP_
#pragma once

#include "plssvm/constants.hpp"                 // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set.hpp"                  // plssvm::data_set
#include "plssvm/detail/data_distribution.hpp"  // plssvm::detail::{triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/kernel_function_types.hpp"     // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                    // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/shape.hpp"                     // plssvm::shape

#include "tests/backends/ground_truth.hpp"  // ground_truth::{perform_dimensional_reduction, assemble_full_kernel_matrix, assemble_device_specific_kernel_matrix, device_specific_gemm,
                                            // calculate_device_specific_w, calculate_w, predict_values}
#include "tests/custom_test_macros.hpp"     // EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "tests/types_to_test.hpp"          // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"                // util::{redirect_output, construct_from_tuple, generate_random_matrix}

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_GT, EXPECT_GE, ASSERT_EQ, ::testing::Test

#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr, std::make_unique
#include <tuple>    // std::ignore
#include <vector>   // std::vector

//*************************************************************************************************************************************//
//                                                 GPU CSVM tests depending on nothing                                                 //
//*************************************************************************************************************************************//

template <typename T>
class GenericGPUCSVM : public ::testing::Test,
                       protected util::redirect_output<> { };

TYPED_TEST_SUITE_P(GenericGPUCSVM);

TYPED_TEST_P(GenericGPUCSVM, get_max_work_group_size) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum allowed work-group size should be greater than 0!
    for (std::size_t device_id = 0; device_id < svm.num_available_devices(); ++device_id) {
        EXPECT_GT(svm.get_max_work_group_size(device_id), 0);
    }
}

TYPED_TEST_P(GenericGPUCSVM, get_max_grid_size) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum allowed work-group size should be greater than 0!
    for (std::size_t device_id = 0; device_id < svm.num_available_devices(); ++device_id) {
        const plssvm::detail::dim_type max_grid = svm.get_max_grid_size(device_id);
        EXPECT_GT(max_grid.x, std::size_t{ 0 });
        EXPECT_GT(max_grid.y, std::size_t{ 0 });
        EXPECT_GT(max_grid.z, std::size_t{ 0 });
    }
}

TYPED_TEST_P(GenericGPUCSVM, run_blas_level_3_kernel_explicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_data_points() - 1, num_devices);

    const plssvm::real_type alpha{ 1.0 };
    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data.data());
    const auto B = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const plssvm::real_type beta{ 0.5 };

    // the complete kernel matrix, used to calculate the ground-truth SYMM result
    const plssvm::aos_matrix<plssvm::real_type> full_kernel_matrix = ground_truth::assemble_full_kernel_matrix(params, data.data(), q_red, QA_cost);

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        // check whether the current device is responsible for at least one data point!
        if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        auto &device = svm.devices_[device_id];

        // create kernel matrix to use in the BLAS calculation
        const std::vector<plssvm::real_type> kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data.data(), q_red, QA_cost, *svm.data_distribution_, device_id);
        device_ptr_type A_d{ kernel_matrix.size(), device };
        A_d.copy_to_device(kernel_matrix);

        device_ptr_type B_d{ B.shape(), B.padding(), device };
        B_d.copy_to_device(B);

        plssvm::soa_matrix<plssvm::real_type> C{};
        if (device_id == 0) {
            C = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        } else {
            C = plssvm::soa_matrix<plssvm::real_type>{ plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::real_type{ 0.0 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
        }
        device_ptr_type C_d{ C.shape(), C.padding(), device };
        C_d.copy_to_device(C);

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const unsigned long long num_rhs = B_d.shape().x;
        const unsigned long long num_rows = B_d.shape().y;
        const unsigned long long device_specific_num_rows = svm.data_distribution_->place_specific_num_rows(device_id);
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };
        const unsigned long long num_mirror_rows = num_rows - svm.data_distribution_->place_row_offset(device_id) - device_specific_num_rows;
        const plssvm::detail::dim_type mirror_grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_mirror_rows) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create execution ranges
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };
        const plssvm::detail::execution_range mirror_exec{ block, svm.get_max_work_group_size(device_id), mirror_grid, svm.get_max_grid_size(device_id) };

        // perform BLAS calculation
        svm.run_blas_level_3_kernel_explicit(device_id, exec, mirror_exec, alpha, A_d, B_d, beta, C_d);

        // retrieve data
        plssvm::soa_matrix<plssvm::real_type> C_res{ C.shape(), C.padding() };
        C_d.copy_to_host(C_res);
        C_res.restore_padding();

        // calculate correct results
        plssvm::soa_matrix<plssvm::real_type> correct_C{ C * beta };
        ground_truth::device_specific_gemm(alpha, full_kernel_matrix, B, correct_C, *svm.data_distribution_, device_id);

        // check C for correctness
        EXPECT_FLOATING_POINT_MATRIX_NEAR(C_res, correct_C);
    }
}

TYPED_TEST_P(GenericGPUCSVM, run_w_kernel) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    const std::size_t num_features = data.num_features();
    const std::size_t num_classes = data.num_classes();

    // the weights (i.e., alpha values) for all support vectors
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::rectangular_data_distribution>(data.num_data_points(), num_devices);

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        // check whether the current device is responsible for at least one data point!
        if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        auto &device = svm.devices_[device_id];
        const std::size_t device_specific_num_rows = svm.data_distribution_->place_specific_num_rows(device_id);
        const std::size_t row_offset = svm.data_distribution_->place_row_offset(device_id);

        // upload partial support vectors depending on the current device
        device_ptr_type sv_d{ plssvm::shape{ device_specific_num_rows, num_features }, data.data().padding(), device };
        sv_d.copy_to_device_strided(data.data(), row_offset, device_specific_num_rows);

        // create weights vector on the device
        device_ptr_type weights_d{ weights.shape(), weights.padding(), device };
        weights_d.copy_to_device(weights);

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_features) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create execution range
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };

        // calculate (partial) w on the device
        const device_ptr_type w_d = svm.run_w_kernel(device_id, exec, weights_d, sv_d);

        // check sizes
        plssvm::soa_matrix<plssvm::real_type> w(plssvm::shape{ weights.num_rows(), data.data().num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        ASSERT_EQ(w_d.shape(), w.shape());
        ASSERT_EQ(w_d.padding(), w.padding());
        w_d.copy_to_host(w);
        w.restore_padding();

        // calculate partial ground truth
        const plssvm::soa_matrix<plssvm::real_type> correct_w = ground_truth::calculate_device_specific_w(weights, data.data(), *svm.data_distribution_, device_id);

        // check for correctness
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, correct_w);
    }
}

TYPED_TEST_P(GenericGPUCSVM, run_inplace_matrix_addition) {
    using namespace plssvm::operators;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    const plssvm::soa_matrix<plssvm::real_type> &matr = data.data();

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();

    // calculate correct output
    const plssvm::soa_matrix<plssvm::real_type> correct_result = matr + matr;

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        auto &device = svm.devices_[device_id];

        // move the matrices to the device
        device_ptr_type A_d{ matr.shape(), matr.padding(), device };
        A_d.copy_to_device(matr);
        device_ptr_type B_d{ matr.shape(), matr.padding(), device };
        B_d.copy_to_device(matr);

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const unsigned long long num_rhs = A_d.shape().x;
        const unsigned long long num_rows = A_d.shape().y;
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create execution range
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };

        // add the two matrices to each other
        svm.run_inplace_matrix_addition(device_id, exec, A_d, B_d);

        // copy result back to host
        plssvm::soa_matrix<plssvm::real_type> A{ matr.shape(), matr.padding() };
        A_d.copy_to_host(A);

        // check result for correctness
        EXPECT_FLOATING_POINT_MATRIX_NEAR(A, correct_result);
    }
}

TYPED_TEST_P(GenericGPUCSVM, run_inplace_matrix_scale) {
    using namespace plssvm::operators;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    const plssvm::soa_matrix<plssvm::real_type> &matr = data.data();
    const plssvm::real_type scaling_factor = 3.1415;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();

    // calculate correct output
    const plssvm::soa_matrix<plssvm::real_type> correct_result = matr * scaling_factor;

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        auto &device = svm.devices_[device_id];

        // move the matrices to the device
        device_ptr_type A_d{ matr.shape(), matr.padding(), device };
        A_d.copy_to_device(matr);

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const unsigned long long num_rhs = A_d.shape().x;
        const unsigned long long num_rows = A_d.shape().y;
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create execution range
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };

        // add the two matrices to each other
        svm.run_inplace_matrix_scale(device_id, exec, A_d, scaling_factor);

        // copy result back to host
        plssvm::soa_matrix<plssvm::real_type> A{ matr.shape(), matr.padding() };
        A_d.copy_to_host(A);

        // check result for correctness
        EXPECT_FLOATING_POINT_MATRIX_NEAR(A, correct_result);
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVM,
                            get_max_work_group_size,
                            get_max_grid_size,
                            run_blas_level_3_kernel_explicit,
                            run_w_kernel,
                            run_inplace_matrix_addition,
                            run_inplace_matrix_scale);

//*************************************************************************************************************************************//
//                                        GPU CSVM tests depending on the kernel function type                                         //
//*************************************************************************************************************************************//

template <typename T>
class GenericGPUCSVMKernelFunction : public GenericGPUCSVM<T> { };

TYPED_TEST_SUITE_P(GenericGPUCSVMKernelFunction);

TYPED_TEST_P(GenericGPUCSVMKernelFunction, run_assemble_kernel_matrix_explicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_data_points() - 1, num_devices);

    // perform dimensional reduction
    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data_matr);

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        // check whether the current device is responsible for at least one data point!
        if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        auto &device = svm.devices_[device_id];

        // upload complete A and q_red to each device
        device_ptr_type data_d{ data_matr.shape(), data_matr.padding(), device };
        data_d.copy_to_device(data_matr);

        device_ptr_type q_red_d{ q_red.size() + plssvm::PADDING_SIZE, device };
        q_red_d.copy_to_device(q_red, 0, q_red.size());

        // kernel launch specific sizes
        const unsigned long long num_rows_reduced = data_matr.shape().x;
        const unsigned long long device_specific_num_rows = svm.data_distribution_->place_specific_num_rows(device_id);
        const unsigned long long device_row_offset = svm.data_distribution_->place_row_offset(device_id);

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced - device_row_offset) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create the final execution range
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };

        // calculate the current part of the kernel matrix
        const device_ptr_type kernel_matrix_d = svm.run_assemble_kernel_matrix_explicit(device_id, exec, params, data_d, q_red_d, QA_cost);

        // copy the kernel matrix back to the host
        std::vector<plssvm::real_type> kernel_matrix(kernel_matrix_d.size());
        kernel_matrix_d.copy_to_host(kernel_matrix);

        // calculate ground truth
        const std::vector<plssvm::real_type> correct_kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data_matr, q_red, QA_cost, *svm.data_distribution_, device_id);

        // check for correctness
        ASSERT_EQ(kernel_matrix.size(), correct_kernel_matrix.size());
        EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(kernel_matrix, correct_kernel_matrix, 1e6);
    }
}

TYPED_TEST_P(GenericGPUCSVMKernelFunction, run_assemble_kernel_matrix_implicit_blas_level_3) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_data_points() - 1, num_devices);

    // perform dimensional reduction
    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data_matr);
    const plssvm::real_type alpha{ 1.0 };
    const auto B = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_classes(), data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    // note: beta not applied inside the cg_implicit kernel!

    const plssvm::aos_matrix<plssvm::real_type> full_kernel_matrix = ground_truth::assemble_full_kernel_matrix(params, data_matr, q_red, QA_cost);

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        // check whether the current device is responsible for at least one data point!
        if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        auto &device = svm.devices_[device_id];

        plssvm::soa_matrix<plssvm::real_type> C{ plssvm::shape{ data.num_classes(), data.num_data_points() - 1 }, plssvm::real_type{ 0.0 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

        // upload complete A and q_red to each device
        device_ptr_type data_d{ data_matr.shape(), data_matr.padding(), device };
        data_d.copy_to_device(data_matr);

        device_ptr_type q_red_d{ q_red.size() + plssvm::PADDING_SIZE, device };
        q_red_d.copy_to_device(q_red, 0, q_red.size());

        // upload complete BLAS B and C matrices to each device
        device_ptr_type B_d{ B.shape(), B.padding(), device };
        B_d.copy_to_device(B);
        device_ptr_type C_d{ C.shape(), C.padding(), device };
        C_d.copy_to_device(C);

        // kernel launch specific sizes
        const unsigned long long num_rows_reduced = data_matr.shape().x;
        const unsigned long long device_specific_num_rows = svm.data_distribution_->place_specific_num_rows(device_id);
        const unsigned long long device_row_offset = svm.data_distribution_->place_row_offset(device_id);

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced - device_row_offset) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create the final execution range
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };

        // run the compute kernel
        svm.run_assemble_kernel_matrix_implicit_blas_level_3(device_id, exec, alpha, data_d, params, q_red_d, QA_cost, B_d, C_d);

        // copy results back to host
        C_d.copy_to_host(C);
        C.restore_padding();

        // calculate correct result
        plssvm::soa_matrix<plssvm::real_type> correct_C{ plssvm::shape{ data.num_classes(), data.num_data_points() - 1 }, plssvm::real_type{ 0.0 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
        ground_truth::device_specific_gemm(alpha, full_kernel_matrix, B, correct_C, *svm.data_distribution_, device_id);

        // check for correctness
        EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(C, correct_C, 1e6);
    }
}

TYPED_TEST_P(GenericGPUCSVMKernelFunction, run_predict_kernel) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = 1.0;
    }

    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.num_classes(), data_matr.num_rows() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto predict_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data_matr.num_rows(), data_matr.num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
    const plssvm::soa_matrix<plssvm::real_type> correct_w = ground_truth::calculate_w(weights, data_matr);

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::rectangular_data_distribution>(predict_points.num_rows(), num_devices);

    for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
        SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

        // check whether the current device is responsible for at least one data point!
        if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        auto &device = svm.devices_[device_id];
        const std::size_t device_specific_num_rows = svm.data_distribution_->place_specific_num_rows(device_id);
        const std::size_t row_offset = svm.data_distribution_->place_row_offset(device_id);

        // create support vectors or w vector and load them to the device depending on the used kernel function
        device_ptr_type sv_or_w_d{};
        if constexpr (kernel == plssvm::kernel_function_type::linear) {
            sv_or_w_d = device_ptr_type{ plssvm::shape{ data.num_classes(), data.num_features() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }, device };
            sv_or_w_d.copy_to_device(correct_w);
        } else {
            sv_or_w_d = device_ptr_type{ data_matr.shape(), data_matr.padding(), device };
            sv_or_w_d.copy_to_device(data_matr);
        }

        // load weights to device
        device_ptr_type weights_d{ weights.shape(), weights.padding(), device };
        weights_d.copy_to_device(weights);

        // create predict points
        device_ptr_type predict_points_d{ plssvm::shape{ device_specific_num_rows, data.num_features() }, predict_points.padding(), device };
        predict_points_d.copy_to_device_strided(predict_points, row_offset, device_specific_num_rows);

        // create rho vector
        device_ptr_type rho_d{ rho.size() + plssvm::PADDING_SIZE, device };
        rho_d.copy_to_device(rho, 0, rho.size());
        rho_d.memset(0, rho.size());

        // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
        const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

        // define the full execution grid
        const unsigned long long device_specific_num_predict_points = predict_points_d.shape().x;
        const unsigned long long y_dim_size = params.kernel_type == plssvm::kernel_function_type::linear ? weights_d.shape().x : sv_or_w_d.shape().x;
        const plssvm::detail::dim_type grid{
            static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_predict_points) / static_cast<double>(block.x * plssvm::INTERNAL_BLOCK_SIZE))),
            static_cast<std::size_t>(std::ceil(static_cast<double>(y_dim_size) / static_cast<double>(block.y * plssvm::INTERNAL_BLOCK_SIZE)))
        };

        // create execution range
        const plssvm::detail::execution_range exec{ block, svm.get_max_work_group_size(device_id), grid, svm.get_max_grid_size(device_id) };

        // call predict kernel
        const device_ptr_type out_d = svm.run_predict_kernel(device_id, exec, params, weights_d, rho_d, sv_or_w_d, predict_points_d);

        // check sizes
        plssvm::aos_matrix<plssvm::real_type> out{ plssvm::shape{ device_specific_num_rows, weights.num_rows() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
        ASSERT_EQ(out_d.shape(), out.shape());
        out_d.copy_to_host(out);
        out.restore_padding();

        // check out for correctness
        const plssvm::aos_matrix<plssvm::real_type> correct_out = ground_truth::predict_device_specific_values(params, correct_w, weights, rho, data_matr, predict_points, *svm.data_distribution_, device_id);
        EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(out, correct_out, 1e6);
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVMKernelFunction,
                            run_assemble_kernel_matrix_explicit,
                            run_assemble_kernel_matrix_implicit_blas_level_3,
                            run_predict_kernel);

template <typename T>
class GenericGPUCSVMDeathTest : public GenericGPUCSVM<T> { };

TYPED_TEST_SUITE_P(GenericGPUCSVMDeathTest);

TYPED_TEST_P(GenericGPUCSVMDeathTest, get_max_work_group_size_out_of_range) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();

    // try querying an invalid device_id
    EXPECT_DEATH(std::ignore = svm.get_max_work_group_size(num_devices), fmt::format("Invalid device {} requested!", num_devices));
}

TYPED_TEST_P(GenericGPUCSVMDeathTest, get_max_grid_size_out_of_range) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();

    // try querying an invalid device_id
    EXPECT_DEATH(std::ignore = svm.get_max_grid_size(num_devices), fmt::format("Invalid device {} requested!", num_devices));
}

REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVMDeathTest,
                            get_max_work_group_size_out_of_range,
                            get_max_grid_size_out_of_range);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_GPU_CSVM_TESTS_HPP_
