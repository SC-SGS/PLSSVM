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

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/detail/memory_size.hpp"     // memory size literals
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "backends/compare.hpp"    // compare::{perform_dimensional_reduction, kernel_function, assemble_kernel_matrix_gemm, assemble_kernel_matrix_symm, gemm, calculate_w, predict_values}
#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "types_to_test.hpp"       // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "utility.hpp"             // util::{redirect_output, construct_from_tuple, generate_random_matrix}

#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_GT, EXPECT_GE, ASSERT_EQ, ::testing::Test

#include <cmath>   // std::pow, std::exp
#include <vector>  // std::vector

//*************************************************************************************************************************************//
//                                                 GPU CSVM tests depending on nothing                                                 //
//*************************************************************************************************************************************//

template <typename T>
class GenericGPUCSVM : public ::testing::Test, protected util::redirect_output<> {};
TYPED_TEST_SUITE_P(GenericGPUCSVM);

TYPED_TEST_P(GenericGPUCSVM, get_max_work_group_size) {
    using namespace plssvm::detail::literals;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    EXPECT_GT(svm.get_max_work_group_size(), 0);
}
TYPED_TEST_P(GenericGPUCSVM, num_available_devices) {
    using namespace plssvm::detail::literals;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // test the number of available devices
    ASSERT_EQ(svm.num_available_devices(), svm.devices_.size());
    EXPECT_GE(svm.num_available_devices(), 1);
}

TYPED_TEST_P(GenericGPUCSVM, run_blas_level_3_kernel_explicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // create C-SVM: must be done using the mock class, since solve_lssvm_system_of_linear_equations is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    auto &device = svm.devices_[0];

    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const plssvm::parameter params{};
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    const std::vector<plssvm::real_type> q_red = compare::perform_dimensional_reduction(params, data.data());
    const plssvm::real_type QA_cost = compare::kernel_function(params, data.data(), data.num_data_points() - 1, data.data(), data.num_data_points() - 1);
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> kernel_matrix = compare::assemble_kernel_matrix_gemm(params, data.data(), q_red, QA_cost, plssvm::PADDING_SIZE);
#else
    const std::vector<plssvm::real_type> kernel_matrix = compare::assemble_kernel_matrix_symm(params, data.data(), q_red, QA_cost, plssvm::PADDING_SIZE);
#endif

    device_ptr_type A_d{ kernel_matrix.size(), device };
    A_d.copy_to_device(kernel_matrix);

    const auto B = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(data.num_data_points() - 1, data.num_data_points() - 1, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    device_ptr_type B_d{ B.shape(), B.padding(), device };
    B_d.copy_to_device(B);

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(data.num_data_points() - 1, data.num_data_points() - 1, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    device_ptr_type C_d{ C.shape(), C.padding(), device };
    C_d.copy_to_device(C);

    // perform BLAS calculation
    svm.run_blas_level_3_kernel_explicit(B.num_cols(), B.num_rows(), B.num_cols(), alpha, A_d, B_d, beta, C_d);

    // retrieve data
    plssvm::soa_matrix<plssvm::real_type> C_res{ C.num_rows(), C.num_cols(), C.padding()[0], C.padding()[1] };
    C_d.copy_to_host(C_res);
    C_res.restore_padding();

    // calculate correct results
#if defined(PLSSVM_USE_GEMM)
    compare::gemm(alpha, kernel_matrix, B, beta, C);
#else
    const std::vector<plssvm::real_type> kernel_matrix_gemm = compare::assemble_kernel_matrix_gemm(params, data.data(), q_red, QA_cost, plssvm::PADDING_SIZE);
    compare::gemm(alpha, kernel_matrix_gemm, B, beta, C);
#endif

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C_res, C);
}
TYPED_TEST_P(GenericGPUCSVM, run_w_kernel) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);
    auto &device = svm.devices_[0];

    // create support vectors
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    device_ptr_type sv_d{ data.data().shape(), data.data().padding(),  device };
    sv_d.copy_to_device(data.data());
    // create weights vector
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(3, data.num_data_points(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    device_ptr_type weights_d{ weights.shape(), weights.padding(), device };
    weights_d.copy_to_device(weights);

    // calculate w
    const device_ptr_type w_d = svm.run_w_kernel(weights_d, sv_d);
    plssvm::soa_matrix<plssvm::real_type> w(weights.num_rows(), data.data().num_cols(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // check sizes
    ASSERT_EQ(w_d.extents(), w.shape());
    ASSERT_EQ(w_d.padding(0), w.padding()[0]);
    ASSERT_EQ(w_d.padding(1), w.padding()[1]);
    w_d.copy_to_host(w);
    w.restore_padding();

    // calculate correct w vector
    const plssvm::soa_matrix<plssvm::real_type> correct_w = compare::calculate_w(weights, data.data());

    // check w for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(w, correct_w);
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVM,
                            get_max_work_group_size, num_available_devices,
                            run_blas_level_3_kernel_explicit, run_w_kernel);
// clang-format on

//*************************************************************************************************************************************//
//                                        GPU CSVM tests depending on the kernel function type                                         //
//*************************************************************************************************************************************//

template <typename T>
class GenericGPUCSVMKernelFunction : public GenericGPUCSVM<T> {};
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

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    auto &device = svm.devices_[0];

    device_ptr_type data_d{ data.data().shape(), data.data().padding(), device };
    data_d.copy_to_device(data.data());

    const std::vector<plssvm::real_type> q_red = compare::perform_dimensional_reduction(params, data.data());
    device_ptr_type q_red_d{ q_red.size() + plssvm::PADDING_SIZE, device };
    q_red_d.memset(0, q_red.size());
    q_red_d.copy_to_device(q_red, 0, q_red.size());
    const plssvm::real_type QA_cost = compare::kernel_function(params, data.data(), data.num_data_points() - 1, data.data(), data.num_data_points() - 1);

    const device_ptr_type kernel_matrix_d = svm.run_assemble_kernel_matrix_explicit(params, data_d, q_red_d, QA_cost);

    // get result based on used backend
    std::vector<plssvm::real_type> kernel_matrix(kernel_matrix_d.size());
    kernel_matrix_d.copy_to_host(kernel_matrix);

    // calculate ground truth
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> correct_kernel_matrix = compare::assemble_kernel_matrix_gemm(params, data.data(), q_red, QA_cost, plssvm::PADDING_SIZE);
#else
    const std::vector<plssvm::real_type> correct_kernel_matrix = compare::assemble_kernel_matrix_symm(params, data.data(), q_red, QA_cost, plssvm::PADDING_SIZE);
#endif

    // check for correctness
    ASSERT_EQ(kernel_matrix.size(), correct_kernel_matrix.size());
    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(kernel_matrix, correct_kernel_matrix, 1e4);
}
TYPED_TEST_P(GenericGPUCSVMKernelFunction, run_predict_kernel) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    auto &device = svm.devices_[0];

    // create support vectors
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    device_ptr_type sv_d{ data.data().shape(), data.data().padding(), device };
    sv_d.copy_to_device(data.data());
    // create weights vector
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(3, data.data().num_rows(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    device_ptr_type weights_d{ weights.shape(), weights.padding(), device };
    weights_d.copy_to_device(weights);
    // calculate w if the linear kernel function is used
    device_ptr_type w_d;
    if constexpr (kernel == plssvm::kernel_function_type::linear) {
        w_d = svm.run_w_kernel(weights_d, sv_d);
    }
    // create predict points
    const auto predict_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(data.data().num_rows(), data.data().num_cols(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    device_ptr_type predict_points_d{ predict_points.shape(), predict_points.padding(), device };
    predict_points_d.copy_to_device(predict_points);
    // create rho vector
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
    device_ptr_type rho_d{ rho.size() + plssvm::PADDING_SIZE, device };
    rho_d.copy_to_device(rho, 0, rho.size());
    rho_d.memset(0, rho.size());

    // call predict kernel
    const device_ptr_type out_d = svm.run_predict_kernel(params, w_d, weights_d, rho_d, sv_d, predict_points_d);
    plssvm::aos_matrix<plssvm::real_type> out{ predict_points.num_rows(), weights.num_rows(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };

    // check sizes
    ASSERT_EQ(out_d.extents(), out.shape());
    out_d.copy_to_host(out);
    out.restore_padding();

    // calculate correct predict values
    plssvm::soa_matrix<plssvm::real_type> correct_w;
    if (kernel == plssvm::kernel_function_type::linear) {
        correct_w = compare::calculate_w(weights, data.data());
    }
    const plssvm::aos_matrix<plssvm::real_type> correct_out = compare::predict_values(params, correct_w, weights, rho, data.data(), predict_points);

    // check out for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(out, correct_out);
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVMKernelFunction,
                            run_assemble_kernel_matrix_explicit, run_predict_kernel);
// clang-format on

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_GPU_CSVM_TESTS_HPP_