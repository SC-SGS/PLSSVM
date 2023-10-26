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

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/memory_size.hpp"     // memory size literals
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "../custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "../types_to_test.hpp"       // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "../utility.hpp"             // util::{redirect_output, construct_from_tuple, flatten, generate_specific_matrix}

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

    const plssvm::real_type alpha{ 1.0 };

#if defined(PLSSVM_USE_GEMM)
    // clang-format off
    std::vector<plssvm::real_type> matr_A{
        plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 },
        plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 },
        plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 }
    };
    // clang-format on
#else
    std::vector<plssvm::real_type> matr_A = { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } };
#endif
    device_ptr_type A_d{ matr_A.size() };
    A_d.copy_to_device(matr_A);

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };
    device_ptr_type B_d{ B.shape() };
    B_d.copy_to_device(B.data());

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(3, 3);
    device_ptr_type C_d{ C.shape() };
    C_d.copy_to_device(C.data());

    // create C-SVM: must be done using the mock class, since solve_lssvm_system_of_linear_equations is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // perform BLAS calculation
    svm.run_blas_level_3_kernel_explicit(B.num_cols(), B.num_rows(), B.num_cols(), alpha, A_d, B_d, beta, C_d);

    // retrieve data
    C_d.copy_to_host(C.data());

    // check C for correctness
    const plssvm::aos_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.45 }, plssvm::real_type{ 6.6 }, plssvm::real_type{ 9.95 } },
                                                             { plssvm::real_type{ 3.75 }, plssvm::real_type{ 15.2 }, plssvm::real_type{ 22.15 } },
                                                             { plssvm::real_type{ 6.05 }, plssvm::real_type{ 23.8 }, plssvm::real_type{ 34.35 } } } };
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
}
TYPED_TEST_P(GenericGPUCSVM, run_w_kernel) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create support vectors
    const auto sv = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(6, 4);
    device_ptr_type sv_d{ sv.shape() };
    sv_d.copy_to_device(sv.data());
    // create weights vector
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 6);
    device_ptr_type weights_d{ weights.shape() };
    weights_d.copy_to_device(weights.data());

    // calculate w
    const device_ptr_type w_d = svm.run_w_kernel(weights_d, sv_d);
    plssvm::aos_matrix<plssvm::real_type> w(2, 4);

    // check sizes
    ASSERT_EQ(w_d.extents(), w.shape());
    w_d.copy_to_host(w.data());

    // check w for correctness
    const plssvm::aos_matrix<plssvm::real_type> correct_w{ { { plssvm::real_type{ 7.21 }, plssvm::real_type{ 7.42 }, plssvm::real_type{ 7.63 }, plssvm::real_type{ 7.84 } },
                                                             { plssvm::real_type{ 22.81 }, plssvm::real_type{ 23.62 }, plssvm::real_type{ 24.43 }, plssvm::real_type{ 25.24 } } } };
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
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.gamma = 1.0 / 3.0;
        params.coef0 = 1.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        params.gamma = 1.0 / 3.0;
    }
    const plssvm::aos_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                        { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                        { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };
    device_ptr_type data_d{ data.shape() };
    data_d.copy_to_device(data.data());

    const std::vector<plssvm::real_type> q_red = { plssvm::real_type{ 3.0 }, plssvm::real_type{ 4.0 } };
    device_ptr_type q_red_d{ q_red.size() };
    q_red_d.copy_to_device(q_red);
    const plssvm::real_type QA_cost{ 2.0 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    const device_ptr_type kernel_matrix_d = svm.run_assemble_kernel_matrix_explicit(params, data_d, q_red_d, QA_cost);

    // get result based on used backend
    std::vector<plssvm::real_type> kernel_matrix(kernel_matrix_d.size());
    kernel_matrix_d.copy_to_host(kernel_matrix);

    // check returned values
#if defined(PLSSVM_USE_GEMM)
    ASSERT_EQ(kernel_matrix.size(), 4);
#else
    ASSERT_EQ(kernel_matrix.size(), 3);
#endif

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
#if defined(PLSSVM_USE_GEMM)
            EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, (std::vector<plssvm::real_type>{ plssvm::real_type{ 11.0 }, plssvm::real_type{ 27.0 }, plssvm::real_type{ 27.0 }, plssvm::real_type{ 72.0 } }));
#else
            EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, (std::vector<plssvm::real_type>{ plssvm::real_type{ 11.0 }, plssvm::real_type{ 27.0 }, plssvm::real_type{ 72.0 } }));
#endif
            break;
        case plssvm::kernel_function_type::polynomial:
#if defined(PLSSVM_USE_GEMM)
            EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, (std::vector<plssvm::real_type>{ plssvm::real_type{ std::pow(14.0 / 3.0 + 1.0, 3) - 3.0 }, plssvm::real_type{ std::pow(32.0 / 3.0 + 1.0, 3) - 5.0 }, plssvm::real_type{ std::pow(32.0 / 3.0 + 1.0, 3) - 5.0 }, plssvm::real_type{ std::pow(77.0 / 3.0 + 1.0, 3) + 1.0 - 6.0 } }));
#else
            EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, (std::vector<plssvm::real_type>{ plssvm::real_type{ std::pow(14.0 / 3.0 + 1.0, 3) - 3.0 }, plssvm::real_type{ std::pow(32.0 / 3.0 + 1.0, 3) - 5.0 }, plssvm::real_type{ std::pow(77.0 / 3.0 + 1.0, 3) + 1.0 - 6.0 } }));
#endif
            break;
        case plssvm::kernel_function_type::rbf:
#if defined(PLSSVM_USE_GEMM)
            EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, (std::vector<plssvm::real_type>{ plssvm::real_type{ -2.0 }, plssvm::real_type{ std::exp(-9.0) - 5.0 }, plssvm::real_type{ std::exp(-9.0) - 5.0 }, plssvm::real_type{ -4.0 } }));
#else
            EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, (std::vector<plssvm::real_type>{ plssvm::real_type{ -2.0 }, plssvm::real_type{ std::exp(-9.0) - 5.0 }, plssvm::real_type{ -4.0 } }));
#endif
            break;
    }
}
TYPED_TEST_P(GenericGPUCSVMKernelFunction, run_predict_kernel) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    const plssvm::parameter params{
        plssvm::kernel_type = kernel,
        plssvm::coef0 = 1.0,
        plssvm::gamma = 1.0 / 3.0
    };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create support vectors
    const auto sv = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(6, 4);
    device_ptr_type sv_d{ sv.shape() };
    sv_d.copy_to_device(sv.data());
    // create weights vector
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 6);
    device_ptr_type weights_d{ weights.shape() };
    weights_d.copy_to_device(weights.data());
    // calculate w if the linear kernel function is used
    device_ptr_type w_d;
    if constexpr (kernel == plssvm::kernel_function_type::linear) {
        w_d = svm.run_w_kernel(weights_d, sv_d);
    }
    // create predict points
    const auto predict_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4);
    device_ptr_type predict_points_d{ predict_points.shape() };
    predict_points_d.copy_to_device(predict_points.data());
    // create rho vector
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 1.5 }, plssvm::real_type{ 2.5 } };
    device_ptr_type rho_d{ rho.size() };
    rho_d.copy_to_device(rho);

    // call predict kernel
    const device_ptr_type out_d = svm.run_predict_kernel(params, w_d, weights_d, rho_d, sv_d, predict_points_d);
    plssvm::aos_matrix<plssvm::real_type> out{ 2, 2 };

    // check sizes
    ASSERT_EQ(out_d.extents(), out.shape());
    out_d.copy_to_host(out.data());

    // check out for correctness
    plssvm::aos_matrix<plssvm::real_type> correct_out;
    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            correct_out = plssvm::aos_matrix<plssvm::real_type>{ { { plssvm::real_type{ 6.13 }, plssvm::real_type{ 21.93 } },
                                                                   { plssvm::real_type{ 36.23 }, plssvm::real_type{ 118.03 } } } };
            break;
        case plssvm::kernel_function_type::polynomial:
            correct_out = plssvm::aos_matrix<plssvm::real_type>{ { { plssvm::real_type{ 24.4910259259259 }, plssvm::real_type{ 78.1270259259259 } },
                                                                   { plssvm::real_type{ 968.441285185185 }, plssvm::real_type{ 2837.80395185185 } } } };
            break;
        case plssvm::kernel_function_type::rbf:
            correct_out = plssvm::aos_matrix<plssvm::real_type>{ { { plssvm::real_type{ -1.3458297294221 }, plssvm::real_type{ -1.07739849655688 } },
                                                                   { plssvm::real_type{ -1.19262689232401 }, plssvm::real_type{ -0.660598521343059 } } } };
            break;
    }
    EXPECT_FLOATING_POINT_MATRIX_NEAR(out, correct_out);
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVMKernelFunction,
                            run_assemble_kernel_matrix_explicit, run_predict_kernel);
// clang-format on

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_GPU_CSVM_TESTS_HPP_