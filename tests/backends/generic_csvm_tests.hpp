/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic tests for all non-GPU backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_
#pragma once

#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/detail/data_distribution.hpp"          // plssvm::detail::{triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/kernel_function_types.hpp"             // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                            // plssvm::aos_matrix
#include "plssvm/mpi/communicator.hpp"                  // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                         // plssvm::parameter
#include "plssvm/shape.hpp"                             // plssvm::shape

#include "tests/backends/ground_truth.hpp"  // ground_truth::{perform_dimensional_reduction, assemble_full_kernel_matrix, assemble_device_specific_kernel_matrix,
                                            // calculate_w, predict_values}
#include "tests/custom_test_macros.hpp"     // EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "tests/types_to_test.hpp"          // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"                // util::{redirect_output, generate_random_matrix}

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_GT, EXPECT_GE, ASSERT_EQ, ::testing::Test

#include <cstddef>  // std::size_t
#include <tuple>    // std::get
#include <vector>   // std::vector

//*************************************************************************************************************************************//
//                                              non-GPU C-SVM tests depending on nothing                                               //
//*************************************************************************************************************************************//

template <typename T>
class GenericBackendCSVM : public ::testing::Test,
                           protected util::redirect_output<> { };

TYPED_TEST_SUITE_P(GenericBackendCSVM);

TYPED_TEST_P(GenericBackendCSVM, blas_level_3_kernel_explicit) {
    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 0.001 } };
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };
    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data.data());

    // emulate two devices to ensure device_kernel_symm_mirror is called
    const std::size_t num_devices = 2;

    // create correct data distribution for the ground truth calculation
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points() - 1, num_devices };

    const auto B = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 });

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 });
    auto ground_truth_C{ C };

    const std::size_t num_rhs = B.shape().x;
    const std::size_t num_rows = B.shape().y;

    plssvm::aos_matrix<plssvm::real_type> C_res{ C.shape(), plssvm::real_type{ 0.0 } };

    for (std::size_t device = 0; device < num_devices; ++device) {
        // create kernel matrix
        const std::vector<plssvm::real_type> kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data.data(), q_red, QA_cost, dist, device);

        plssvm::aos_matrix<plssvm::real_type> C_temp{ C.shape(), plssvm::real_type{ 0.0 } };
        if (device == 0) {
            C_temp = C;
        }

        const std::size_t specific_num_rows = dist.place_specific_num_rows(device);
        const std::size_t row_offset = dist.place_row_offset(device);
        device_kernel_symm(num_rows, num_rhs, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B, beta, C_temp);
        const std::size_t num_mirror_rows = num_rows - row_offset - specific_num_rows;
        if (num_mirror_rows > 0) {
            device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B, beta, C_temp);
        }

        C_res += C_temp;
    }

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> kernel_matrix_gemm = ground_truth::assemble_full_kernel_matrix(params, data.data(), q_red, QA_cost);
    ground_truth::gemm(alpha, kernel_matrix_gemm, B, beta, ground_truth_C);

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C_res, ground_truth_C);
}

TYPED_TEST_P(GenericBackendCSVM, calculate_w) {
    // the data used for prediction
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };

    // the weights (i.e., alpha values) for all support vectors
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() });

    // calculate w
    plssvm::aos_matrix<plssvm::real_type> w{ plssvm::shape{ weights.num_rows(), data.data().num_cols() } };

    // create correct data distribution
    const plssvm::detail::rectangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points(), 1 };

    device_kernel_w_linear(w, weights, data.data(), dist.place_specific_num_rows(0), dist.place_row_offset(0));

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> correct_w = ground_truth::calculate_w(weights, data.data());

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(w, correct_w);
}

REGISTER_TYPED_TEST_SUITE_P(GenericBackendCSVM,
                            blas_level_3_kernel_explicit,
                            calculate_w);

//*************************************************************************************************************************************//
//                                      non-GPU C-SVM tests depending on the kernel function type                                      //
//*************************************************************************************************************************************//

template <typename T>
class GenericBackendCSVMKernelFunction : public GenericBackendCSVM<T> { };

TYPED_TEST_SUITE_P(GenericBackendCSVMKernelFunction);

TYPED_TEST_P(GenericBackendCSVMKernelFunction, assemble_kernel_matrix_explicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    // create correct data distribution for the ground truth calculation
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points() - 1, 1 };

    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data_matr);
    const plssvm::real_type cost = plssvm::real_type{ 1.0 } / params.cost;

    std::vector<plssvm::real_type> kernel_matrix(dist.calculate_explicit_kernel_matrix_num_entries(0));  // only explicitly store the upper triangular matrix

    const std::size_t device_specific_num_rows = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            device_kernel_assembly<plssvm::kernel_function_type::linear>(kernel_matrix.data(), data_matr, device_specific_num_rows, row_offset, q_red, QA_cost, cost);
            break;
        case plssvm::kernel_function_type::polynomial:
            device_kernel_assembly<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>(kernel_matrix.data(), data_matr, device_specific_num_rows, row_offset, q_red, QA_cost, cost, params.degree, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::rbf:
            device_kernel_assembly<plssvm::kernel_function_type::rbf, plssvm::real_type>(kernel_matrix.data(), data_matr, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::sigmoid:
            device_kernel_assembly<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>(kernel_matrix.data(), data_matr, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::laplacian:
            device_kernel_assembly<plssvm::kernel_function_type::laplacian, plssvm::real_type>(kernel_matrix.data(), data_matr, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::chi_squared:
            device_kernel_assembly<plssvm::kernel_function_type::chi_squared, plssvm::real_type>(kernel_matrix.data(), data_matr, device_specific_num_rows, row_offset, q_red, QA_cost, cost, std::get<plssvm::real_type>(params.gamma));
            break;
    }
    const std::vector<plssvm::real_type> correct_kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data_matr, q_red, QA_cost, dist, 0);

    // check for correctness
    ASSERT_EQ(kernel_matrix.size(), correct_kernel_matrix.size());
    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(kernel_matrix, correct_kernel_matrix, 1e6);
}

TYPED_TEST_P(GenericBackendCSVMKernelFunction, blas_level_3_kernel_implicit) {
    using namespace plssvm::operators;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data_matr);
    const plssvm::real_type cost = plssvm::real_type{ 1.0 } / params.cost;

    const auto B = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 });

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 });
    auto ground_truth_C{ C };

    // scale C
    C *= beta;

    // create correct data distribution
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points() - 1, 1 };

    const std::size_t device_specific_num_rows = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            device_kernel_assembly_symm<plssvm::kernel_function_type::linear>(alpha, q_red, data_matr, device_specific_num_rows, row_offset, QA_cost, cost, B, C);
            break;
        case plssvm::kernel_function_type::polynomial:
            device_kernel_assembly_symm<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>(alpha, q_red, data_matr, device_specific_num_rows, row_offset, QA_cost, cost, B, C, params.degree, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::rbf:
            device_kernel_assembly_symm<plssvm::kernel_function_type::rbf, plssvm::real_type>(alpha, q_red, data_matr, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::sigmoid:
            device_kernel_assembly_symm<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>(alpha, q_red, data_matr, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::laplacian:
            device_kernel_assembly_symm<plssvm::kernel_function_type::laplacian, plssvm::real_type>(alpha, q_red, data_matr, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::chi_squared:
            device_kernel_assembly_symm<plssvm::kernel_function_type::chi_squared, plssvm::real_type>(alpha, q_red, data_matr, device_specific_num_rows, row_offset, QA_cost, cost, B, C, std::get<plssvm::real_type>(params.gamma));
            break;
    }

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> kernel_matrix_gemm = ground_truth::assemble_full_kernel_matrix(params, data_matr, q_red, QA_cost);
    ground_truth::gemm(alpha, kernel_matrix_gemm, B, beta, ground_truth_C);

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(C, ground_truth_C, 1e6);
}

TYPED_TEST_P(GenericBackendCSVMKernelFunction, predict_values) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data_matr.num_rows() });
    const auto predict_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data_matr.num_rows(), data_matr.num_cols() });
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
    const plssvm::aos_matrix<plssvm::real_type> correct_w = ground_truth::calculate_w(weights, data_matr);

    plssvm::aos_matrix<plssvm::real_type> out{ plssvm::shape{ predict_points.num_rows(), weights.num_rows() } };

    // create correct data distribution
    const plssvm::detail::rectangular_data_distribution dist{ plssvm::mpi::communicator{}, predict_points.num_rows(), 1 };

    const std::size_t device_specific_num_predict_points = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            device_kernel_predict_linear(out, correct_w, rho, predict_points, device_specific_num_predict_points, row_offset);
            break;
        case plssvm::kernel_function_type::polynomial:
            device_kernel_predict<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>(out, weights, rho, data_matr, predict_points, device_specific_num_predict_points, row_offset, params.degree, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::rbf:
            device_kernel_predict<plssvm::kernel_function_type::rbf, plssvm::real_type>(out, weights, rho, data_matr, predict_points, device_specific_num_predict_points, row_offset, std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::sigmoid:
            device_kernel_predict<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>(out, weights, rho, data_matr, predict_points, device_specific_num_predict_points, row_offset, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::laplacian:
            device_kernel_predict<plssvm::kernel_function_type::laplacian, plssvm::real_type>(out, weights, rho, data_matr, predict_points, device_specific_num_predict_points, row_offset, std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::chi_squared:
            device_kernel_predict<plssvm::kernel_function_type::chi_squared, plssvm::real_type>(out, weights, rho, data_matr, predict_points, device_specific_num_predict_points, row_offset, std::get<plssvm::real_type>(params.gamma));
            break;
    }

    // check out for correctness
    const plssvm::aos_matrix<plssvm::real_type> correct_out = ground_truth::predict_values(params, correct_w, weights, rho, data_matr, predict_points);
    EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(out, correct_out, 1e6);
}

REGISTER_TYPED_TEST_SUITE_P(GenericBackendCSVMKernelFunction,
                            assemble_kernel_matrix_explicit,
                            blas_level_3_kernel_implicit,
                            predict_values);

//*************************************************************************************************************************************//
//                                            non-GPU C-SVM DeathTests depending on nothing                                            //
//*************************************************************************************************************************************//

template <typename T>
class GenericBackendCSVMDeathTest : public GenericBackendCSVM<T> { };

TYPED_TEST_SUITE_P(GenericBackendCSVMDeathTest);

TYPED_TEST_P(GenericBackendCSVMDeathTest, blas_level_3_kernel_explicit) {
    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const std::vector<plssvm::real_type> kernel_matrix(4 * (4 + 1) / 2);

    const auto B = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });

    const std::size_t num_rhs = B.shape().x;
    const std::size_t num_rows = B.shape().y;

    // create correct data distribution
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, num_rows, 1 };
    const std::size_t specific_num_rows = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    {
        // the B matrix must have the correct shape
        const auto B_wrong = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ std::min<std::size_t>(0ULL, num_rows - 1), std::min<std::size_t>(0ULL, num_rhs - 2) });
        EXPECT_DEATH(device_kernel_symm(num_rows, num_rhs, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B_wrong, beta, C), ::testing::HasSubstr(fmt::format("B matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(num_rows) - 1), std::min(0, static_cast<int>(num_rhs) - 2), num_rows, num_rhs)));

        // the C matrix must have the correct shape
        auto C_wrong = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ std::min<std::size_t>(0ULL, num_rows - 1), std::min<std::size_t>(0ULL, num_rhs - 2) });
        EXPECT_DEATH(device_kernel_symm(num_rows, num_rhs, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B, beta, C_wrong), ::testing::HasSubstr(fmt::format("C matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(num_rows) - 1), std::min(0, static_cast<int>(num_rhs) - 2), num_rows, num_rhs)));

        // the place specific number of rows may not be too large
        EXPECT_DEATH(device_kernel_symm(num_rows, num_rhs, num_rows + 1, row_offset, alpha, kernel_matrix.data(), B, beta, C), ::testing::HasSubstr(fmt::format("The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", num_rows + 1, num_rows)));

        // the row offset may not be too large
        EXPECT_DEATH(device_kernel_symm(num_rows, num_rhs, specific_num_rows, num_rows + 1, alpha, kernel_matrix.data(), B, beta, C), ::testing::HasSubstr(fmt::format("The row offset ({}) cannot be greater the the total number of rows ({})!", num_rows + 1, num_rows)));
    }
    {
        const std::size_t num_mirror_rows = num_rows - row_offset - specific_num_rows;

        // the B matrix must have the correct shape
        const auto B_wrong = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ std::min<std::size_t>(0ULL, num_rows - 1), std::min<std::size_t>(0ULL, num_rhs - 2) });
        EXPECT_DEATH(device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B_wrong, beta, C), ::testing::HasSubstr(fmt::format("B matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(num_rows) - 1), std::min(0, static_cast<int>(num_rhs) - 2), num_rows, num_rhs)));

        // the C matrix must have the correct shape
        auto C_wrong = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ std::min<std::size_t>(0ULL, num_rows - 1), std::min<std::size_t>(0ULL, num_rhs - 2) });
        EXPECT_DEATH(device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B, beta, C_wrong), ::testing::HasSubstr(fmt::format("C matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(num_rows) - 1), std::min(0, static_cast<int>(num_rhs) - 2), num_rows, num_rhs)));

        // the place specific number of rows may not be too large
        EXPECT_DEATH(device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, num_rows + 1, row_offset, alpha, kernel_matrix.data(), B, beta, C), ::testing::HasSubstr(fmt::format("The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", num_rows + 1, num_rows)));

        // the mirror number of rows may not be too large
        EXPECT_DEATH(device_kernel_symm_mirror(num_rows, num_rhs, num_rows + 1, specific_num_rows, row_offset, alpha, kernel_matrix.data(), B, beta, C), ::testing::HasSubstr(fmt::format("The number of mirror rows ({}) cannot be greater the the total number of rows ({})!", num_rows + 1, num_rows)));

        // the row offset may not be too large
        EXPECT_DEATH(device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, specific_num_rows, num_rows + 1, alpha, kernel_matrix.data(), B, beta, C), ::testing::HasSubstr(fmt::format("The row offset ({}) cannot be greater the the total number of rows ({})!", num_rows + 1, num_rows)));
    }
}

TYPED_TEST_P(GenericBackendCSVMDeathTest, calculate_w) {
    // the data used for prediction
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };

    // the weights (i.e., alpha values) for all support vectors
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() });
    plssvm::aos_matrix<plssvm::real_type> w(plssvm::shape{ weights.num_rows(), data.data().num_cols() });

    // create correct data distribution
    const plssvm::detail::rectangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points(), 1 };
    const std::size_t specific_num_rows = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    // the weights and support vector matrix shapes must match
    const auto weights_wrong = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() + 1 });
    EXPECT_DEATH(device_kernel_w_linear(w, weights_wrong, data.data(), specific_num_rows, row_offset), fmt::format("Size mismatch: {} vs {}!", weights_wrong.num_cols(), data.data().num_rows()));

    // the w shape must be correct
    plssvm::aos_matrix<plssvm::real_type> w_wrong{};
    EXPECT_DEATH(device_kernel_w_linear(w_wrong, weights, data.data(), specific_num_rows, row_offset), ::testing::HasSubstr(fmt::format("Shape mismatch: [0, 0] vs [{}, {}]!", weights.num_rows(), data.data().num_cols())));

    // the place specific number of rows may not be too large
    EXPECT_DEATH(device_kernel_w_linear(w, weights, data.data(), data.num_data_points() + 1, row_offset), ::testing::HasSubstr(fmt::format("The number of place specific sv ({}) cannot be greater the the total number of sv ({})!", data.num_data_points() + 1, data.num_data_points())));

    // the row offset may not be too large
    EXPECT_DEATH(device_kernel_w_linear(w, weights, data.data(), specific_num_rows, data.num_data_points() + 1), ::testing::HasSubstr(fmt::format("The sv offset ({}) cannot be greater the the total number of sv ({})!", data.num_data_points() + 1, data.num_data_points())));
}

REGISTER_TYPED_TEST_SUITE_P(GenericBackendCSVMDeathTest,
                            blas_level_3_kernel_explicit,
                            calculate_w);

//*************************************************************************************************************************************//
//                                   non-GPU C-SVM DeathTests depending on the kernel function type                                    //
//*************************************************************************************************************************************//

template <typename T>
class GenericBackendCSVMKernelFunctionDeathTest : public GenericBackendCSVMDeathTest<T> { };

TYPED_TEST_SUITE_P(GenericBackendCSVMKernelFunctionDeathTest);

TYPED_TEST_P(GenericBackendCSVMKernelFunctionDeathTest, assemble_kernel_matrix_explicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create correct data for the function call
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };

    // create correct data distribution for the ground truth calculation
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points() - 1, 1 };

    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data.data());

    // create correct data distribution for the ground truth calculation
    std::vector<plssvm::real_type> kernel_matrix(dist.calculate_explicit_kernel_matrix_num_entries(0));  // only explicitly store the upper triangular matrix

    const std::size_t device_specific_num_rows = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    // helper lambda to reduce the amount of needed switches!
    const auto run_assembly = [=](const plssvm::parameter &params_p, plssvm::real_type *kernel_matrix_p, const plssvm::aos_matrix<plssvm::real_type> &data_p, const std::size_t device_specific_num_rows_p, const std::size_t row_offset_p, const std::vector<plssvm::real_type> &q_red_p, const plssvm::real_type QA_cost_p) {
        switch (kernel) {
            case plssvm::kernel_function_type::linear:
                device_kernel_assembly<plssvm::kernel_function_type::linear>(kernel_matrix_p, data_p, device_specific_num_rows_p, row_offset_p, q_red_p, QA_cost_p, params_p.cost);
                break;
            case plssvm::kernel_function_type::polynomial:
                device_kernel_assembly<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>(kernel_matrix_p, data_p, device_specific_num_rows_p, row_offset_p, q_red_p, QA_cost_p, params_p.cost, params_p.degree, std::get<plssvm::real_type>(params_p.gamma), params_p.coef0);
                break;
            case plssvm::kernel_function_type::rbf:
                device_kernel_assembly<plssvm::kernel_function_type::rbf, plssvm::real_type>(kernel_matrix_p, data_p, device_specific_num_rows_p, row_offset_p, q_red_p, QA_cost_p, params_p.cost, std::get<plssvm::real_type>(params_p.gamma));
                break;
            case plssvm::kernel_function_type::sigmoid:
                device_kernel_assembly<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>(kernel_matrix_p, data_p, device_specific_num_rows_p, row_offset_p, q_red_p, QA_cost_p, params_p.cost, std::get<plssvm::real_type>(params_p.gamma), params_p.coef0);
                break;
            case plssvm::kernel_function_type::laplacian:
                device_kernel_assembly<plssvm::kernel_function_type::laplacian, plssvm::real_type>(kernel_matrix_p, data_p, device_specific_num_rows_p, row_offset_p, q_red_p, QA_cost_p, params_p.cost, std::get<plssvm::real_type>(params_p.gamma));
                break;
            case plssvm::kernel_function_type::chi_squared:
                device_kernel_assembly<plssvm::kernel_function_type::chi_squared, plssvm::real_type>(kernel_matrix_p, data_p, device_specific_num_rows_p, row_offset_p, q_red_p, QA_cost_p, params_p.cost, std::get<plssvm::real_type>(params_p.gamma));
                break;
        }
    };

    // check q_red size (must be equal to the number of data points - 1
    EXPECT_DEATH(run_assembly(params, kernel_matrix.data(), data.data(), device_specific_num_rows, row_offset, std::vector<plssvm::real_type>{}, QA_cost), fmt::format("Sizes mismatch!: 0 != {}", data.num_data_points() - 1));

    // the result kernel matrix must point to a valid chunk of memory
    EXPECT_DEATH(run_assembly(params, nullptr, data.data(), device_specific_num_rows, row_offset, q_red, QA_cost), "The kernel matrix result pointer must be valid!");

    // check place specific number of rows
    EXPECT_DEATH(run_assembly(params, kernel_matrix.data(), data.data(), q_red.size() + 1, row_offset, q_red, QA_cost), ::testing::HasSubstr(fmt::format("The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", q_red.size() + 1, q_red.size())));

    // check the row offset
    EXPECT_DEATH(run_assembly(params, kernel_matrix.data(), data.data(), device_specific_num_rows, q_red.size() + 1, q_red, QA_cost), ::testing::HasSubstr(fmt::format("The row offset ({}) cannot be greater the the total number of rows ({})!", q_red.size() + 1, q_red.size())));

    // cost must not be 0.0 since 1.0 / cost is used
    params.cost = plssvm::real_type{ 0.0 };
    EXPECT_DEATH(run_assembly(params, kernel_matrix.data(), data.data(), device_specific_num_rows, row_offset, q_red, QA_cost), "cost must not be 0.0 since it is 1 / plssvm::cost!");
}

TYPED_TEST_P(GenericBackendCSVMKernelFunctionDeathTest, blas_level_3_kernel_implicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create correct data for the function call
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };

    std::vector<plssvm::real_type> q_red{};
    plssvm::real_type QA_cost{};
    std::tie(q_red, QA_cost) = ground_truth::perform_dimensional_reduction(params, data.data());
    const plssvm::real_type alpha{ 1.0 };
    plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ data.num_classes(), data.num_data_points() - 1 } };
    const plssvm::real_type beta{ 1.0 };
    plssvm::aos_matrix<plssvm::real_type> C{ B };

    // scale C
    C *= beta;

    // create correct data distribution
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_data_points() - 1, 1 };

    const std::size_t device_specific_num_rows = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    // helper lambda to reduce the amount of needed switches!
    const auto run_assembly_symm = [=](const plssvm::parameter &params_p, const std::vector<plssvm::real_type> &q_red_p, const plssvm::aos_matrix<plssvm::real_type> &data_p, const std::size_t device_specific_num_rows_p, const std::size_t row_offset_p, const plssvm::aos_matrix<plssvm::real_type> &B_p, plssvm::aos_matrix<plssvm::real_type> &C_p) {
        switch (kernel) {
            case plssvm::kernel_function_type::linear:
                device_kernel_assembly_symm<plssvm::kernel_function_type::linear>(alpha, q_red_p, data_p, device_specific_num_rows_p, row_offset_p, QA_cost, params_p.cost, B_p, C_p);
                break;
            case plssvm::kernel_function_type::polynomial:
                device_kernel_assembly_symm<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>(alpha, q_red_p, data_p, device_specific_num_rows_p, row_offset_p, QA_cost, params_p.cost, B_p, C_p, params_p.degree, std::get<plssvm::real_type>(params_p.gamma), params_p.coef0);
                break;
            case plssvm::kernel_function_type::rbf:
                device_kernel_assembly_symm<plssvm::kernel_function_type::rbf, plssvm::real_type>(alpha, q_red_p, data_p, device_specific_num_rows_p, row_offset_p, QA_cost, params_p.cost, B_p, C_p, std::get<plssvm::real_type>(params_p.gamma));
                break;
            case plssvm::kernel_function_type::sigmoid:
                device_kernel_assembly_symm<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>(alpha, q_red_p, data_p, device_specific_num_rows_p, row_offset_p, QA_cost, params_p.cost, B_p, C_p, std::get<plssvm::real_type>(params_p.gamma), params_p.coef0);
                break;
            case plssvm::kernel_function_type::laplacian:
                device_kernel_assembly_symm<plssvm::kernel_function_type::laplacian, plssvm::real_type>(alpha, q_red_p, data_p, device_specific_num_rows_p, row_offset_p, QA_cost, params_p.cost, B_p, C_p, std::get<plssvm::real_type>(params_p.gamma));
                break;
            case plssvm::kernel_function_type::chi_squared:
                device_kernel_assembly_symm<plssvm::kernel_function_type::chi_squared, plssvm::real_type>(alpha, q_red_p, data_p, device_specific_num_rows_p, row_offset_p, QA_cost, params_p.cost, B_p, C_p, std::get<plssvm::real_type>(params_p.gamma));
                break;
        }
    };

    // check q_red size (must be equal to the number of data points - 1
    EXPECT_DEATH(run_assembly_symm(params, std::vector<plssvm::real_type>{}, data.data(), device_specific_num_rows, row_offset, B, C), fmt::format("Sizes mismatch!: 0 != {}", data.num_data_points() - 1));

    // check place specific number of rows
    EXPECT_DEATH(run_assembly_symm(params, q_red, data.data(), q_red.size() + 1, row_offset, B, C), ::testing::HasSubstr(fmt::format("The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", q_red.size() + 1, q_red.size())));

    // check the row offset
    EXPECT_DEATH(run_assembly_symm(params, q_red, data.data(), device_specific_num_rows, q_red.size() + 1, B, C), ::testing::HasSubstr(fmt::format("The row offset ({}) cannot be greater the the total number of rows ({})!", q_red.size() + 1, q_red.size())));

    // cost must not be 0.0 since 1.0 / cost is used
    plssvm::parameter params2{ params };
    params2.cost = plssvm::real_type{ 0.0 };
    EXPECT_DEATH(run_assembly_symm(params2, q_red, data.data(), device_specific_num_rows, row_offset, B, C), "cost must not be 0.0 since it is 1 / plssvm::cost!");

    // B and C must be of the same shape
    B = plssvm::aos_matrix<plssvm::real_type>{ plssvm::shape{ 1, 1 } };
    EXPECT_DEATH(run_assembly_symm(params, q_red, data.data(), device_specific_num_rows, row_offset, B, C), "The matrices B and C must have the same shape!");

    // the number of columns in B must match the number of rows in the data set - 1
    B = plssvm::aos_matrix<plssvm::real_type>{ plssvm::shape{ data.num_classes(), data.num_data_points() - 2 } };
    C = B;
    EXPECT_DEATH(run_assembly_symm(params, q_red, data.data(), device_specific_num_rows, row_offset, B, C), ::testing::HasSubstr(fmt::format("The number of columns in B ({}) must be the same as the values in q ({})!", B.num_cols(), data.num_data_points() - 1)));
}

TYPED_TEST_P(GenericBackendCSVMKernelFunctionDeathTest, predict_values) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::classification_data_set data{ PLSSVM_CLASSIFICATION_TEST_FILE };

    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.data().num_rows() });
    const auto predict_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.data().num_rows(), data.data().num_cols() });
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
    const plssvm::aos_matrix<plssvm::real_type> w = ground_truth::calculate_w(weights, data.data());

    plssvm::aos_matrix<plssvm::real_type> out{ plssvm::shape{ predict_points.num_rows(), weights.num_rows() }};

    // create correct data distribution
    const plssvm::detail::rectangular_data_distribution dist{ plssvm::mpi::communicator{}, predict_points.num_rows(), 1 };
    const std::size_t device_specific_num_predict_points = dist.place_specific_num_rows(0);
    const std::size_t row_offset = dist.place_row_offset(0);

    if constexpr (kernel == plssvm::kernel_function_type::linear) {
        // the number of classes must match
        std::vector<plssvm::real_type> rho_wrong = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
        rho_wrong.pop_back();
        EXPECT_DEATH(device_kernel_predict_linear(out, w, rho_wrong, predict_points, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", w.num_rows(), rho_wrong.size())));

        // the number of features must match
        const auto predict_points_wrong = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.data().num_rows(), data.data().num_cols() + 1 });
        EXPECT_DEATH(device_kernel_predict_linear(out, w, rho, predict_points_wrong, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", w.num_cols(), predict_points_wrong.num_cols())));

        // the output shape must match
        plssvm::aos_matrix<plssvm::real_type> out_wrong{};
        EXPECT_DEATH(device_kernel_predict_linear(out_wrong, w, rho, predict_points, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Shape mismatch: [0, 0] vs {}!", (plssvm::shape{ predict_points.num_rows(), w.num_rows() }))));

        // the place specific number of rows may not be too large
        EXPECT_DEATH(device_kernel_predict_linear(out, w, rho, predict_points, predict_points.num_rows() + 1, row_offset),
                     ::testing::HasSubstr(fmt::format("The number of place specific predict points ({}) cannot be greater the the total number of predict points ({})!", predict_points.num_rows() + 1, predict_points.num_rows())));

        // the row offset may not be too large
        EXPECT_DEATH(device_kernel_predict_linear(out, w, rho, predict_points, device_specific_num_predict_points, predict_points.num_rows() + 1),
                     ::testing::HasSubstr(fmt::format("The row offset ({}) cannot be greater the the total number of predict points ({})!", predict_points.num_rows() + 1, predict_points.num_rows())));
    } else {
        // helper lambda to reduce the amount of needed switches!
        const auto run_predict_values = [=](const plssvm::parameter &params_p, plssvm::aos_matrix<plssvm::real_type> &out_p, const plssvm::aos_matrix<plssvm::real_type> &weights_p, const std::vector<plssvm::real_type> &rho_p, const plssvm::aos_matrix<plssvm::real_type> &support_vectors_p, const plssvm::aos_matrix<plssvm::real_type> &predict_points_p, const std::size_t device_specific_num_predict_points_p, const std::size_t row_offset_p) {
            switch (kernel) {
                case plssvm::kernel_function_type::linear:
                    // unreachable
                    break;
                case plssvm::kernel_function_type::polynomial:
                    device_kernel_predict<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, device_specific_num_predict_points_p, row_offset_p, params_p.degree, std::get<plssvm::real_type>(params_p.gamma), params_p.coef0);
                    break;
                case plssvm::kernel_function_type::rbf:
                    device_kernel_predict<plssvm::kernel_function_type::rbf, plssvm::real_type>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, device_specific_num_predict_points_p, row_offset_p, std::get<plssvm::real_type>(params_p.gamma));
                    break;
                case plssvm::kernel_function_type::sigmoid:
                    device_kernel_predict<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, device_specific_num_predict_points_p, row_offset_p, std::get<plssvm::real_type>(params_p.gamma), params_p.coef0);
                    break;
                case plssvm::kernel_function_type::laplacian:
                    device_kernel_predict<plssvm::kernel_function_type::laplacian, plssvm::real_type>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, device_specific_num_predict_points_p, row_offset_p, std::get<plssvm::real_type>(params_p.gamma));
                    break;
                case plssvm::kernel_function_type::chi_squared:
                    device_kernel_predict<plssvm::kernel_function_type::chi_squared, plssvm::real_type>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, device_specific_num_predict_points_p, row_offset_p, std::get<plssvm::real_type>(params_p.gamma));
                    break;
            }
        };

        // the number of classes must match
        std::vector<plssvm::real_type> rho_wrong = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
        rho_wrong.pop_back();
        EXPECT_DEATH(run_predict_values(params, out, weights, rho_wrong, data.data(), predict_points, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", w.num_rows(), rho_wrong.size())));

        // the number of support vectors and weights must match
        const auto weights_wrong = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.data().num_rows() + 1 });
        EXPECT_DEATH(run_predict_values(params, out, weights_wrong, rho, data.data(), predict_points, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", weights_wrong.num_cols(), data.data().num_rows())));

        // the number of features must match
        const auto predict_points_wrong = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ data.data().num_rows(), data.data().num_cols() + 1 });
        EXPECT_DEATH(run_predict_values(params, out, weights, rho, data.data(), predict_points_wrong, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", data.data().num_cols(), predict_points_wrong.num_cols())));

        // the output shape must match
        plssvm::aos_matrix<plssvm::real_type> out_wrong{};
        EXPECT_DEATH(run_predict_values(params, out_wrong, weights, rho, data.data(), predict_points, device_specific_num_predict_points, row_offset),
                     ::testing::HasSubstr(fmt::format("Shape mismatch: [0, 0] vs {}!", (plssvm::shape{ predict_points.num_rows(), w.num_rows() }))));

        // the place specific number of rows may not be too large
        EXPECT_DEATH(run_predict_values(params, out, weights, rho, data.data(), predict_points, predict_points.num_rows() + 1, row_offset),
                     ::testing::HasSubstr(fmt::format("The number of place specific predict points ({}) cannot be greater the the total number of predict points ({})!", predict_points.num_rows() + 1, predict_points.num_rows())));

        // the row offset may not be too large
        EXPECT_DEATH(run_predict_values(params, out, weights, rho, data.data(), predict_points, device_specific_num_predict_points, predict_points.num_rows() + 1),
                     ::testing::HasSubstr(fmt::format("The row offset ({}) cannot be greater the the total number of predict points ({})!", predict_points.num_rows() + 1, predict_points.num_rows())));
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericBackendCSVMKernelFunctionDeathTest,
                            assemble_kernel_matrix_explicit,
                            blas_level_3_kernel_implicit,
                            predict_values);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_
