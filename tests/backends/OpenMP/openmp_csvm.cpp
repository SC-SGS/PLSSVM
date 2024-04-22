/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "plssvm/backend_types.hpp"                                                   // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/OpenMP/csvm.hpp"                                            // plssvm::openmp::csvm
#include "plssvm/backends/OpenMP/exceptions.hpp"                                      // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/kernel/cg_explicit/blas.hpp"                         // plssvm::openmp::device_kernel_symm
#include "plssvm/backends/OpenMP/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::openmp::device_kernel_assembly
#include "plssvm/backends/OpenMP/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::openmp::device_kernel_assembly_symm
#include "plssvm/backends/OpenMP/kernel/predict_kernel.hpp"                           // plssvm::openmp::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                       // plssvm::PADDING_SIZE
#include "plssvm/data_set.hpp"                                                        // plssvm::data_set
#include "plssvm/detail/arithmetic_type_name.hpp"                                     // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/data_distribution.hpp"                                        // plssvm::detail::triangular_data_distribution
#include "plssvm/detail/type_list.hpp"                                                // plssvm::detail::supported_label_types
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                                          // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                                                       // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/shape.hpp"                                                           // plssvm::shape
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform

#include "tests/backends/generic_csvm_tests.hpp"       // generic CSVM tests to instantiate
#include "tests/backends/ground_truth.hpp"             // ground_truth::{perform_dimensional_reduction, assemble_device_specific_kernel_matrix, assemble_full_kernel_matrix, gemm, calculate_w}
#include "tests/backends/OpenMP/mock_openmp_csvm.hpp"  // mock_openmp_csvm
#include "tests/custom_test_macros.hpp"                // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                           // util::redirect_output

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <algorithm>  // std::min
#include <tuple>      // std::make_tuple, std::tuple
#include <vector>     // std::vector

class OpenMPCSVM : public ::testing::Test,
                   private util::redirect_output<> { };

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(OpenMPCSVM, construct_parameter) {
#if defined(PLSSVM_HAS_CPU_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::openmp::csvm{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::parameter{} }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TEST_F(OpenMPCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::automatic, params }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::cpu, params }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

TEST_F(OpenMPCSVM, construct_target_and_named_args) {
#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

struct openmp_csvm_test_type {
    using mock_csvm_type = mock_openmp_csvm;
    using csvm_type = plssvm::openmp::csvm;
    using device_ptr_type = const plssvm::soa_matrix<plssvm::real_type> *;
    inline constexpr static auto additional_arguments = std::make_tuple();
};

using openmp_csvm_test_tuple = std::tuple<openmp_csvm_test_type>;
using openmp_csvm_test_label_type_list = util::cartesian_type_product_t<openmp_csvm_test_tuple, plssvm::detail::supported_label_types>;
using openmp_csvm_test_type_list = util::cartesian_type_product_t<openmp_csvm_test_tuple>;

// the tests used in the instantiated GTest test suites
using openmp_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list>;
using openmp_solver_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list, util::solver_type_list>;
using openmp_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list, util::kernel_function_type_list>;
using openmp_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
using openmp_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_label_type_list, util::kernel_function_and_classification_type_list>;
using openmp_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;

// instantiate type-parameterized tests
// generic CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVM, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMKernelFunction, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolver, openmp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolverKernelFunction, openmp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMKernelFunctionClassification, openmp_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolverKernelFunctionClassification, openmp_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMDeathTest, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMSolverDeathTest, openmp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, openmp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

TEST_F(OpenMPCSVM, blas_level_3_kernel_explicit) {
    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const plssvm::parameter params{};
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data.data());

    // create correct data distribution for the ground truth calculation
    const plssvm::detail::triangular_data_distribution dist{ data.num_data_points() - 1, 1 };
    const std::vector<plssvm::real_type> kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data.data(), q_red, QA_cost, dist, 0);

    const auto B = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    auto ground_truth_C{ C };

    const unsigned long long num_rhs = B.shape().x;
    const unsigned long long num_rows = B.shape().y;
    plssvm::openmp::detail::device_kernel_symm(num_rows, num_rhs, alpha, kernel_matrix, B, beta, C);

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> kernel_matrix_gemm_padded = ground_truth::assemble_full_kernel_matrix(params, data.data(), q_red, QA_cost);
    ground_truth::gemm(alpha, kernel_matrix_gemm_padded, B, beta, ground_truth_C);

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, ground_truth_C);
}

TEST_F(OpenMPCSVM, calculate_w) {
    // the data used for prediction
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    // the weights (i.e., alpha values) for all support vectors
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // calculate w
    plssvm::soa_matrix<plssvm::real_type> w{ plssvm::shape{ weights.num_rows(), data.data().num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    plssvm::openmp::detail::device_kernel_w_linear(w, weights, data.data());

    // calculate correct results
    const plssvm::soa_matrix<plssvm::real_type> correct_w = ground_truth::calculate_w(weights, data.data());

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(w, correct_w);
}

using kernel_function_type_list_gtest = util::combine_test_parameters_gtest_t<util::kernel_function_type_list>;

template <typename T>
class OpenMPCSVMKernelFunction : public OpenMPCSVM { };

TYPED_TEST_SUITE(OpenMPCSVMKernelFunction, kernel_function_type_list_gtest, naming::test_parameter_to_name);

TYPED_TEST(OpenMPCSVMKernelFunction, assemble_kernel_matrix_explicit) {
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

    // create correct data distribution for the ground truth calculation
    const plssvm::detail::triangular_data_distribution dist{ data.num_data_points() - 1, 1 };

    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data_matr);
    const plssvm::real_type cost = plssvm::real_type{ 1.0 } / params.cost;

    std::vector<plssvm::real_type> kernel_matrix(dist.calculate_explicit_kernel_matrix_num_entries_padded(0));  // only explicitly store the upper triangular matrix

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::linear>(q_red, kernel_matrix, data_matr, QA_cost, cost);
            break;
        case plssvm::kernel_function_type::polynomial:
            plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::polynomial>(q_red, kernel_matrix, data_matr, QA_cost, cost, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::rbf:
            plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::rbf>(q_red, kernel_matrix, data_matr, QA_cost, cost, params.gamma.value());
            break;
        case plssvm::kernel_function_type::sigmoid:
            plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::sigmoid>(q_red, kernel_matrix, data_matr, QA_cost, cost, params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::laplacian:
            plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::laplacian>(q_red, kernel_matrix, data_matr, QA_cost, cost, params.gamma.value());
            break;
        case plssvm::kernel_function_type::chi_squared:
            plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::chi_squared>(q_red, kernel_matrix, data_matr, QA_cost, cost, params.gamma.value());
            break;
    }
    const std::vector<plssvm::real_type> correct_kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data_matr, q_red, QA_cost, dist, 0);

    // check for correctness
    ASSERT_EQ(kernel_matrix.size(), correct_kernel_matrix.size());
    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(kernel_matrix, correct_kernel_matrix, 1e6);
}

TYPED_TEST(OpenMPCSVMKernelFunction, blas_level_3_kernel_implicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
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

    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data_matr);
    const plssvm::real_type cost = plssvm::real_type{ 1.0 } / params.cost;

    const auto B = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.num_data_points() - 1, data.num_data_points() - 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    auto ground_truth_C{ C };

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::linear>(alpha, q_red, data_matr, QA_cost, cost, B, beta, C);
            break;
        case plssvm::kernel_function_type::polynomial:
            plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::polynomial>(alpha, q_red, data_matr, QA_cost, cost, B, beta, C, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::rbf:
            plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::rbf>(alpha, q_red, data_matr, QA_cost, cost, B, beta, C, params.gamma.value());
            break;
        case plssvm::kernel_function_type::sigmoid:
            plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::sigmoid>(alpha, q_red, data_matr, QA_cost, cost, B, beta, C, params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::laplacian:
            plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::laplacian>(alpha, q_red, data_matr, QA_cost, cost, B, beta, C, params.gamma.value());
            break;
        case plssvm::kernel_function_type::chi_squared:
            plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::chi_squared>(alpha, q_red, data_matr, QA_cost, cost, B, beta, C, params.gamma.value());
            break;
    }

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> kernel_matrix_gemm_padded = ground_truth::assemble_full_kernel_matrix(params, data_matr, q_red, QA_cost);
    ground_truth::gemm(alpha, kernel_matrix_gemm_padded, B, beta, ground_truth_C);

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, ground_truth_C);
}

TYPED_TEST(OpenMPCSVMKernelFunction, predict_values) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    const plssvm::parameter params{ plssvm::kernel_type = kernel };
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    auto data_matr{ data.data() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        data_matr = util::matrix_abs(data_matr);
    }

    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data_matr.num_rows() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto predict_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data_matr.num_rows(), data_matr.num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
    const plssvm::soa_matrix<plssvm::real_type> correct_w = ground_truth::calculate_w(weights, data_matr);

    plssvm::aos_matrix<plssvm::real_type> out{ plssvm::shape{ predict_points.num_rows(), weights.num_rows() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            plssvm::openmp::detail::device_kernel_predict_linear(out, correct_w, rho, predict_points);
            break;
        case plssvm::kernel_function_type::polynomial:
            plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::polynomial>(out, weights, rho, data_matr, predict_points, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::rbf:
            plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::rbf>(out, weights, rho, data_matr, predict_points, params.gamma.value());
            break;
        case plssvm::kernel_function_type::sigmoid:
            plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::sigmoid>(out, weights, rho, data_matr, predict_points, params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::laplacian:
            plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::laplacian>(out, weights, rho, data_matr, predict_points, params.gamma.value());
            break;
        case plssvm::kernel_function_type::chi_squared:
            plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::chi_squared>(out, weights, rho, data_matr, predict_points, params.gamma.value());
            break;
    }

    // check out for correctness
    const plssvm::aos_matrix<plssvm::real_type> correct_out = ground_truth::predict_values(params, correct_w, weights, rho, data_matr, predict_points);
    EXPECT_FLOATING_POINT_MATRIX_NEAR(out, correct_out);
}

//*************************************************************************************************************************************//
//                                                           CSVM DeathTests                                                           //
//*************************************************************************************************************************************//

class OpenMPCSVMDeathTest : public OpenMPCSVM { };

TEST_F(OpenMPCSVMDeathTest, blas_level_3_kernel_explicit) {
    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const std::vector<plssvm::real_type> kernel_matrix((4 + plssvm::PADDING_SIZE) * (4 + plssvm::PADDING_SIZE + 1) / 2);

    const auto B = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    const unsigned long long num_rhs = B.shape().x;
    const unsigned long long num_rows = B.shape().y;

    // the A matrix must have the correct size
    EXPECT_DEATH(plssvm::openmp::detail::device_kernel_symm(num_rows, num_rows, alpha, std::vector<plssvm::real_type>{}, B, beta, C), fmt::format("A matrix sizes mismatch!: 0 != {}", kernel_matrix.size()));

    // the B matrix must have the correct shape
    const auto B_wrong = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ std::min<std::size_t>(0ULL, num_rows - 1), std::min<std::size_t>(0ULL, num_rhs - 2) }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    EXPECT_DEATH(plssvm::openmp::detail::device_kernel_symm(num_rows, num_rows, alpha, kernel_matrix, B_wrong, beta, C), ::testing::HasSubstr(fmt::format("B matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(num_rows) - 1), std::min(0, static_cast<int>(num_rhs) - 2), num_rows, num_rhs)));

    // the C matrix must have the correct shape
    auto C_wrong = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ std::min<std::size_t>(0ULL, num_rows - 1), std::min<std::size_t>(0ULL, num_rhs - 2) }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    EXPECT_DEATH(plssvm::openmp::detail::device_kernel_symm(num_rows, num_rows, alpha, kernel_matrix, B, beta, C_wrong), ::testing::HasSubstr(fmt::format("C matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(num_rows) - 1), std::min(0, static_cast<int>(num_rhs) - 2), num_rows, num_rhs)));
}

TEST_F(OpenMPCSVMDeathTest, calculate_w) {
    // the data used for prediction
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    // the weights (i.e., alpha values) for all support vectors
    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    plssvm::soa_matrix<plssvm::real_type> w(plssvm::shape{ weights.num_rows(), data.data().num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // the weights and support vector matrix shapes must match
    const auto weights_wrong = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.num_data_points() + 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    EXPECT_DEATH(plssvm::openmp::detail::device_kernel_w_linear(w, weights_wrong, data.data()), fmt::format("Size mismatch: {} vs {}!", weights_wrong.num_cols(), data.data().num_rows()));
    // the w shape must be correct
    plssvm::soa_matrix<plssvm::real_type> w_wrong{};
    EXPECT_DEATH(plssvm::openmp::detail::device_kernel_w_linear(w_wrong, weights, data.data()), ::testing::HasSubstr(fmt::format("Shape mismatch: [0, 0] vs [{}, {}]!", weights.num_rows(), data.data().num_cols())));
}

template <typename T>
class OpenMPCSVMKernelFunctionDeathTest : public OpenMPCSVM { };

TYPED_TEST_SUITE(OpenMPCSVMKernelFunctionDeathTest, kernel_function_type_list_gtest, naming::test_parameter_to_name);

TYPED_TEST(OpenMPCSVMKernelFunctionDeathTest, assemble_kernel_matrix_explicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create correct data for the function call
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    // create correct data distribution for the ground truth calculation
    const plssvm::detail::triangular_data_distribution dist{ data.num_data_points() - 1, 1 };

    const auto [q_red, QA_cost] = ground_truth::perform_dimensional_reduction(params, data.data());

    // create correct data distribution for the ground truth calculation
    std::vector<plssvm::real_type> kernel_matrix(dist.calculate_explicit_kernel_matrix_num_entries_padded(0));  // only explicitly store the upper triangular matrix

    // helper lambda to reduce the amount of needed switches!
    const auto run_assembly = [=](const plssvm::parameter &params_p, const std::vector<plssvm::real_type> &q_red_p, std::vector<plssvm::real_type> &kernel_matrix_p, const plssvm::soa_matrix<plssvm::real_type> &data_p, const plssvm::real_type QA_cost_p) {
        switch (kernel) {
            case plssvm::kernel_function_type::linear:
                plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::linear>(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value());
                break;
            case plssvm::kernel_function_type::polynomial:
                plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::polynomial>(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.degree.value(), params_p.gamma.value(), params_p.coef0.value());
                break;
            case plssvm::kernel_function_type::rbf:
                plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::rbf>(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.gamma.value());
                break;
            case plssvm::kernel_function_type::sigmoid:
                plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::sigmoid>(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.gamma.value(), params_p.coef0.value());
                break;
            case plssvm::kernel_function_type::laplacian:
                plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::laplacian>(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.gamma.value());
                break;
            case plssvm::kernel_function_type::chi_squared:
                plssvm::openmp::detail::device_kernel_assembly<plssvm::kernel_function_type::chi_squared>(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.gamma.value());
                break;
        }
    };

    // check q_red size (must be equal to the number of data points - 1
    EXPECT_DEATH(run_assembly(params, std::vector<plssvm::real_type>{}, kernel_matrix, data.data(), QA_cost), fmt::format("Sizes mismatch!: 0 != {}", data.num_data_points() - 1));

    // check the kernel matrix size (depending on the usage of GEMM/SYMM)
    std::vector<plssvm::real_type> ret;
    EXPECT_DEATH(run_assembly(params, q_red, ret, data.data(), QA_cost), ::testing::HasSubstr(fmt::format("Sizes mismatch (SYMM)!: 0 != {}", kernel_matrix.size())));

    // cost must not be 0.0 since 1.0 / cost is used
    params.cost = plssvm::real_type{ 0.0 };
    EXPECT_DEATH(run_assembly(params, q_red, kernel_matrix, data.data(), QA_cost), "cost must not be 0.0 since it is 1 / plssvm::cost!");
}

TYPED_TEST(OpenMPCSVMKernelFunctionDeathTest, blas_level_3_kernel_implicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create correct data for the function call
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    std::vector<plssvm::real_type> q_red{};
    plssvm::real_type QA_cost{};
    std::tie(q_red, QA_cost) = ground_truth::perform_dimensional_reduction(params, data.data());
    const plssvm::real_type alpha{ 1.0 };
    plssvm::soa_matrix<plssvm::real_type> B{ plssvm::shape{ data.num_classes(), data.num_data_points() - 1 } };
    const plssvm::real_type beta{ 1.0 };
    plssvm::soa_matrix<plssvm::real_type> C{ B };

    // helper lambda to reduce the amount of needed switches!
    const auto run_assembly_symm = [=](const plssvm::parameter &params_p, const std::vector<plssvm::real_type> &q_red_p, const plssvm::soa_matrix<plssvm::real_type> &data_p, const plssvm::soa_matrix<plssvm::real_type> &B_p, plssvm::soa_matrix<plssvm::real_type> &C_p) {
        switch (kernel) {
            case plssvm::kernel_function_type::linear:
                plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::linear>(alpha, q_red_p, data_p, QA_cost, params_p.cost.value(), B_p, beta, C_p);
                break;
            case plssvm::kernel_function_type::polynomial:
                plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::polynomial>(alpha, q_red_p, data_p, QA_cost, params_p.cost.value(), B_p, beta, C_p, params_p.degree.value(), params_p.gamma.value(), params_p.coef0.value());
                break;
            case plssvm::kernel_function_type::rbf:
                plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::rbf>(alpha, q_red_p, data_p, QA_cost, params_p.cost.value(), B_p, beta, C_p, params_p.gamma.value());
                break;
            case plssvm::kernel_function_type::sigmoid:
                plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::sigmoid>(alpha, q_red_p, data_p, QA_cost, params_p.cost.value(), B_p, beta, C_p, params_p.gamma.value(), params_p.coef0.value());
                break;
            case plssvm::kernel_function_type::laplacian:
                plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::laplacian>(alpha, q_red_p, data_p, QA_cost, params_p.cost.value(), B_p, beta, C_p, params_p.gamma.value());
                break;
            case plssvm::kernel_function_type::chi_squared:
                plssvm::openmp::detail::device_kernel_assembly_symm<plssvm::kernel_function_type::chi_squared>(alpha, q_red_p, data_p, QA_cost, params_p.cost.value(), B_p, beta, C_p, params_p.gamma.value());
                break;
        }
    };

    // check q_red size (must be equal to the number of data points - 1
    EXPECT_DEATH(run_assembly_symm(params, std::vector<plssvm::real_type>{}, data.data(), B, C), fmt::format("Sizes mismatch!: 0 != {}", data.num_data_points() - 1));

    // cost must not be 0.0 since 1.0 / cost is used
    plssvm::parameter params2{ params };
    params2.cost = plssvm::real_type{ 0.0 };
    EXPECT_DEATH(run_assembly_symm(params2, q_red, data.data(), B, C), "cost must not be 0.0 since it is 1 / plssvm::cost!");

    // B and C must be of the same shape
    B = plssvm::soa_matrix<plssvm::real_type>{ plssvm::shape{ 1, 1 } };
    EXPECT_DEATH(run_assembly_symm(params, q_red, data.data(), B, C), "The matrices B and C must have the same shape!");

    // the number of columns in B must match the number of rows in the data set - 1
    B = plssvm::soa_matrix<plssvm::real_type>{ plssvm::shape{ data.num_classes(), data.num_data_points() - 2 } };
    C = B;
    EXPECT_DEATH(run_assembly_symm(params, q_red, data.data(), B, C), ::testing::HasSubstr(fmt::format("The number of columns in B ({}) must be the same as the values in q ({})!", B.num_cols(), data.num_data_points() - 1)));
}

TYPED_TEST(OpenMPCSVMKernelFunctionDeathTest, predict_values) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    const plssvm::parameter params{ plssvm::kernel_type = kernel };
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    const auto weights = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.data().num_rows() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto predict_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.data().num_rows(), data.data().num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
    const plssvm::soa_matrix<plssvm::real_type> w = ground_truth::calculate_w(weights, data.data());

    plssvm::aos_matrix<plssvm::real_type> out{ plssvm::shape{ predict_points.num_rows(), weights.num_rows() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    if constexpr (kernel == plssvm::kernel_function_type::linear) {
        // the number of classes must match
        std::vector<plssvm::real_type> rho_wrong = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
        rho_wrong.pop_back();
        EXPECT_DEATH(plssvm::openmp::detail::device_kernel_predict_linear(out, w, rho_wrong, predict_points),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", w.num_rows(), rho_wrong.size())));

        // the number of features must match
        const auto predict_points_wrong = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.data().num_rows(), data.data().num_cols() + 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        EXPECT_DEATH(plssvm::openmp::detail::device_kernel_predict_linear(out, w, rho, predict_points_wrong),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", w.num_cols(), predict_points_wrong.num_cols())));

        // the output shape must match
        plssvm::aos_matrix<plssvm::real_type> out_wrong{};
        EXPECT_DEATH(plssvm::openmp::detail::device_kernel_predict_linear(out_wrong, w, rho, predict_points),
                     ::testing::HasSubstr(fmt::format("Shape mismatch: [0, 0] vs {}!", (plssvm::shape{ predict_points.num_rows(), w.num_rows() }))));
    } else {
        // helper lambda to reduce the amount of needed switches!
        const auto run_predict_values = [=](const plssvm::parameter &params_p, plssvm::aos_matrix<plssvm::real_type> &out_p, const plssvm::aos_matrix<plssvm::real_type> &weights_p, const std::vector<plssvm::real_type> &rho_p, const plssvm::soa_matrix<plssvm::real_type> &support_vectors_p, const plssvm::soa_matrix<plssvm::real_type> &predict_points_p) {
            switch (kernel) {
                case plssvm::kernel_function_type::linear:
                    // unreachable
                    break;
                case plssvm::kernel_function_type::polynomial:
                    plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::polynomial>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, params_p.degree.value(), params_p.gamma.value(), params_p.coef0.value());
                    break;
                case plssvm::kernel_function_type::rbf:
                    plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::rbf>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, params_p.gamma.value());
                    break;
                case plssvm::kernel_function_type::sigmoid:
                    plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::sigmoid>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, params_p.gamma.value(), params_p.coef0.value());
                    break;
                case plssvm::kernel_function_type::laplacian:
                    plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::laplacian>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, params_p.gamma.value());
                    break;
                case plssvm::kernel_function_type::chi_squared:
                    plssvm::openmp::detail::device_kernel_predict<plssvm::kernel_function_type::chi_squared>(out_p, weights_p, rho_p, support_vectors_p, predict_points_p, params_p.gamma.value());
                    break;
            }
        };

        // the number of classes must match
        std::vector<plssvm::real_type> rho_wrong = util::generate_random_vector<plssvm::real_type>(weights.num_rows());
        rho_wrong.pop_back();
        EXPECT_DEATH(run_predict_values(params, out, weights, rho_wrong, data.data(), predict_points),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", w.num_rows(), rho_wrong.size())));

        // the number of support vectors and weights must match
        const auto weights_wrong = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, data.data().num_rows() + 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        EXPECT_DEATH(run_predict_values(params, out, weights_wrong, rho, data.data(), predict_points),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", weights_wrong.num_cols(), data.data().num_rows())));

        // the number of features must match
        const auto predict_points_wrong = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ data.data().num_rows(), data.data().num_cols() + 1 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        EXPECT_DEATH(run_predict_values(params, out, weights, rho, data.data(), predict_points_wrong),
                     ::testing::HasSubstr(fmt::format("Size mismatch: {} vs {}!", data.data().num_cols(), predict_points_wrong.num_cols())));

        // the output shape must match
        plssvm::aos_matrix<plssvm::real_type> out_wrong{};
        EXPECT_DEATH(run_predict_values(params, out_wrong, weights, rho, data.data(), predict_points),
                     ::testing::HasSubstr(fmt::format("Shape mismatch: [0, 0] vs {}!", (plssvm::shape{ predict_points.num_rows(), w.num_rows() }))));
    }
}
