/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "backends/OpenMP/mock_openmp_csvm.hpp"

#include "plssvm/backend_types.hpp"                // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/OpenMP/csvm.hpp"         // plssvm::openmp::csvm
#include "plssvm/backends/OpenMP/exceptions.hpp"   // plssvm::openmp::backend_exception
#include "plssvm/data_set.hpp"                     // plssvm::data_set
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"        // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"             // plssvm::target_platform

#include "plssvm/backends/OpenMP/cg_explicit/blas.hpp"
#include "plssvm/backends/OpenMP/cg_explicit/kernel_matrix_assembly.hpp"

#include "backends/generic_csvm_tests.hpp"      // generic CSVM tests to instantiate
#include "backends/generic_gpu_csvm_tests.hpp"  // generic GPU CSVM tests to instantiate
#include "custom_test_macros.hpp"               // EXPECT_THROW_WHAT
#include "naming.hpp"                           // naming::test_parameter_to_name
#include "types_to_test.hpp"                    // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "utility.hpp"                          // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>   // std::make_tuple, std::tuple
#include <vector>  // std::vector

class OpenMPCSVM : public ::testing::Test, private util::redirect_output<> {};

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
    inline static constexpr auto additional_arguments = std::make_tuple();
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
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolver, openmp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMKernelFunction, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolverKernelFunction, openmp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMKernelFunctionClassification, openmp_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolverKernelFunctionClassification, openmp_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMSolverDeathTest, openmp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);

using kernel_function_type_list_gtest = util::combine_test_parameters_gtest_t<util::kernel_function_type_list>;

template <typename T>
class OpenMPCSVMKernelFunction : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMKernelFunction, kernel_function_type_list_gtest, naming::test_parameter_to_name);

TYPED_TEST(OpenMPCSVMKernelFunction, assemble_kernel_matrix_explicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    const std::vector<plssvm::real_type> q_red = compare::perform_dimensional_reduction(params, data.data());
    const plssvm::real_type QA_cost = compare::kernel_function(params, data.data(), data.num_data_points() - 1, data.data(), data.num_data_points() - 1);

    const std::size_t num_rows_reduced = data.data().num_rows() - 1;
#if defined(PLSSVM_USE_GEMM)
    std::vector<plssvm::real_type> kernel_matrix(num_rows_reduced * num_rows_reduced);  // store full matrix
#else
    std::vector<plssvm::real_type> kernel_matrix(num_rows_reduced * (num_rows_reduced + 1) / 2);  // only explicitly store the upper triangular matrix
#endif

    switch (kernel) {
        case plssvm::kernel_function_type::linear:
            plssvm::openmp::device_kernel_assembly_linear(q_red, kernel_matrix, data.data(), QA_cost, params.cost.value());
            break;
        case plssvm::kernel_function_type::polynomial:
            plssvm::openmp::device_kernel_assembly_polynomial(q_red, kernel_matrix, data.data(), QA_cost, params.cost.value(), params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case plssvm::kernel_function_type::rbf:
            plssvm::openmp::device_kernel_assembly_rbf(q_red, kernel_matrix, data.data(), QA_cost, params.cost.value(), params.gamma.value());
            break;
    }

        // calculate ground truth
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> correct_kernel_matrix = compare::assemble_kernel_matrix_gemm(params, data.data(), q_red, QA_cost, std::size_t{ 0 });
#else
    const std::vector<plssvm::real_type> correct_kernel_matrix = compare::assemble_kernel_matrix_symm(params, data.data(), q_red, QA_cost, std::size_t{ 0 });
#endif

    // check for correctness
    ASSERT_EQ(kernel_matrix.size(), correct_kernel_matrix.size());
    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(kernel_matrix, correct_kernel_matrix, 1e5);
}

TEST_F(OpenMPCSVM, blas_level_3_kernel_explicit) {
    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const plssvm::parameter params{};
    const plssvm::data_set data{ PLSSVM_TEST_FILE };
    const std::vector<plssvm::real_type> q_red = compare::perform_dimensional_reduction(params, data.data());
    const plssvm::real_type QA_cost = compare::kernel_function(params, data.data(), data.num_data_points() - 1, data.data(), data.num_data_points() - 1);
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> kernel_matrix = compare::assemble_kernel_matrix_gemm(params, data.data(), q_red, QA_cost, std::size_t{ 0 });
#else
    const std::vector<plssvm::real_type> kernel_matrix = compare::assemble_kernel_matrix_symm(params, data.data(), q_red, QA_cost, std::size_t{ 0 });
#endif

    const auto B = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(data.num_data_points() - 1, data.num_data_points() - 1, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(data.num_data_points() - 1, data.num_data_points() - 1, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    plssvm::soa_matrix<plssvm::real_type> C_res{ C };

    const auto m = static_cast<unsigned long long>(B.num_cols());
    const auto n = static_cast<unsigned long long>(B.num_rows());
    const auto k = static_cast<unsigned long long>(B.num_cols());
#if defined(PLSSVM_USE_GEMM)
    plssvm::openmp::device_kernel_gemm(m, n, k, alpha, kernel_matrix, B, beta, C_res);
#else
    plssvm::openmp::device_kernel_symm(m, n, k, alpha, kernel_matrix, B, beta, C_res);
#endif

    // calculate correct results
    const std::vector<plssvm::real_type> kernel_matrix_gemm_padded = compare::assemble_kernel_matrix_gemm(params, data.data(), q_red, QA_cost, plssvm::PADDING_SIZE);
    compare::gemm(alpha, kernel_matrix_gemm_padded, B, beta, C);

    // check C for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C_res, C);
}

template <typename T>
class OpenMPCSVMKernelFunctionDeathTest : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMKernelFunctionDeathTest, kernel_function_type_list_gtest, naming::test_parameter_to_name);

TYPED_TEST(OpenMPCSVMKernelFunctionDeathTest, assemble_kernel_matrix_explicit) {
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create correct data for the function call
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 0.001 };
    }
    const plssvm::data_set data{ PLSSVM_TEST_FILE };

    const std::vector<plssvm::real_type> q_red = compare::perform_dimensional_reduction(params, data.data());
    const plssvm::real_type QA_cost = compare::kernel_function(params, data.data(), data.num_data_points() - 1, data.data(), data.num_data_points() - 1);

    const std::size_t num_rows_reduced = data.data().num_rows() - 1;
#if defined(PLSSVM_USE_GEMM)
    std::vector<plssvm::real_type> kernel_matrix(num_rows_reduced * num_rows_reduced);  // store full matrix
#else
    std::vector<plssvm::real_type> kernel_matrix(num_rows_reduced * (num_rows_reduced + 1) / 2);  // only explicitly store the upper triangular matrix
#endif

    // helper lambda to reduce the amount of needed switches!
    const auto run_assembly = [=](const plssvm::parameter &params_p, const std::vector<plssvm::real_type> &q_red_p, std::vector<plssvm::real_type> &kernel_matrix_p, const plssvm::soa_matrix<plssvm::real_type> &data_p, const plssvm::real_type QA_cost_p) {
        switch (kernel) {
            case plssvm::kernel_function_type::linear:
                plssvm::openmp::device_kernel_assembly_linear(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value());
                break;
            case plssvm::kernel_function_type::polynomial:
                plssvm::openmp::device_kernel_assembly_polynomial(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.degree.value(), params_p.gamma.value(), params_p.coef0.value());
                break;
            case plssvm::kernel_function_type::rbf:
                plssvm::openmp::device_kernel_assembly_rbf(q_red_p, kernel_matrix_p, data_p, QA_cost_p, params_p.cost.value(), params_p.gamma.value());
                break;
        }
    };

    // gamma, if used, must not be 0.0
    if constexpr (kernel == plssvm::kernel_function_type::linear) {
        SUCCEED() << "gamma not needed in the linear kernel!";
    } else {
        EXPECT_DEATH(run_assembly(plssvm::parameter{}, q_red, kernel_matrix, data.data(), QA_cost), "gamma must be greater than 0, but is 0!");
    }

    // check q_red size (must be equal to the number of data points - 1
    EXPECT_DEATH(run_assembly(params, std::vector<plssvm::real_type>{}, kernel_matrix, data.data(), QA_cost), fmt::format("Sizes mismatch!: 0 != {}", data.num_data_points() - 1));

    // check the kernel matrix size (depending on the usage of GEMM/SYMM)
    std::vector<plssvm::real_type> ret;
#if defined(PLSSVM_USE_GEMM)
    EXPECT_DEATH(run_assembly(params, q_red, ret, data.data(), QA_cost), ::testing::HasSubstr(fmt::format("Sizes mismatch (GEMM)!: 0 != {}", q_red.size() * q_red.size())));
#else
    EXPECT_DEATH(run_assembly(params, q_red, ret, data.data(), QA_cost), ::testing::HasSubstr(fmt::format("Sizes mismatch (SYMM)!: 0 != {}", q_red.size() * (q_red.size() + 1) / 2)));
#endif

    // cost must not be 0.0 since 1.0 / cost is used
    params.cost = plssvm::real_type{ 0.0 };
    EXPECT_DEATH(run_assembly(params, q_red, kernel_matrix, data.data(), QA_cost), "cost must not be 0.0 since it is 1 / plssvm::cost!");
}

class OpenMPCSVMDeathTest : public OpenMPCSVM {};

TEST_F(OpenMPCSVMDeathTest, blas_level_3_kernel_explicit) {
    const plssvm::real_type alpha{ 1.0 };

    // create kernel matrix to use in the BLAS calculation
    const plssvm::parameter params{};
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> kernel_matrix(4 * 4);
#else
    const std::vector<plssvm::real_type> kernel_matrix(4 * (4 + 1) / 2);
#endif

    const auto B = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    const auto m = static_cast<unsigned long long>(B.num_cols());
    const auto n = static_cast<unsigned long long>(B.num_rows());
    const auto k = static_cast<unsigned long long>(B.num_cols());

    // the A matrix must have the correct size
#if defined(PLSSVM_USE_GEMM)
    EXPECT_DEATH(plssvm::openmp::device_kernel_gemm(m, n, k, alpha, std::vector<plssvm::real_type>{}, B, beta, C), fmt::format("A matrix sizes mismatch!: 0 != {}", B.num_cols() * B.num_cols()));
#else
    EXPECT_DEATH(plssvm::openmp::device_kernel_symm(m, n, k, alpha, std::vector<plssvm::real_type>{}, B, beta, C), fmt::format("A matrix sizes mismatch!: 0 != {}", B.num_cols() * (B.num_cols() + 1) / 2));
#endif

    // the B matrix must have the correct shape
    const auto B_wrong = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(std::min(0, static_cast<int>(n) - 1), std::min(0, static_cast<int>(k) - 2), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
#if defined(PLSSVM_USE_GEMM)
    EXPECT_DEATH(plssvm::openmp::device_kernel_gemm(m, n, k, alpha, kernel_matrix, B_wrong, beta, C), ::testing::HasSubstr(fmt::format("B matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(n) - 1), std::min(0, static_cast<int>(k) - 2), n, k)));
#else
    EXPECT_DEATH(plssvm::openmp::device_kernel_symm(m, n, k, alpha, kernel_matrix, B_wrong, beta, C), ::testing::HasSubstr(fmt::format("B matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(n) - 1), std::min(0, static_cast<int>(k) - 2), n, k)));
#endif

    // the C matrix must have the correct shape
    auto C_wrong = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(std::min(0, static_cast<int>(n) - 1), std::min(0, static_cast<int>(m) - 2), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
#if defined(PLSSVM_USE_GEMM)
    EXPECT_DEATH(plssvm::openmp::device_kernel_gemm(m, n, k, alpha, kernel_matrix, B, beta, C_wrong), ::testing::HasSubstr(fmt::format("C matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(n) - 1), std::min(0, static_cast<int>(m) - 2), n, m)));
#else
    EXPECT_DEATH(plssvm::openmp::device_kernel_symm(m, n, k, alpha, kernel_matrix, B, beta, C_wrong), ::testing::HasSubstr(fmt::format("C matrix sizes mismatch!: [{}, {}] != [{}, {}]", std::min(0, static_cast<int>(n) - 1), std::min(0, static_cast<int>(m) - 2), n, m)));
#endif
}