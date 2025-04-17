/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the stdpar backend.
 */

#include "plssvm/backend_types.hpp"                                                   // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/stdpar/csvm.hpp"                                            // plssvm::stdpar::{csvm, csvc, csvr}
#include "plssvm/backends/stdpar/exceptions.hpp"                                      // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/kernel/cg_explicit/blas.hpp"                         // plssvm::stdpar::device_kernel_symm
#include "plssvm/backends/stdpar/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::stdpar::device_kernel_assembly
#include "plssvm/backends/stdpar/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::stdpar::device_kernel_assembly_symm
#include "plssvm/backends/stdpar/kernel/predict_kernel.hpp"                           // plssvm::stdpar::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                       // plssvm::PADDING_SIZE
#include "plssvm/data_set/classification_data_set.hpp"                                // plssvm::classification_data_set
#include "plssvm/detail/arithmetic_type_name.hpp"                                     // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/data_distribution.hpp"                                        // plssvm::detail::triangular_data_distribution
#include "plssvm/detail/type_list.hpp"                                                // plssvm::detail::supported_label_types
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                                          // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                                                       // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/shape.hpp"                                                           // plssvm::shape
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform

#include "tests/backends/generic_base_csvc_tests.hpp"  // generic C-SVC tests to instantiate
#include "tests/backends/generic_base_csvm_tests.hpp"  // generic C-SVM tests to instantiate
#include "tests/backends/generic_base_csvr_tests.hpp"  // generic C-SVR tests to instantiate
#include "tests/backends/ground_truth.hpp"             // ground_truth::{perform_dimensional_reduction, assemble_device_specific_kernel_matrix, assemble_full_kernel_matrix, gemm, calculate_w}
#include "tests/backends/stdpar/mock_stdpar_csvm.hpp"  // mock_stdpar_csvm
#include "tests/custom_test_macros.hpp"                // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                           // util::redirect_output

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <algorithm>  // std::min
#include <cstddef>    // std::size_t
#include <tuple>      // std::make_tuple, std::tuple
#include <vector>     // std::vector

struct stdpar_csvm_test_type {
    using mock_csvm_type = mock_stdpar_csvm;
    using csvm_type = plssvm::stdpar::csvm;
    using csvc_type = plssvm::stdpar::csvc;
    using csvr_type = plssvm::stdpar::csvr;
    using device_ptr_type = const plssvm::soa_matrix<plssvm::real_type> *;
    inline constexpr static auto additional_arguments = std::make_tuple();
};

// a tuple containing the test structs
using stdpar_csvm_test_tuple = std::tuple<stdpar_csvm_test_type>;

// the tests used in the instantiated GTest test suites
// general test types
using stdpar_csvm_test_type_list = util::cartesian_type_product_t<stdpar_csvm_test_tuple>;
using stdpar_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list>;
using stdpar_solver_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list, util::solver_type_list>;
using stdpar_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list, util::kernel_function_type_list>;
using stdpar_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using stdpar_csvm_test_classification_label_type_list = util::cartesian_type_product_t<stdpar_csvm_test_tuple, util::classification_label_types>;
using stdpar_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using stdpar_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using stdpar_csvm_test_regression_label_type_list = util::cartesian_type_product_t<stdpar_csvm_test_tuple, util::regression_label_types>;
using stdpar_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using stdpar_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVM, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVMKernelFunction, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVMSolver, stdpar_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVMSolverKernelFunction, stdpar_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVC, GenericCSVC, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVC, GenericCSVCKernelFunctionClassification, stdpar_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVC, GenericCSVCSolverKernelFunctionClassification, stdpar_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVR, GenericCSVR, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVR, GenericCSVRKernelFunction, stdpar_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVR, GenericCSVRSolverKernelFunction, stdpar_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMDeathTest, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMSolverDeathTest, stdpar_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, stdpar_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// define the exact functions to be used in the generic header
using plssvm::stdpar::detail::device_kernel_assembly;
using plssvm::stdpar::detail::device_kernel_assembly_symm;
using plssvm::stdpar::detail::device_kernel_predict;
using plssvm::stdpar::detail::device_kernel_predict_linear;
using plssvm::stdpar::detail::device_kernel_symm;
using plssvm::stdpar::detail::device_kernel_w_linear;
#include "tests/backends/generic_csvm_tests.hpp"  // generic backend C-SVM tests to instantiate

// generic non-GPU C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericBackendCSVM, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericBackendCSVMKernelFunction, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic non-GPU C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericBackendCSVMDeathTest, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericBackendCSVMKernelFunctionDeathTest, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);
