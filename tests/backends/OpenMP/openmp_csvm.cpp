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
#include "plssvm/backends/OpenMP/csvm.hpp"                                            // plssvm::openmp::{csvm, csvc, csvr}
#include "plssvm/backends/OpenMP/exceptions.hpp"                                      // plssvm::openmp::backend_exception
#include "plssvm/backends/OpenMP/kernel/cg_explicit/blas.hpp"                         // plssvm::openmp::device_kernel_symm
#include "plssvm/backends/OpenMP/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::openmp::device_kernel_assembly
#include "plssvm/backends/OpenMP/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::openmp::device_kernel_assembly_symm
#include "plssvm/backends/OpenMP/kernel/predict_kernel.hpp"                           // plssvm::openmp::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
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
#include "tests/backends/OpenMP/mock_openmp_csvm.hpp"  // mock_openmp_csvm
#include "tests/custom_test_macros.hpp"                // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                           // util::redirect_output

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <algorithm>  // std::min
#include <cstddef>    // std::size_t
#include <tuple>      // std::make_tuple, std::tuple
#include <vector>     // std::vector

using openmp_csvm_types_list = std::tuple<plssvm::openmp::csvc, plssvm::openmp::csvr>;
using openmp_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<openmp_csvm_types_list>>;

template <typename T>
class OpenMPCSVMConstructor : public ::testing::Test,
                              private util::redirect_output<> {
  protected:
    using fixture_csvm_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(OpenMPCSVMConstructor, openmp_csvm_types_gtest, naming::test_parameter_to_name);

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(OpenMPCSVMConstructor, construct_parameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::parameter{} }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(OpenMPCSVMConstructor, construct_target_and_parameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, params }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

TYPED_TEST(OpenMPCSVMConstructor, construct_target_and_named_args) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

struct openmp_csvm_test_type {
    using mock_csvm_type = mock_openmp_csvm;
    using csvm_type = plssvm::openmp::csvm;
    using csvc_type = plssvm::openmp::csvc;
    using csvr_type = plssvm::openmp::csvr;
    using device_ptr_type = const plssvm::soa_matrix<plssvm::real_type> *;
    inline constexpr static auto additional_arguments = std::make_tuple();
};

// a tuple containing the test structs
using openmp_csvm_test_tuple = std::tuple<openmp_csvm_test_type>;

// the tests used in the instantiated GTest test suites
// general test types
using openmp_csvm_test_type_list = util::cartesian_type_product_t<openmp_csvm_test_tuple>;
using openmp_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list>;
using openmp_solver_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list, util::solver_type_list>;
using openmp_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list, util::kernel_function_type_list>;
using openmp_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using openmp_csvm_test_classification_label_type_list = util::cartesian_type_product_t<openmp_csvm_test_tuple, util::classification_label_types>;
using openmp_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using openmp_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using openmp_csvm_test_regression_label_type_list = util::cartesian_type_product_t<openmp_csvm_test_tuple, util::regression_label_types>;
using openmp_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using openmp_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<openmp_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVM, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMKernelFunction, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolver, openmp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericCSVMSolverKernelFunction, openmp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVC, GenericCSVC, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVC, GenericCSVCKernelFunctionClassification, openmp_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVC, GenericCSVCSolverKernelFunctionClassification, openmp_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVR, GenericCSVR, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVR, GenericCSVRKernelFunction, openmp_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVR, GenericCSVRSolverKernelFunction, openmp_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMDeathTest, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMSolverDeathTest, openmp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, openmp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// define the exact functions to be used in the generic header
using plssvm::openmp::detail::device_kernel_assembly;
using plssvm::openmp::detail::device_kernel_assembly_symm;
using plssvm::openmp::detail::device_kernel_predict;
using plssvm::openmp::detail::device_kernel_predict_linear;
using plssvm::openmp::detail::device_kernel_symm;
using plssvm::openmp::detail::device_kernel_w_linear;
#include "tests/backends/generic_csvm_tests.hpp"  // generic backend C-SVM tests to instantiate

// generic non-GPU C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericBackendCSVM, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVM, GenericBackendCSVMKernelFunction, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic non-GPU C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericBackendCSVMDeathTest, openmp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPCSVMDeathTest, GenericBackendCSVMKernelFunctionDeathTest, openmp_kernel_function_type_gtest, naming::test_parameter_to_name);
